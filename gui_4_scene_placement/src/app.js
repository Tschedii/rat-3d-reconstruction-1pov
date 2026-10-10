import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';

const API = 'http://127.0.0.1:8973';

// calib-frame vector [x,y,z] where z is "up" -> three.js [x,y,z] where y is up.
// This must be a proper rotation (det +1), not a plain axis swap (det -1,
// i.e. a reflection) - a reflection still places plain positions in a
// plausible-looking spot, but it silently breaks the handedness of the
// right/up/forward basis used to build each camera's rotation matrix,
// which is exactly why the cameras all ended up facing similar/wrong
// directions instead of a proper inward-facing ring.
function remap(v) { return new THREE.Vector3(v[0], v[2], -v[1]); }

const CAM_COLORS = {
  camera_001: 0x5b8def, camera_002: 0x5bcf9e, camera_003: 0xd6c15b,
  camera_004: 0xf96167, camera_005: 0xb87bd9, camera_006: 0xe0824a,
};

let scene, camera, renderer, orbit, transform;
let rigGroup, tablePlane, environmentObject;
const camMarkers = {}; // name -> { holder, persp, helper, pickMesh }
let pickables = []; // {mesh, kind, ref}
let selectedKind = null, selectedRef = null;
let initialRig = null;
let sceneConfig = null;
let rigLocked = false;
let ratsLocked = false;

// rigGroup-local coordinates are expressed relative to the rig's table centre
// (see buildRigGroup), so anything else that arrives in raw calibration-frame
// coordinates has to be shifted by the same amount to line up with the cameras.
let tableCenterLocal = new THREE.Vector3();

// currently-placed rat objects / FOV meshes / eye cameras, keyed by id.
const ratObjects = { extra1: null, extra2: null };
const ratFovMeshes = { extra1: [], extra2: [] };
const ratEyeCams = { extra1: null, extra2: null };
const meshInfoCache = { extra1: null, extra2: null };
let extraEyesCache = null;

// Guards against a stale async mesh load from a previous checkbox state
// re-adding itself after rebuildRatPlacement() has already cleared the scene.
let placementGeneration = 0;

// --- first-person view state ---
let fpvRat = 'extra1';
let fpvEye = 'L';
let fpvActive = false;
let eyePickState = null; // { id, stage: 'L'|'R', picked: {} }

function makeLabelSprite(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(20,20,28,0.78)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = 'bold 30px system-ui, sans-serif';
  ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, sizeAttenuation: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.28, 0.07, 1);
  return sprite;
}

function init(config) {
  sceneConfig = config;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x15140f);

  camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.02, 100);
  camera.position.set(3, 2.5, 3);

  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.getElementById('stage').appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(3, 6, 4);
  scene.add(sun);

  // Tagged so the first-person render can drop them: a rat does not see the
  // editor's scaffolding.
  const axes = new THREE.AxesHelper(0.5); axes.userData.editorOnly = true;
  const grid = new THREE.GridHelper(6, 24, 0x444444, 0x2a2a22); grid.userData.editorOnly = true;
  scene.add(axes, grid);

  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 0.5, 0);
  orbit.update();

  transform = new TransformControls(camera, renderer.domElement);
  transform.setSize(0.9);
  scene.add(transform);
  transform.addEventListener('dragging-changed', (e) => { orbit.enabled = !e.value; });

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('keydown', onKeyDown);

  rigGroup = new THREE.Group();
  rigGroup.name = 'rig';
  scene.add(rigGroup);

  loadEnvironment();
  loadRig();

  animate();
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function basename(p) { return p.split('/').filter(Boolean).pop(); }
function withExt(p, ext) { return p.replace(/\.[^./]+$/, ext); }
function fmtVec(v, digits = 3) { return `(${v.map(x => x.toFixed(digits)).join(', ')})`; }

// A missing or unparseable .mtl must not prevent the .obj from loading. The
// original code nested the OBJ load inside the MTL success callback, so a
// reconstruction exported as geometry-only (no sidecar .mtl) silently never
// appeared at all - the single most likely reason a rat "does not load".
function loadObjMaybeMtl(urlPrefix, objName, onLoad, onFail) {
  const mtlName = withExt(objName, '.mtl');

  const loadObj = (materials) => {
    const objLoader = new OBJLoader();
    if (materials) objLoader.setMaterials(materials);
    objLoader.setPath(urlPrefix);
    objLoader.load(objName, onLoad, undefined, (err) => {
      console.error(`[${objName}] OBJ load failed`, err);
      onFail(`${objName} not found or unreadable (check the path in setup)`);
    });
  };

  const mtlLoader = new MTLLoader();
  mtlLoader.setPath(urlPrefix);
  mtlLoader.load(mtlName, (materials) => {
    try {
      materials.preload();
      loadObj(materials);
    } catch (e) {
      console.warn(`[${mtlName}] preload failed, loading untextured`, e);
      loadObj(null);
    }
  }, undefined, () => {
    console.warn(`[${mtlName}] not found - loading ${objName} untextured`);
    loadObj(null);
  });
}

function loadEnvironment() {
  loadObjMaybeMtl(API + '/env/', basename(sceneConfig.env_mesh_path), (object) => {
    object.name = 'environment';
    environmentObject = object;
    scene.add(object);
    setStatus('environment loaded');
  }, (msg) => setStatus('environment: ' + msg));
}

function buildRigGroup(rig, savedState) {
  const savedCams = savedState && savedState.cameras;
  const tableGeo = new THREE.PlaneGeometry(1.6, 1.6);
  const tableMat = new THREE.MeshBasicMaterial({ color: 0xc9a97e, transparent: true, opacity: 0.28, side: THREE.DoubleSide });
  tablePlane = new THREE.Mesh(tableGeo, tableMat);
  tablePlane.rotation.x = -Math.PI / 2;
  tablePlane.userData.editorOnly = true;
  rigGroup.add(tablePlane);
  pickables.push({ mesh: tablePlane, kind: 'rig', ref: null });

  tableCenterLocal = remap(rig.table_center);
  setCalibToRigFromBasis(rig.display_basis);

  for (const c of rig.cameras) {
    const pos = remap(c.position).sub(tableCenterLocal);
    const fwd = remap(c.forward).normalize();
    const up = remap(c.up).normalize();
    const right = remap(c.right).normalize();

    const holder = new THREE.Object3D();
    holder.position.copy(pos);
    const m = new THREE.Matrix4().makeBasis(right, up, fwd.clone().negate());
    holder.quaternion.setFromRotationMatrix(m);
    rigGroup.add(holder);

    const persp = new THREE.PerspectiveCamera(44, 1920 / 1200, 0.03, 0.22);
    holder.add(persp);
    const color = CAM_COLORS[c.name] || 0x8888ff;
    const helper = new THREE.CameraHelper(persp);
    helper.userData.editorOnly = true;
    helper.setColors ? helper.setColors(new THREE.Color(color), new THREE.Color(color), new THREE.Color(color), new THREE.Color(color), new THREE.Color(color)) : null;
    scene.add(helper);

    const pickGeo = new THREE.SphereGeometry(0.035, 12, 12);
    const pickMat = new THREE.MeshBasicMaterial({ color });
    const pickMesh = new THREE.Mesh(pickGeo, pickMat);
    pickMesh.userData.editorOnly = true;
    holder.add(pickMesh);

    const sprite = makeLabelSprite(c.name.replace('camera_', 'cam '), color);
    sprite.position.set(0, 0.09, 0);
    sprite.userData.editorOnly = true;
    holder.add(sprite);

    // calibPos is where the calibration itself puts this camera (rigGroup-local),
    // kept so rats can follow any later per-camera adjustment (see rigAdjustmentFit).
    camMarkers[c.name] = { holder, persp, helper, pickMesh, calibPos: pos.clone() };
    pickables.push({ mesh: pickMesh, kind: 'camera', ref: c.name });
  }

  if (savedCams && savedCams.group) {
    rigGroup.position.fromArray(savedCams.group.position);
    rigGroup.quaternion.fromArray(savedCams.group.quaternion);
    rigGroup.scale.fromArray(savedCams.group.scale);
  } else {
    rigGroup.position.set(0, -0.7, 0);
  }
  if (savedCams && savedCams.perCamera) {
    for (const name in savedCams.perCamera) {
      const m = camMarkers[name]; if (!m) continue;
      m.holder.position.fromArray(savedCams.perCamera[name].position);
      m.holder.quaternion.fromArray(savedCams.perCamera[name].quaternion);
    }
  }
  buildSidebarCameraButtons();
}

let savedRatTransforms = {};

function loadRig() {
  const withState = (state) => {
    savedRatTransforms = (state && state.rats) || {};
    fetch(API + '/api/initial_rig').then(r => r.json()).then(rig => {
      if (rig.error) { setStatus('rig error: ' + rig.error); return; }
      initialRig = rig;
      buildRigGroup(rig, state);
      refreshMeshInfo();
    });
  };
  fetch(API + '/api/state').then(r => r.json()).then(withState).catch(() => withState(null));
}

// A rat mesh.obj comes out of the reconstruction in the calibration's own
// world frame and in real metres, the same frame the camera poses live in -
// so the rat-to-rig transform is fully determined and does not need a manual
// fit. It is the composition of two rotations plus one translation:
//
//   1. calibration frame -> table-aligned display frame (the rig's own
//      display_basis, computed server-side from the average viewing
//      direction). The camera poses arrive already in this frame; a mesh does
//      not, which is the step that is easy to miss - skipping it leaves the
//      animal metres away from the rig.
//   2. display frame (z up) -> three.js (y up), the same proper rotation the
//      camera positions go through in remap().
//   3. translate by -tableCenter, because rigGroup-local coordinates are
//      measured from the table centre.
//
// Parenting the rat under rigGroup then makes it inherit whatever coarse fit
// the rig was placed with.
const REMAP_MATRIX = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, 0, 1, 0,
  0, -1, 0, 0,
  0, 0, 0, 1,
);
let calibToRigQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

function setCalibToRigFromBasis(basis) {
  if (!basis || basis.length !== 3) return;
  const [bx, by, bz] = basis;
  // rows are the display basis vectors, so this maps calib -> display
  const toDisplay = new THREE.Matrix4().set(
    bx[0], bx[1], bx[2], 0,
    by[0], by[1], by[2], 0,
    bz[0], bz[1], bz[2], 0,
    0, 0, 0, 1,
  );
  calibToRigQuat = new THREE.Quaternion().setFromRotationMatrix(
    REMAP_MATRIX.clone().multiply(toDisplay),
  );
}

// --- FOV cone visualization, ported from batch_eye_marker.html's own math
// (buildFovCap / eyeAxis / FOV_HALF_ANGLE=100deg, az=20deg, el=25deg). ---
const FOV_HALF_ANGLE = 100 * Math.PI / 180;
const EYE_AZ = 20, EYE_EL = 25;

function eyeAxis(lateral, azimuthDeg, elevationDeg, forward, up) {
  const az = azimuthDeg * Math.PI / 180, el = elevationDeg * Math.PI / 180;
  const horiz = lateral.clone().multiplyScalar(Math.cos(az)).addScaledVector(forward, Math.sin(az));
  return horiz.multiplyScalar(Math.cos(el)).addScaledVector(up, Math.sin(el)).normalize();
}

function buildFovCapGeometry(center, axis, halfAngleRad, radius, segsTheta, segsPhi) {
  axis = axis.clone().normalize();
  const arbitrary = Math.abs(axis.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
  const u = new THREE.Vector3().crossVectors(arbitrary, axis).normalize();
  const v = new THREE.Vector3().crossVectors(axis, u);
  const positions = [];
  for (let i = 0; i <= segsTheta; i++) {
    const theta = halfAngleRad * i / segsTheta;
    const ct = Math.cos(theta), st = Math.sin(theta);
    for (let j = 0; j <= segsPhi; j++) {
      const phi = 2 * Math.PI * j / segsPhi;
      const ring = u.clone().multiplyScalar(Math.cos(phi)).addScaledVector(v, Math.sin(phi));
      const dir = axis.clone().multiplyScalar(ct).addScaledVector(ring, st);
      const p = center.clone().addScaledVector(dir, radius);
      positions.push(p.x, p.y, p.z);
    }
  }
  const indices = [];
  for (let i = 0; i < segsTheta; i++) {
    for (let j = 0; j < segsPhi; j++) {
      const a = i * (segsPhi + 1) + j, b = a + segsPhi + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

function placementMode() {
  const el = document.querySelector('input[name="placeMode"]:checked');
  return el ? el.value : 'world';
}

// The transform that puts a mesh still carrying its reconstruction-frame
// coordinates in the right place relative to the cameras. In 'split' mode the
// mesh is instead dropped on the table origin with a display-only sideways
// offset, which is the right choice when the meshes have been re-centred on
// their own origin by a later step (the mesh-info panel says which it is).
//
// In 'world' mode the mesh also follows the per-camera fine-tuning: the
// calibration fixes the rat relative to the cameras as calibrated, so if the
// cameras were then moved individually (say the whole ring turned 180 deg to
// match the room), the rat has to undergo the same rigid motion or it ends up
// seen from the wrong side.
function ratBaseTransform(displayOffsetXZ) {
  if (placementMode() === 'world') {
    const fit = rigAdjustmentFit();
    const position = tableCenterLocal.clone().negate().applyQuaternion(fit.quaternion).add(fit.translation);
    const quaternion = fit.quaternion.clone().multiply(calibToRigQuat);
    return { position, quaternion };
  }
  return {
    position: new THREE.Vector3(displayOffsetXZ[0], 0, displayOffsetXZ[1]),
    quaternion: calibToRigQuat.clone(),
  };
}

function loadExtraRat(id, meshPath, displayOffsetXZ, fovEnabled, generation) {
  loadObjMaybeMtl(`${API}/${id}/`, basename(meshPath), (object) => {
    if (generation !== placementGeneration) return; // superseded while loading
    object.name = id;
    const base = ratBaseTransform(displayOffsetXZ);
    object.position.copy(base.position);
    object.quaternion.copy(base.quaternion);
    object.scale.set(1, 1, 1);

    // A previously saved manual adjustment wins over the computed default, so
    // a confirmed placement survives a reload.
    const saved = savedRatTransforms[id];
    if (saved) {
      object.position.fromArray(saved.position);
      object.quaternion.fromArray(saved.quaternion);
      if (saved.scale) object.scale.fromArray(saved.scale);
    }

    rigGroup.add(object);
    pickables.push({ mesh: object, kind: 'extraRat', ref: id });
    ratObjects[id] = object;
    updateRatButtons();

    if (!fovEnabled) { setStatus(`${id}: mesh loaded`); return; }

    const box = new THREE.Box3().setFromObject(object);
    const fovRadius = box.getSize(new THREE.Vector3()).length() * 0.35;

    fetchExtraEyes().then(eyes => {
      if (generation !== placementGeneration) return;
      const e = eyes[id];
      if (!e) {
        setStatus(`${id}: mesh loaded, no eye positions yet (mark them below)`);
        updateFovAvailability();
        return;
      }
      addFovForRat(id, e, fovRadius);
      setStatus(`${id}: mesh + field-of-view loaded`);
      updateFovAvailability();
    });
  }, (msg) => setStatus(`${id}: ${msg}`));
}

function fetchExtraEyes(force) {
  if (extraEyesCache && !force) return Promise.resolve(extraEyesCache);
  return fetch(`${API}/api/extra_eyes`).then(r => r.json()).then(eyes => { extraEyesCache = eyes; return eyes; });
}

// Eye positions come from the eye-marking stage in the mesh's own local frame,
// so the FOV geometry is built in that frame and parented to the rat object.
// It therefore follows the rat when the rat is moved, instead of being baked
// into rigGroup coordinates at load time.
function addFovForRat(id, eyeData, fovRadius) {
  const object = ratObjects[id];
  if (!object) return;

  const eyeL = new THREE.Vector3(...eyeData.eyeL);
  const eyeR = new THREE.Vector3(...eyeData.eyeR);
  let headForward = new THREE.Vector3(...(eyeData.headForward || [1, 0, 0])).normalize();
  const headUp = new THREE.Vector3(...(eyeData.headUp || [0, 0, 1])).normalize();

  const headRight = eyeL.clone().sub(eyeR).normalize();
  const fRaw = headForward.clone().addScaledVector(headRight, -headForward.dot(headRight));
  if (fRaw.length() > 1e-6) headForward = fRaw.normalize();

  const axisL = eyeAxis(headRight, EYE_AZ, EYE_EL, headForward, headUp);
  const axisR = eyeAxis(headRight.clone().negate(), EYE_AZ, EYE_EL, headForward, headUp);

  const capMat = (color) => new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.3, depthWrite: false, side: THREE.DoubleSide,
  });
  const dotMat = (color) => new THREE.MeshBasicMaterial({ color });

  const capL = new THREE.Mesh(buildFovCapGeometry(eyeL, axisL, FOV_HALF_ANGLE, fovRadius, 26, 36), capMat(0x40a2ff));
  const capR = new THREE.Mesh(buildFovCapGeometry(eyeR, axisR, FOV_HALF_ANGLE, fovRadius, 26, 36), capMat(0xff8026));
  const dotL = new THREE.Mesh(new THREE.SphereGeometry(fovRadius * 0.05, 12, 12), dotMat(0x59a8ff));
  dotL.position.copy(eyeL);
  const dotR = new THREE.Mesh(new THREE.SphereGeometry(fovRadius * 0.05, 12, 12), dotMat(0xff9452));
  dotR.position.copy(eyeR);
  for (const m of [capL, capR, dotL, dotR]) m.userData.editorOnly = true;

  object.add(capL, capR, dotL, dotR);
  ratFovMeshes[id] = [capL, capR, dotL, dotR];

  buildEyeCameras(id, { eyeL, eyeR, axisL, axisR, headForward, headUp });
}

// One child camera per eye plus a "cyclopean" camera on the head-forward axis.
// Parenting them to the rat object means they inherit the rat's placement in
// the scene, so the first-person render needs no transform bookkeeping of its
// own.
function buildEyeCameras(id, basis) {
  const object = ratObjects[id];
  if (!object) return;
  if (ratEyeCams[id]) for (const c of Object.values(ratEyeCams[id])) object.remove(c);

  const mid = basis.eyeL.clone().add(basis.eyeR).multiplyScalar(0.5);
  const make = (pos, axis) => {
    const cam = new THREE.PerspectiveCamera(120, 4 / 3, 0.005, 60);
    cam.position.copy(pos);
    // three.js cameras look down -Z, so the camera's +Z is the negated view axis.
    const zc = axis.clone().normalize().negate();
    let xc = new THREE.Vector3().crossVectors(basis.headUp, zc);
    if (xc.lengthSq() < 1e-8) xc = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), zc);
    xc.normalize();
    const yc = new THREE.Vector3().crossVectors(zc, xc).normalize();
    cam.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(xc, yc, zc));
    object.add(cam);
    return cam;
  };

  ratEyeCams[id] = {
    L: make(basis.eyeL, basis.axisL),
    R: make(basis.eyeR, basis.axisR),
    C: make(mid, basis.headForward),
  };
}

function clearRat(id) {
  if (ratObjects[id]) {
    rigGroup.remove(ratObjects[id]);
    pickables = pickables.filter(p => p.ref !== id);
    ratObjects[id] = null;
  }
  ratFovMeshes[id] = [];
  ratEyeCams[id] = null;
}

// Rebuilds rat placement from scratch whenever the checkboxes change, rather
// than trying to incrementally reposition already-placed meshes - simplest
// correct way to keep the placement mode consistent, and cheap for two small
// meshes. The generation counter makes a slow load from a superseded state
// drop itself instead of reappearing.
function rebuildRatPlacement() {
  const wantObj1 = document.getElementById('chkObj1').checked;
  const wantObj2 = document.getElementById('chkObj2').checked;
  const wantFov = document.getElementById('chkFov').checked;

  placementGeneration++;
  const generation = placementGeneration;
  clearRat('extra1');
  clearRat('extra2');
  updateRatButtons();

  const active = [];
  if (wantObj1 && sceneConfig.rat1_mesh_path) active.push('extra1');
  if (wantObj2 && sceneConfig.rat2_mesh_path) active.push('extra2');

  if (wantObj1 && !sceneConfig.rat1_mesh_path) setStatus('object 1: no mesh path configured');
  if (wantObj2 && !sceneConfig.rat2_mesh_path) setStatus('object 2: no mesh path configured');

  active.forEach((id, i) => {
    let offset;
    if (active.length === 2) offset = i === 0 ? [-0.22, 0] : [0.22, 0];
    else offset = [0, 0];
    const meshPath = id === 'extra1' ? sceneConfig.rat1_mesh_path : sceneConfig.rat2_mesh_path;
    loadExtraRat(id, meshPath, offset, wantFov, generation);
  });

  if (!active.length) setStatus('no meshes placed');
  updateFovAvailability();
}

// =============================================================================
// Mesh provenance readout
//
// Answers, from the actual configured files rather than from an assumption,
// whether a mesh still sits in reconstruction-frame world coordinates (in
// which case the two animals' relative placement is real and must not be
// overridden) or has been re-centred on its own origin (in which case only
// the size is meaningful and the placement is manual).
// =============================================================================

function refreshMeshInfo() {
  const box = document.getElementById('meshInfo');
  if (!box) return;
  const ids = [['extra1', 'rat1_mesh_path'], ['extra2', 'rat2_mesh_path']];
  Promise.all(ids.map(([id]) => fetch(`${API}/api/mesh_info?which=${id}`).then(r => r.json()).catch(() => null)))
    .then(infos => {
      const lines = [];
      infos.forEach((info, i) => {
        const id = ids[i][0];
        meshInfoCache[id] = info;
        const label = id === 'extra1' ? 'rat 1' : 'rat 2';
        if (!info || !info.configured) { lines.push(`${label}: not configured`); return; }
        if (!info.exists) { lines.push(`${label}: FILE NOT FOUND<br>${info.path}`); return; }
        const size = info.size.map(x => (x * 100).toFixed(1)).join(' x ');
        lines.push(`${label}: ${info.vertex_count} verts, ${size} cm` + (info.has_mtl ? '' : ', no .mtl (untextured)'));
        if (info.table_center) {
          // table_center is reported in the display frame; the mesh centroid is
          // in the raw calibration frame, so compare distances from origin
          // rather than componentwise.
          const c = info.bbox_center;
          const dOrigin = Math.hypot(c[0], c[1], c[2]);
          lines.push(`&nbsp;&nbsp;centre ${fmtVec(c)} m, |c| = ${dOrigin.toFixed(3)} m`);
          lines.push(`&nbsp;&nbsp;${dOrigin < 0.05 ? 'looks RE-CENTRED on its own origin' : 'looks like world coordinates'}`);
        }
      });

      const f1 = frameOf(sceneConfig.rat1_mesh_path), f2 = frameOf(sceneConfig.rat2_mesh_path);
      if (f1 && f2 && f1 !== f2) {
        lines.push(`<span style="color:#e0a34a">warning: rat 1 is frame ${f1}, rat 2 is frame ${f2}. Different instants, so their relative pose is not a real simultaneous configuration.</span>`);
      }
      box.innerHTML = lines.join('<br>');
    });
}

function frameOf(path) {
  if (!path) return null;
  const m = path.match(/frame_(\d+)/);
  return m ? m[1] : null;
}

function select(kind, ref) {
  if (rigLocked && (kind === 'rig' || kind === 'camera')) return;
  if (ratsLocked && kind === 'extraRat') return;
  selectedKind = kind; selectedRef = ref;
  let obj = null;
  if (kind === 'rig') obj = rigGroup;
  else if (kind === 'camera') obj = camMarkers[ref].holder;
  else if (kind === 'extraRat') obj = ratObjects[ref];
  if (obj) { transform.attach(obj); } else { transform.detach(); }
  updateSidebarSelection();
}

function onPointerDown(ev) {
  if (transform.dragging) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(ndc, camera);

  if (eyePickState) { handleEyePick(raycaster); return; }

  const meshes = pickables.map(p => p.mesh);
  const hits = raycaster.intersectObjects(meshes, true);
  if (hits.length === 0) return;
  let hitObj = hits[0].object;
  const found = pickables.find(p => p.mesh === hitObj || (p.mesh.getObjectById && p.mesh.getObjectById(hitObj.id)) || isDescendant(p.mesh, hitObj));
  if (found) select(found.kind, found.ref);
}
function isDescendant(root, node) {
  let p = node;
  while (p) { if (p === root) return true; p = p.parent; }
  return false;
}

function onKeyDown(ev) {
  if (ev.key === 'Escape' && fpvActive) { exitFpv(); return; }
  if (ev.key === 'Escape' && eyePickState) { eyePickState = null; setStatus('eye marking cancelled'); return; }
  if (ev.key === 't' || ev.key === 'T') transform.setMode('translate');
  if (ev.key === 'r' || ev.key === 'R') transform.setMode('rotate');
  if (ev.key === 's' || ev.key === 'S') transform.setMode('scale');
  if (ev.key >= '1' && ev.key <= '6') {
    const names = Object.keys(camMarkers);
    const idx = parseInt(ev.key, 10) - 1;
    if (names[idx]) select('camera', names[idx]);
  }
  if (ev.key === '0') select('rig', null);
}

function buildSidebarCameraButtons() {
  const box = document.getElementById('camList');
  box.innerHTML = '';
  Object.keys(camMarkers).forEach((name, i) => {
    const btn = document.createElement('button');
    btn.textContent = name.replace('camera_', 'cam ') + ` (${i + 1})`;
    btn.dataset.kind = 'camera'; btn.dataset.ref = name;
    btn.style.borderLeft = `4px solid #${(CAM_COLORS[name] || 0x8888ff).toString(16).padStart(6, '0')}`;
    btn.addEventListener('click', () => select('camera', name));
    box.appendChild(btn);
  });
}

function updateSidebarSelection() {
  document.querySelectorAll('#panel button[data-kind]').forEach(b => {
    const match = b.dataset.kind === selectedKind &&
      ((b.dataset.kind !== 'camera' && b.dataset.kind !== 'extraRat') || b.dataset.ref === selectedRef);
    b.classList.toggle('active', match);
  });
  document.getElementById('selLabel').textContent =
    selectedKind === 'rig' ? 'Rig (all cameras)' :
    selectedKind === 'camera' ? selectedRef.replace('camera_', 'cam ') :
    selectedKind === 'extraRat' ? (selectedRef === 'extra1' ? 'rat 1' : 'rat 2') : 'none';
}

function updateRatButtons() {
  document.getElementById('selRat1').disabled = !ratObjects.extra1 || ratsLocked;
  document.getElementById('selRat2').disabled = !ratObjects.extra2 || ratsLocked;
  document.getElementById('confirmRatsBtn').disabled =
    ratsLocked || (!ratObjects.extra1 && !ratObjects.extra2);
}

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
  return true;
}

function confirmCameraPosition() {
  rigLocked = true;
  transform.detach();
  selectedKind = null; selectedRef = null;
  updateSidebarSelection();
  document.getElementById('confirmBtn').textContent = 'Camera position confirmed';
  document.getElementById('confirmBtn').classList.add('confirmed');
  document.getElementById('selectRig').disabled = true;
  document.querySelectorAll('#camList button').forEach(b => b.disabled = true);
  document.getElementById('placementSection').style.display = 'block';
  refreshMeshInfo();
  setStatus('camera position locked - choose meshes to place');
}

function confirmRatPlacement() {
  ratsLocked = true;
  transform.detach();
  selectedKind = null; selectedRef = null;
  updateSidebarSelection();
  updateRatButtons();
  const btn = document.getElementById('confirmRatsBtn');
  btn.textContent = 'Rat placement confirmed';
  btn.classList.add('confirmed');
  document.getElementById('fpvSection').style.display = 'block';
  updateFovAvailability();
  saveState();
  setStatus('rat placement locked - first-person view unlocked');
}

// =============================================================================
// First-person view
// =============================================================================

function currentEyeCam() {
  const cams = ratEyeCams[fpvRat];
  return cams ? cams[fpvEye] : null;
}

function updateFovAvailability() {
  const info = document.getElementById('fpvInfo');
  const hasRat = !!ratObjects[fpvRat];
  const hasEyes = !!(ratEyeCams[fpvRat]);
  document.getElementById('fpvPreview').disabled = !hasEyes;
  document.getElementById('fpvShot').disabled = !hasEyes;
  document.getElementById('pickEyes').disabled = !hasRat;
  document.querySelectorAll('#fpvSection button[data-fpvrat]').forEach(b => {
    b.classList.toggle('active', b.dataset.fpvrat === fpvRat);
  });
  document.querySelectorAll('#fpvSection button[data-fpveye]').forEach(b => {
    b.classList.toggle('active', b.dataset.fpveye === fpvEye);
  });
  if (!info) return;
  const label = fpvRat === 'extra1' ? 'rat 1' : 'rat 2';
  if (!hasRat) { info.textContent = `${label}: mesh not placed`; return; }
  if (!hasEyes) {
    info.innerHTML = `${label}: no eye positions. Either set the eye-positions JSON in setup, or click "Mark eyes on mesh" and pick the left then the right eye.`;
    return;
  }
  const src = (extraEyesCache && extraEyesCache[fpvRat] && extraEyesCache[fpvRat].source) || 'unknown';
  info.innerHTML = `${label}: eyes from ${src}.<br>The rendered cone is a pinhole view; the real monocular field is ~200&deg;, which no single perspective image can show.`;
}

function setFpvRat(id) { fpvRat = id; updateFovAvailability(); if (fpvActive) enterFpv(); }
function setFpvEye(e) { fpvEye = e; updateFovAvailability(); if (fpvActive) enterFpv(); }

// Hide everything the animal could not see: the editor's own scaffolding
// (grid, axes, table plane, camera markers and frusta, FOV cones, eye dots,
// the transform gizmo) and, optionally, the viewing animal's own body, which
// otherwise fills the frame from an eye sitting on its surface.
function setSceneForFpv(on) {
  scene.traverse(o => {
    if (o.userData && o.userData.editorOnly) o.visible = !on;
  });
  transform.visible = !on;
  transform.enabled = !on;
  const hideSelf = document.getElementById('chkHideSelf').checked;
  const self = ratObjects[fpvRat];
  if (self) {
    self.traverse(o => { if (o.isMesh && !(o.userData && o.userData.editorOnly)) o.visible = !(on && hideSelf); });
  }
}

function applyFpvFov(cam) {
  cam.fov = parseFloat(document.getElementById('fovSlider').value);
  cam.updateProjectionMatrix();
}

function enterFpv() {
  const cam = currentEyeCam();
  if (!cam) { setStatus('no eye camera for that rat'); return; }
  fpvActive = true;
  setSceneForFpv(true);
  cam.aspect = window.innerWidth / window.innerHeight;
  applyFpvFov(cam);
  document.getElementById('fpvPreview').textContent = 'Exit first-person view (Esc)';
  setStatus(`first-person view: ${fpvRat === 'extra1' ? 'rat 1' : 'rat 2'}, ${fpvEye === 'C' ? 'cyclopean' : fpvEye === 'L' ? 'left eye' : 'right eye'}`);
}

function exitFpv() {
  fpvActive = false;
  setSceneForFpv(false);
  document.getElementById('fpvPreview').textContent = 'Preview through eye';
  setStatus('back to editor view');
}

function toggleFpv() { fpvActive ? exitFpv() : enterFpv(); }

// Renders off-screen at a fixed resolution so the saved image does not depend
// on the browser window size, then reads the pixels back and flips them (GL
// origin is bottom-left, canvas origin is top-left).
function fpvScreenshot() {
  const cam = currentEyeCam();
  if (!cam) { setStatus('no eye camera for that rat'); return; }

  const W = 1600, H = 1200;
  const target = new THREE.WebGLRenderTarget(W, H, { colorSpace: THREE.SRGBColorSpace });

  const wasActive = fpvActive;
  setSceneForFpv(true);
  const prevAspect = cam.aspect;
  cam.aspect = W / H;
  applyFpvFov(cam);

  renderer.setRenderTarget(target);
  renderer.render(scene, cam);
  const pixels = new Uint8Array(W * H * 4);
  renderer.readRenderTargetPixels(target, 0, 0, W, H, pixels);
  renderer.setRenderTarget(null);

  cam.aspect = prevAspect;
  cam.updateProjectionMatrix();
  if (!wasActive) setSceneForFpv(false);

  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    const src = (H - 1 - y) * W * 4;
    img.data.set(pixels.subarray(src, src + W * 4), y * W * 4);
  }
  ctx.putImageData(img, 0, 0);

  const eyeName = fpvEye === 'C' ? 'cyclopean' : fpvEye === 'L' ? 'eyeL' : 'eyeR';
  const name = `${fpvRat}_${eyeName}`;
  const meta = {
    rat: fpvRat,
    eye: eyeName,
    fov_deg: cam.fov,
    resolution: [W, H],
    mesh_path: fpvRat === 'extra1' ? sceneConfig.rat1_mesh_path : sceneConfig.rat2_mesh_path,
    eye_source: (extraEyesCache && extraEyesCache[fpvRat] && extraEyesCache[fpvRat].source) || null,
    placement_mode: placementMode(),
    note: 'pinhole render; true rodent monocular field is ~200 deg and cannot be shown in one perspective image',
  };

  target.dispose();
  setStatus('saving screenshot...');
  fetch(API + '/api/screenshot', {
    method: 'POST',
    body: JSON.stringify({ name, png: canvas.toDataURL('image/png'), meta }),
  }).then(r => r.json())
    .then(res => setStatus(res.ok ? `saved ${res.path}` : 'screenshot failed'))
    .catch(e => { console.error(e); setStatus('screenshot failed (see console)'); });
}

// --- marking eyes directly on the mesh, for when no eye-positions JSON
// exists yet. Click the left eye, then the right; the head basis is seeded
// the same way the eye-marking tool seeds it (longest horizontal bounding-box
// axis as a nose-to-tail proxy, re-orthogonalised against the inter-eye axis).
function startEyePick() {
  const id = fpvRat;
  if (!ratObjects[id]) { setStatus('place the mesh first'); return; }
  eyePickState = { id, stage: 'L', picked: {} };
  setStatus('click the LEFT eye on the mesh (Esc to cancel)');
}

function handleEyePick(raycaster) {
  const obj = ratObjects[eyePickState.id];
  if (!obj) { eyePickState = null; return; }
  const hits = raycaster.intersectObject(obj, true);
  if (!hits.length) { setStatus('click on the mesh surface'); return; }

  const local = obj.worldToLocal(hits[0].point.clone());
  if (eyePickState.stage === 'L') {
    eyePickState.picked.eyeL = local.toArray();
    eyePickState.stage = 'R';
    setStatus('now click the RIGHT eye');
    return;
  }
  eyePickState.picked.eyeR = local.toArray();

  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  // mesh local frame is z-up, so the longer of x/y is the nose-to-tail axis
  const forward = size.x >= size.y ? [1, 0, 0] : [0, 1, 0];
  const entry = { ...eyePickState.picked, headForward: forward, headUp: [0, 0, 1] };
  const id = eyePickState.id;
  eyePickState = null;

  fetch(API + '/api/manual_eyes', { method: 'POST', body: JSON.stringify({ id, ...entry }) })
    .then(r => r.json())
    .then(() => fetchExtraEyes(true))
    .then(() => {
      const bbox = new THREE.Box3().setFromObject(ratObjects[id]);
      const fovRadius = bbox.getSize(new THREE.Vector3()).length() * 0.35;
      for (const m of ratFovMeshes[id]) ratObjects[id].remove(m);
      ratFovMeshes[id] = [];
      addFovForRat(id, extraEyesCache[id], fovRadius);
      updateFovAvailability();
      setStatus(`${id}: eyes marked and saved`);
    })
    .catch(e => { console.error(e); setStatus('saving eyes failed'); });
}

function saveState() {
  const perCamera = {};
  for (const name in camMarkers) {
    const h = camMarkers[name].holder;
    perCamera[name] = { position: h.position.toArray(), quaternion: h.quaternion.toArray() };
  }
  const rats = {};
  for (const id of ['extra1', 'extra2']) {
    const o = ratObjects[id];
    if (!o) continue;
    rats[id] = { position: o.position.toArray(), quaternion: o.quaternion.toArray(), scale: o.scale.toArray() };
  }
  const payload = {
    cameras: {
      group: {
        position: rigGroup.position.toArray(),
        quaternion: rigGroup.quaternion.toArray(),
        scale: rigGroup.scale.toArray(),
      },
      perCamera,
    },
    rats,
    placement_mode: placementMode(),
  };
  fetch(API + '/api/state', { method: 'POST', body: JSON.stringify(payload) })
    .then(() => setStatus('saved to scene_placements.json'))
    .catch(() => setStatus('save failed'));
}

function exportSceneObj() {
  if (!environmentObject || !rigGroup) { setStatus('scene not fully loaded yet'); return; }
  setStatus('exporting OBJ...');

  // Custom exporter (not THREE.OBJExporter, which only writes bare geometry
  // under one generic "usemtl Material" with no real material/texture link):
  // walk every Mesh, bake its world transform into the vertices, and give
  // each one its own named material block - textured (map_Kd) if it has an
  // image texture, flat Kd color + opacity otherwise (camera markers, table
  // plane, FOV cones).
  let vOff = 0, vtOff = 0, vnOff = 0;
  let objLines = ['mtllib scene.mtl'];
  let mtlLines = [];
  const usedTexUrls = new Map();
  let matCounter = 0, partCounter = 0;

  function materialFor(mat) {
    if (mat && mat.map && mat.map.image && mat.map.image.src) {
      const url = mat.map.image.src;
      if (usedTexUrls.has(url)) return usedTexUrls.get(url).matName;
      const ext = (url.split('.').pop().split(/[?#]/)[0] || 'png').toLowerCase();
      const matName = `tex_${matCounter}`;
      const fileName = `tex_${matCounter}.${ext}`;
      matCounter++;
      usedTexUrls.set(url, { matName, fileName });
      mtlLines.push(`newmtl ${matName}`, 'Ka 1 1 1', 'Kd 1 1 1', 'Ks 0 0 0', 'd 1', `map_Kd ${fileName}`, '');
      return matName;
    }
    const color = (mat && mat.color) ? mat.color : new THREE.Color(0xcccccc);
    const opacity = (mat && mat.opacity !== undefined) ? mat.opacity : 1;
    const name = `flat_${matCounter++}`;
    mtlLines.push(`newmtl ${name}`, 'Ka 1 1 1', `Kd ${color.r.toFixed(4)} ${color.g.toFixed(4)} ${color.b.toFixed(4)}`,
      'Ks 0 0 0', `d ${opacity.toFixed(3)}`, '');
    return name;
  }

  function addMesh(mesh) {
    if (!mesh.geometry || !mesh.geometry.attributes || !mesh.geometry.attributes.position) return;
    mesh.updateMatrixWorld(true);
    const geo = mesh.geometry.clone();
    geo.applyMatrix4(mesh.matrixWorld);
    const pos = geo.attributes.position;
    const uv = geo.attributes.uv;
    const norm = geo.attributes.normal;
    const idx = geo.index;
    const n = pos.count;

    const matName = materialFor(Array.isArray(mesh.material) ? mesh.material[0] : mesh.material);
    const partName = (mesh.name || `part_${partCounter}`).replace(/\s+/g, '_') + `_${partCounter++}`;

    objLines.push(`o ${partName}`, `usemtl ${matName}`);
    for (let i = 0; i < n; i++) objLines.push(`v ${pos.getX(i)} ${pos.getY(i)} ${pos.getZ(i)}`);
    if (uv) for (let i = 0; i < n; i++) objLines.push(`vt ${uv.getX(i)} ${uv.getY(i)}`);
    if (norm) for (let i = 0; i < n; i++) objLines.push(`vn ${norm.getX(i)} ${norm.getY(i)} ${norm.getZ(i)}`);

    const nTris = idx ? idx.count / 3 : n / 3;
    for (let t = 0; t < nTris; t++) {
      const ia = idx ? idx.getX(3 * t) : 3 * t;
      const ib = idx ? idx.getX(3 * t + 1) : 3 * t + 1;
      const ic = idx ? idx.getX(3 * t + 2) : 3 * t + 2;
      const f = (i) => {
        const v = i + 1 + vOff;
        const vt = uv ? (i + 1 + vtOff) : '';
        const vn = norm ? (i + 1 + vnOff) : '';
        if (uv && norm) return `${v}/${vt}/${vn}`;
        if (uv) return `${v}/${vt}`;
        if (norm) return `${v}//${vn}`;
        return `${v}`;
      };
      objLines.push(`f ${f(ia)} ${f(ib)} ${f(ic)}`);
    }
    vOff += n; if (uv) vtOff += n; if (norm) vnOff += n;
  }

  const wasSelected = selectedKind, wasRef = selectedRef;
  transform.detach();

  environmentObject.traverse(o => { if (o.isMesh) addMesh(o); });
  rigGroup.traverse(o => { if (o.isMesh) addMesh(o); });

  if (wasSelected) select(wasSelected, wasRef);

  const payload = {
    obj: objLines.join('\n'),
    mtl: mtlLines.join('\n'),
    textures: Array.from(usedTexUrls, ([url, { fileName }]) => ({ url, fileName })),
  };

  setStatus('exporting textured OBJ...');
  fetch(API + '/api/export_obj_textured', { method: 'POST', body: JSON.stringify(payload) })
    .then(r => r.json())
    .then(res => setStatus(`exported: ${res.dir} (obj ${(res.obj_size / 1e6).toFixed(1)} MB, ${res.textures_copied} textures)`))
    .catch((e) => { console.error(e); setStatus('export failed'); });
}

function animate() {
  requestAnimationFrame(animate);
  if (fpvActive) {
    const cam = currentEyeCam();
    if (cam) {
      cam.aspect = window.innerWidth / window.innerHeight;
      applyFpvFov(cam);
      renderer.render(scene, cam);
      return;
    }
    exitFpv();
  }
  orbit.update();
  renderer.render(scene, camera);
}

document.getElementById('modeTranslate').addEventListener('click', () => transform.setMode('translate'));
document.getElementById('modeRotate').addEventListener('click', () => transform.setMode('rotate'));
document.getElementById('modeScale').addEventListener('click', () => transform.setMode('scale'));
document.getElementById('selectRig').addEventListener('click', () => select('rig', null));
document.getElementById('saveBtn').addEventListener('click', saveState);
document.getElementById('exportBtn').addEventListener('click', exportSceneObj);
document.getElementById('confirmBtn').addEventListener('click', confirmCameraPosition);
document.getElementById('chkObj1').addEventListener('change', rebuildRatPlacement);
document.getElementById('chkObj2').addEventListener('change', rebuildRatPlacement);
document.getElementById('chkFov').addEventListener('change', rebuildRatPlacement);
document.querySelectorAll('input[name="placeMode"]').forEach(r => r.addEventListener('change', rebuildRatPlacement));
document.getElementById('selRat1').addEventListener('click', () => select('extraRat', 'extra1'));
document.getElementById('selRat2').addEventListener('click', () => select('extraRat', 'extra2'));
document.getElementById('confirmRatsBtn').addEventListener('click', confirmRatPlacement);
document.getElementById('fpvRat1').addEventListener('click', () => setFpvRat('extra1'));
document.getElementById('fpvRat2').addEventListener('click', () => setFpvRat('extra2'));
document.getElementById('fpvEyeL').addEventListener('click', () => setFpvEye('L'));
document.getElementById('fpvEyeR').addEventListener('click', () => setFpvEye('R'));
document.getElementById('fpvEyeC').addEventListener('click', () => setFpvEye('C'));
document.getElementById('fpvPreview').addEventListener('click', toggleFpv);
document.getElementById('fpvShot').addEventListener('click', fpvScreenshot);
document.getElementById('pickEyes').addEventListener('click', startEyePick);
document.getElementById('fovSlider').addEventListener('input', (e) => {
  document.getElementById('fovVal').textContent = e.target.value;
});

// entry point called from index.html once the setup form is submitted
window.__enterScene = init;
