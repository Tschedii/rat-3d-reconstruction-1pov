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
const camMarkers = {}; // name -> { holder, cam, pickMesh, sprite }
let pickables = []; // {mesh, kind, ref}
let selectedKind = null, selectedRef = null;
let initialRig = null;
let sceneConfig = null;
let rigLocked = false;

// currently-placed rat objects/FOV meshes, keyed by id ('extra1'/'extra2')
const ratObjects = { extra1: null, extra2: null };
const ratFovMeshes = { extra1: [], extra2: [] };
let extraEyesCache = null;

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

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.getElementById('stage').appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(3, 6, 4);
  scene.add(sun);
  scene.add(new THREE.AxesHelper(0.5));
  scene.add(new THREE.GridHelper(6, 24, 0x444444, 0x2a2a22));

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

function loadEnvironment() {
  const objName = basename(sceneConfig.env_mesh_path);
  const mtlName = withExt(objName, '.mtl');
  const mtlLoader = new MTLLoader();
  mtlLoader.setPath(API + '/env/');
  mtlLoader.load(mtlName, (materials) => {
    materials.preload();
    const objLoader = new OBJLoader();
    objLoader.setMaterials(materials);
    objLoader.setPath(API + '/env/');
    objLoader.load(objName, (object) => {
      object.name = 'environment';
      environmentObject = object;
      scene.add(object);
      setStatus('environment loaded');
    }, undefined, (err) => setStatus('environment load error (see console)') || console.error(err));
  }, undefined, (err) => setStatus('environment .mtl load error (see console)') || console.error(err));
}

function buildRigGroup(rig, savedCams) {
  const tableGeo = new THREE.PlaneGeometry(1.6, 1.6);
  const tableMat = new THREE.MeshBasicMaterial({ color: 0xc9a97e, transparent: true, opacity: 0.28, side: THREE.DoubleSide });
  tablePlane = new THREE.Mesh(tableGeo, tableMat);
  tablePlane.rotation.x = -Math.PI / 2;
  rigGroup.add(tablePlane);
  pickables.push({ mesh: tablePlane, kind: 'rig', ref: null });

  const tableCenter = remap(rig.table_center);

  for (const c of rig.cameras) {
    const pos = remap(c.position).sub(tableCenter);
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
    helper.setColors ? helper.setColors(new THREE.Color(color), new THREE.Color(color), new THREE.Color(color), new THREE.Color(color), new THREE.Color(color)) : null;
    scene.add(helper);

    const pickGeo = new THREE.SphereGeometry(0.035, 12, 12);
    const pickMat = new THREE.MeshBasicMaterial({ color });
    const pickMesh = new THREE.Mesh(pickGeo, pickMat);
    holder.add(pickMesh);

    const sprite = makeLabelSprite(c.name.replace('camera_', 'cam '), color);
    sprite.position.set(0, 0.09, 0);
    holder.add(sprite);

    camMarkers[c.name] = { holder, persp, helper, pickMesh };
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

function loadRig() {
  fetch(API + '/api/state').then(r => r.json()).then(state => {
    fetch(API + '/api/initial_rig').then(r => r.json()).then(rig => {
      initialRig = rig;
      buildRigGroup(rig, state.cameras);
    });
  }).catch(() => {
    fetch(API + '/api/initial_rig').then(r => r.json()).then(rig => {
      initialRig = rig;
      buildRigGroup(rig, null);
    });
  });
}

// A rat mesh.obj is Z-up (table at local Z=0) and already in real-world
// metres, same as the rig's calibration data - so the rat-to-rig transform
// is fully determined: same axis convention as the camera remap (a proper
// Z-up -> Y-up rotation), identity scale (both already metres). Parenting
// the rat under rigGroup means it automatically inherits whatever coarse-fit
// transform the rig was placed with.
const RAT_TO_RIG_QUAT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

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

function loadExtraRat(id, meshPath, tableOffsetXZ, fovEnabled) {
  const objName = basename(meshPath);
  const mtlName = withExt(objName, '.mtl');
  const mtlLoader = new MTLLoader();
  mtlLoader.setPath(`${API}/${id}/`);
  mtlLoader.load(mtlName, (materials) => {
    materials.preload();
    const objLoader = new OBJLoader();
    objLoader.setMaterials(materials);
    objLoader.setPath(`${API}/${id}/`);
    objLoader.load(objName, (object) => {
      object.name = id;
      // Each mesh's own local origin is just wherever the animal happened to
      // sit within its own per-frame reconstruction crop - not a shared
      // absolute arena position - so this offset is purely a display
      // separation so two rats don't render on top of one another; it is
      // NOT derived from any calibration. Their SIZE and their distance FROM
      // THE CAMERA RIG are real (from the shared rig calibration).
      object.position.set(tableOffsetXZ[0], 0, tableOffsetXZ[1]);
      object.quaternion.copy(RAT_TO_RIG_QUAT);
      object.scale.set(1, 1, 1);
      rigGroup.add(object);
      pickables.push({ mesh: object, kind: 'extraRat', ref: id });
      ratObjects[id] = object;

      if (!fovEnabled) { setStatus(`${id}: mesh loaded`); return; }

      const box = new THREE.Box3().setFromObject(object);
      const fovRadius = box.getSize(new THREE.Vector3()).length() * 0.35;

      fetchExtraEyes().then(eyes => {
        const e = eyes[id];
        if (!e) { setStatus(`${id}: mesh loaded, no eye labels found`); return; }
        addFovForRat(id, e, fovRadius, new THREE.Vector3(tableOffsetXZ[0], 0, tableOffsetXZ[1]));
        setStatus(`${id}: mesh + field-of-view loaded`);
      });
    }, undefined, (err) => setStatus(`${id} load error (see console)`) || console.error(err));
  }, undefined, (err) => setStatus(`${id} .mtl load error (see console)`) || console.error(err));
}

function fetchExtraEyes() {
  if (extraEyesCache) return Promise.resolve(extraEyesCache);
  return fetch(`${API}/api/extra_eyes`).then(r => r.json()).then(eyes => { extraEyesCache = eyes; return eyes; });
}

function addFovForRat(id, eyeData, fovRadius, offset) {
  const q = RAT_TO_RIG_QUAT;
  const eyeL = new THREE.Vector3(...eyeData.eyeL).applyQuaternion(q).add(offset);
  const eyeR = new THREE.Vector3(...eyeData.eyeR).applyQuaternion(q).add(offset);
  let headForward = new THREE.Vector3(...eyeData.headForward).applyQuaternion(q).normalize();
  const headUp = new THREE.Vector3(...eyeData.headUp).applyQuaternion(q).normalize();

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

  rigGroup.add(capL, capR, dotL, dotR);
  ratFovMeshes[id] = [capL, capR, dotL, dotR];
}

function clearRat(id) {
  if (ratObjects[id]) {
    rigGroup.remove(ratObjects[id]);
    pickables = pickables.filter(p => p.ref !== id);
    ratObjects[id] = null;
  }
  clearFov(id);
}

function clearFov(id) {
  for (const m of ratFovMeshes[id]) rigGroup.remove(m);
  ratFovMeshes[id] = [];
}

// Rebuilds rat placement from scratch whenever the checkboxes change, rather
// than trying to incrementally reposition already-placed meshes - simplest
// correct way to keep the +-0.22m two-rat display offset (or 0 for a single
// rat) consistent, and cheap for two small meshes.
function rebuildRatPlacement() {
  const wantObj1 = document.getElementById('chkObj1').checked;
  const wantObj2 = document.getElementById('chkObj2').checked;
  const wantFov = document.getElementById('chkFov').checked;

  clearRat('extra1');
  clearRat('extra2');

  const active = [];
  if (wantObj1 && sceneConfig.rat1_mesh_path) active.push('extra1');
  if (wantObj2 && sceneConfig.rat2_mesh_path) active.push('extra2');

  active.forEach((id, i) => {
    let offset;
    if (active.length === 2) offset = i === 0 ? [-0.22, 0] : [0.22, 0];
    else offset = [0, 0];
    const meshPath = id === 'extra1' ? sceneConfig.rat1_mesh_path : sceneConfig.rat2_mesh_path;
    loadExtraRat(id, meshPath, offset, wantFov);
  });

  if (!active.length) setStatus('no meshes placed');
}

function select(kind, ref) {
  if (rigLocked && (kind === 'rig' || kind === 'camera')) return;
  selectedKind = kind; selectedRef = ref;
  let obj = null;
  if (kind === 'rig') obj = rigGroup;
  else if (kind === 'camera') obj = camMarkers[ref].holder;
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
    const match = b.dataset.kind === selectedKind && (b.dataset.kind !== 'camera' || b.dataset.ref === selectedRef);
    b.classList.toggle('active', match);
  });
  document.getElementById('selLabel').textContent =
    selectedKind === 'rig' ? 'Rig (all cameras)' :
    selectedKind === 'camera' ? selectedRef.replace('camera_', 'cam ') : 'none';
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
  setStatus('camera position locked - choose meshes to place');
}

function saveState() {
  const perCamera = {};
  for (const name in camMarkers) {
    const h = camMarkers[name].holder;
    perCamera[name] = { position: h.position.toArray(), quaternion: h.quaternion.toArray() };
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

// entry point called from index.html once the setup form is submitted
window.__enterScene = init;
