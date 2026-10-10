import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';

// same origin as the page (server.py serves both), so a forwarded or changed port still works
const API = location.protocol.startsWith('http') ? location.origin : 'http://127.0.0.1:8976';

// =============================================================================
// Frames. Same conventions as GUI 4/5.1's app.js so a position written here
// means the same thing as a placement there:
//   calibration frame (mesh.obj, camera poses)  --display_basis-->  table frame
//   (z up, origin at table centre)              --remap-->            three.js
//   (y up).  remap is a proper rotation (det +1), not an axis swap.
// The scene root "rigGroup" is the table frame remapped to three.js and sits
// at the origin, so rigGroup-local coordinates are table-frame coordinates
// with y and z exchanged: local = (x, z, -y), table = (x, -z, y).
// =============================================================================
function remap(v) { return new THREE.Vector3(v[0], v[2], -v[1]); }
function unremap(v) { return [v.x, -v.z, v.y]; }

const REMAP_MATRIX = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, 0, 1, 0,
  0, -1, 0, 0,
  0, 0, 0, 1,
);

const CAM_COLORS = {
  camera_001: 0x5b8def, camera_002: 0x5bcf9e, camera_003: 0xd6c15b,
  camera_004: 0xf96167, camera_005: 0xb87bd9, camera_006: 0xe0824a,
};
const STATE_COLORS = { rest: 0x2a78d6, head_movement: 0xeb6834, locomotion: 0x1baf7a };
// Rat identity colours, the same as GUI 5.1's mask tint and this tool's plots
// (analysis_lib.RAT_COLOR): rat 1 = mask label 1 (object 1) red, rat 2 =
// mask label 2 (object 2) blue. Paths, meshes, markers and results all use them.
const RAT_COLORS = { rat1: 0xff2828, rat2: 0x286eff };
// lighter, desaturated version of the same hue (body-centroid paths)
function ratLight(id) { return new THREE.Color(RAT_COLORS[id]).lerp(new THREE.Color(0xd8d4c8), 0.55); }

let scene, camera, renderer, orbit;
let rigGroup, rigVisuals, pathGroup, markerGroup;
let sceneConfig = null;
let initialRig = null;
let tableCenterLocal = new THREE.Vector3();
let calibToRigQuat = new THREE.Quaternion();

// No head-mounted IMU, no gizmo, no per-frame placement: rat1/rat2 meshes are
// shown for visual context (the server also uses each tracked rat's own one as
// its head fallback). Which rats are tracked (rat 1 = mask label 1, rat 2 =
// mask label 2) is chosen per run with the analysis section's checkboxes, not
// per frame, since tracking from the masks has no anchor frame to attach to.
const rats = {
  rat1: { object: null, frames: [], frame: null, info: null, generation: 0 },
  rat2: { object: null, frames: [], frame: null, info: null, generation: 0 },
};
const seqInfo = { rat1: null, rat2: null }; // provenance of each mesh sequence (world coords vs re-centred)
let placeModeChosen = false; // did the user (or a saved state) pick the placement frame explicitly?

let fps = 30;
let currentFrame = 0;
let playing = false, playSpeed = 1, playAccum = 0, lastTick = 0;
let lastResult = null;

function meshFrame() { return (sceneConfig && sceneConfig.mesh_frame) || 'calib'; }
function setStatus(msg) { document.getElementById('status').textContent = msg; }
function fmtVec(v, d = 3) { return `(${Array.from(v).map(x => x.toFixed(d)).join(', ')})`; }
function frameId(n) { return `frame_${String(n).padStart(6, '0')}`; }

// =============================================================================
// Scene
// =============================================================================

function init(config) {
  sceneConfig = config;
  fps = config.fps || 30;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x15140f);

  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.005, 100);
  camera.position.set(1.5, 1.2, 1.5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  document.getElementById('stage').appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(3, 6, 4);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xffffff, 0.4);
  fill.position.set(-4, 2, -3);
  scene.add(fill);

  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 0, 0);
  orbit.update();

  rigGroup = new THREE.Group();
  rigGroup.name = 'rig';
  scene.add(rigGroup);
  rigVisuals = new THREE.Group();
  rigGroup.add(rigVisuals);
  pathGroup = new THREE.Group();
  rigGroup.add(pathGroup);
  markerGroup = new THREE.Group();
  rigGroup.add(markerGroup);

  const grid = new THREE.GridHelper(4, 40, 0x3a3826, 0x26241a);
  rigGroup.add(grid);

  window.addEventListener('resize', onResize);
  window.addEventListener('keydown', onKeyDown);

  Promise.all([loadRig(), loadFrameList('rat1'), loadFrameList('rat2'), fetchState()])
    .then(([, , , state]) => {
      applyDefaults(state);
      loadRatMesh('rat1');
      if (rats.rat2.frames.length && document.getElementById('chkRat2').checked) loadRatMesh('rat2');
      updateFrameUi();
      loadSequenceInfo('rat1'); loadSequenceInfo('rat2');
    })
    .catch(e => { console.error(e); setStatus('startup failed: ' + e.message); });

  lastTick = performance.now();
  animate();
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function makeLabelSprite(text, color, scale = 0.12) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(20,20,28,0.7)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = 'bold 30px system-ui, sans-serif';
  ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, sizeAttenuation: true }));
  sprite.scale.set(scale, scale / 4, 1);
  return sprite;
}

// --- camera rig (context only, not editable) --------------------------------
function loadRig() {
  return fetch(API + '/api/initial_rig').then(r => r.json()).then(rig => {
    if (rig.error) throw new Error('rig: ' + rig.error);
    initialRig = rig;
    tableCenterLocal = remap(rig.table_center);
    const [bx, by, bz] = rig.display_basis;
    const toDisplay = new THREE.Matrix4().set(
      bx[0], bx[1], bx[2], 0,
      by[0], by[1], by[2], 0,
      bz[0], bz[1], bz[2], 0,
      0, 0, 0, 1,
    );
    calibToRigQuat = new THREE.Quaternion().setFromRotationMatrix(REMAP_MATRIX.clone().multiply(toDisplay));
    // A re-oriented export is already y-up in the mesh's own frame: show it as
    // is (identity), measured from the origin rather than from the table centre.
    if (meshFrame() === 'yup') { calibToRigQuat.identity(); tableCenterLocal.set(0, 0, 0); }

    const tableGeo = new THREE.PlaneGeometry(1.6, 1.6);
    const tableMat = new THREE.MeshBasicMaterial({ color: 0xc9a97e, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false });
    const tablePlane = new THREE.Mesh(tableGeo, tableMat);
    tablePlane.rotation.x = -Math.PI / 2;
    rigVisuals.add(tablePlane);

    for (const c of rig.cameras) {
      const pos = remap(c.position).sub(tableCenterLocal);
      const fwd = remap(c.forward).normalize();
      const up = remap(c.up).normalize();
      const right = remap(c.right).normalize();
      const h = new THREE.Object3D();
      h.position.copy(pos);
      h.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, fwd.clone().negate()));
      const color = CAM_COLORS[c.name] || 0x8888ff;
      const persp = new THREE.PerspectiveCamera(44, 1920 / 1200, 0.03, 0.18);
      h.add(persp);
      const helper = new THREE.CameraHelper(persp);
      if (helper.setColors) helper.setColors(new THREE.Color(color), new THREE.Color(color), new THREE.Color(color), new THREE.Color(color), new THREE.Color(color));
      h.add(new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 12), new THREE.MeshBasicMaterial({ color })));
      const sprite = makeLabelSprite(c.name.replace('camera_', 'cam '), color, 0.2);
      sprite.position.set(0, 0.07, 0);
      h.add(sprite);
      rigVisuals.add(h);
      // CameraHelper works in world space and must be a scene child
      helper.userData.rigHelper = true;
      rigVisuals.add(helper);
    }
    rigVisuals.visible = document.getElementById('chkRig').checked;
  });
}

// --- mesh frames --------------------------------------------------------------
function loadFrameList(id) {
  return fetch(`${API}/api/frames?which=${id}`).then(r => r.json()).then(d => {
    rats[id].frames = (d.frames || []).map(f => f.frame);
    const sel = document.getElementById(`${id}Frame`);
    sel.innerHTML = '';
    for (const f of rats[id].frames) {
      const opt = document.createElement('option');
      opt.value = String(f);
      opt.textContent = `${frameId(f)}  (${(f / fps).toFixed(2)} s)`;
      sel.appendChild(opt);
    }
    if (!rats[id].frames.length) {
      const opt = document.createElement('option');
      opt.textContent = d.configured ? (d.error || 'no frame_XXXXXX/ meshes found') : 'not configured';
      sel.appendChild(opt);
      sel.disabled = true;
      document.getElementById(`${id}Prev`).disabled = true;
      document.getElementById(`${id}Next`).disabled = true;
    }
  });
}

// Measures, from the actual mesh files, whether the sequence still carries
// world coordinates. A re-centred sequence (every mesh at one fixed spot)
// cannot show the rats' true relative position, and silently pretending
// otherwise is exactly the failure this readout prevents. (Trajectory
// computation itself does not depend on these meshes at all here -- this is
// display-only context.)
function loadSequenceInfo(id) {
  if (!rats[id].frames.length) return;
  fetch(`${API}/api/sequence_info?which=${id}`).then(r => r.json()).then(info => {
    seqInfo[id] = info;
    if (info.recentred && !placeModeChosen && placementMode() === 'world') {
      document.getElementById('modeSplit').checked = true;
      applyPlacementMode();
      setStatus(`${id}: meshes were re-centred, switched to side-by-side display`);
    }
    updateMeshInfo();
  }).catch(() => {});
}

function placementMode() {
  const el = document.querySelector('input[name="placeMode"]:checked');
  return el ? el.value : 'world';
}

// display-only lateral offset per rat in side-by-side mode (rigGroup-local)
function ratDisplayOffset(id) {
  const both = rats.rat1.frames.length && rats.rat2.frames.length && document.getElementById('chkRat2').checked;
  if (placementMode() !== 'split' || !both) return new THREE.Vector3();
  return new THREE.Vector3(id === 'rat1' ? -0.22 : 0.22, 0, 0);
}

function applyPlacementMode() {
  for (const id of ['rat1', 'rat2']) {
    const o = rats[id].object;
    if (o) o.position.copy(tableCenterLocal).negate().add(ratDisplayOffset(id));
  }
  updateMeshInfo();
  focusOnRats();
}

function fetchState() {
  return fetch(API + '/api/state').then(r => r.json()).catch(() => ({}));
}

function applyDefaults(state) {
  // default: both rats at the first frame they share, so the opening view is a
  // real simultaneous configuration; the user can then pick any frame per rat
  const f1 = rats.rat1.frames, f2 = rats.rat2.frames;
  const common = f1.filter(f => f2.includes(f));
  rats.rat1.frame = f1.length ? (common[0] ?? f1[0]) : null;
  rats.rat2.frame = f2.length ? (common[0] ?? f2[0]) : null;

  if (state && state.rat1_frame != null && f1.includes(state.rat1_frame)) rats.rat1.frame = state.rat1_frame;
  if (state && state.rat2_frame != null && f2.includes(state.rat2_frame)) rats.rat2.frame = state.rat2_frame;
  if (state && (state.place_mode === 'world' || state.place_mode === 'split')) {
    document.getElementById(state.place_mode === 'split' ? 'modeSplit' : 'modeWorld').checked = true;
    placeModeChosen = true;
  }
  if (state && state.params) {
    const p = state.params;
    const set = (id, v) => { if (v != null) document.getElementById(id).value = v; };
    set('pAng', p.rest_ang_speed_dps); set('pLin', p.rest_lin_speed_mps); set('pMinDur', p.min_state_duration_s);
    set('pSmooth', p.smoothing_s); set('pPosSmooth', p.position_smoothing_s); set('pRefFrame', p.reference_frame);
    set('pGap', p.max_interp_gap_frames); set('pMinConf', p.min_confidence);
    if (p.frame_start != null) document.getElementById('pFrameStart').value = p.frame_start;
    if (p.frame_end != null) document.getElementById('pFrameEnd').value = p.frame_end;
  }
  for (const id of ['rat1', 'rat2']) if (rats[id].frame != null) document.getElementById(`${id}Frame`).value = String(rats[id].frame);
  currentFrame = rats.rat1.frame ?? 0;
  document.getElementById('frameSlider').value = String(currentFrame);
}

// A missing or unparseable .mtl must not prevent the .obj from loading (same
// fix as GUI 4): geometry-only exports just render untextured.
function loadObjMaybeMtl(urlPrefix, objName, onLoad, onFail) {
  const mtlName = objName.replace(/\.[^./]+$/, '.mtl');
  const loadObj = (materials) => {
    const objLoader = new OBJLoader();
    if (materials) objLoader.setMaterials(materials);
    objLoader.setPath(urlPrefix);
    objLoader.load(objName, onLoad, undefined, (err) => { console.error(err); onFail(`${objName} not found or unreadable`); });
  };
  const mtlLoader = new MTLLoader();
  mtlLoader.setPath(urlPrefix);
  mtlLoader.load(mtlName, (materials) => {
    try { materials.preload(); loadObj(materials); } catch (e) { console.warn(e); loadObj(null); }
  }, undefined, () => loadObj(null));
}

function clearRat(id) {
  const r = rats[id];
  if (r.object) { rigGroup.remove(r.object); r.object = null; }
}

function loadRatMesh(id) {
  const r = rats[id];
  clearRat(id);
  if (r.frame == null) { updateMeshInfo(); return; }
  const gen = ++r.generation;
  const fid = frameId(r.frame);
  setStatus(`${id}: loading ${fid}/mesh.obj …`);
  fetch(`${API}/api/mesh_info?which=${id}&frame=${r.frame}`).then(x => x.json()).then(info => {
    if (gen !== r.generation) return;
    r.info = info;
    updateMeshInfo();
  }).catch(() => {});

  loadObjMaybeMtl(`${API}/${id}/${fid}/`, 'mesh.obj', (object) => {
    if (gen !== r.generation) return;
    object.name = id;
    // the mesh is in calibration-frame metres: calib -> table -> three.js,
    // then shift by -tableCenter because rigGroup-local is measured from the
    // table centre. Fully determined, so the rats keep their true separation.
    object.position.copy(tableCenterLocal).negate().add(ratDisplayOffset(id));
    object.quaternion.copy(calibToRigQuat);
    // tinted red / blue through the material colour. A texture keeps only its
    // brightness (times that colour), so its surface detail still reads: the
    // plain colour * texture product would turn blue x yellow-green fur into
    // dark green rather than blue.
    const ratColor = new THREE.Color(RAT_COLORS[id]);
    object.traverse(o => {
      if (o.isMesh) {
        for (const m of (Array.isArray(o.material) ? o.material : [o.material])) {
          m.side = THREE.DoubleSide;
          if (m.color) m.color.copy(ratColor);
          if (m.map) {
            m.onBeforeCompile = (shader) => {
              shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>',
                '#include <map_fragment>\n#ifdef USE_MAP\n' +
                '  diffuseColor.rgb = diffuse * mix(0.35, 1.0, dot(sampledDiffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722)));\n#endif');
            };
            m.needsUpdate = true;
          }
        }
      }
    });
    rigGroup.add(object);
    r.object = object;
    setStatus(`${id}: ${fid} loaded`);
    if (!scene.userData.focused) { focusOnRats(); scene.userData.focused = true; }
  }, (msg) => setStatus(`${id}: ${msg}`));
}

function ratBboxLocal(id) {
  const o = rats[id].object;
  if (!o) return null;
  return new THREE.Box3().setFromObject(o);
}

// mesh centroid (calibration frame, from the server) -> rigGroup-local,
// including the display offset of that rat when one is given
function centroidLocal(info, id) {
  if (!info || !info.centroid) return null;
  const c = new THREE.Vector3(...info.centroid).applyQuaternion(calibToRigQuat).sub(tableCenterLocal);
  return id ? c.add(ratDisplayOffset(id)) : c;
}

function updateMeshInfo() {
  const lines = [];
  for (const id of ['rat1', 'rat2']) {
    const r = rats[id];
    const label = id === 'rat1' ? 'rat 1' : 'rat 2';
    if (r.frame == null) { lines.push(`${label}: not configured`); continue; }
    if (id === 'rat2' && !document.getElementById('chkRat2').checked) { lines.push(`${label}: hidden`); continue; }
    if (!r.info) { lines.push(`${label}: ${frameId(r.frame)} …`); continue; }
    if (!r.info.exists || r.info.error) { lines.push(`${label}: ${frameId(r.frame)} ${r.info.error || 'not found'}`); continue; }
    const size = [0, 1, 2].map(i => ((r.info.bbox_max[i] - r.info.bbox_min[i]) * 100).toFixed(1)).join(' x ');
    const cT = unremap(centroidLocal(r.info));
    lines.push(`${label}: ${frameId(r.frame)} (${(r.frame / fps).toFixed(2)} s), ${size} cm`);
    lines.push(`&nbsp;&nbsp;centroid, table frame: ${fmtVec(cT)} m`);
  }
  const a = rats.rat1.frame, b = rats.rat2.frame;
  if (a != null && b != null && rats.rat2.object && a !== b) {
    lines.push(`<span class="warn">rat 1 is frame ${a}, rat 2 is frame ${b}: different instants (${((b - a) / fps).toFixed(2)} s apart), each in its true position at its own time.</span>`);
  }
  for (const id of ['rat1', 'rat2']) {
    const si = seqInfo[id];
    if (!si || !si.n_frames) continue;
    const label = id === 'rat1' ? 'rat 1' : 'rat 2';
    lines.push(si.recentred
      ? `<span class="warn">${label} sequence (${si.n_frames} meshes): ${si.reason}. Relative position between the rats is NOT recoverable from these meshes -- irrelevant here, since the trajectory below never reads them.</span>`
      : `${label} sequence (${si.n_frames} meshes): ${si.reason}.`);
  }
  if (placementMode() === 'split') lines.push('<span class="warn">side-by-side display: the lateral offset is cosmetic, display only.</span>');
  if (meshFrame() === 'yup') lines.push('mesh frame: y-up export shown as is; "table frame" below means the mesh\'s own frame (z up after remap), not the calibrated rig.');
  document.getElementById('meshInfo').innerHTML = lines.join('<br>');
}

function focusOnRats() {
  const box = new THREE.Box3();
  let any = false;
  for (const id of ['rat1', 'rat2']) { const b = ratBboxLocal(id); if (b) { box.union(b); any = true; } }
  if (!any) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  const dist = Math.max(0.5, size * 1.6);
  const dir = camera.position.clone().sub(orbit.target).normalize();
  if (dir.lengthSq() < 1e-6) dir.set(1, 0.8, 1).normalize();
  orbit.target.copy(center);
  camera.position.copy(center).addScaledVector(dir, dist);
  camera.near = 0.005; camera.updateProjectionMatrix();
  orbit.update();
}

// =============================================================================
// Frame selection + playback (3D view only)
// =============================================================================

function setRatFrame(id, frame, fromSync = false) {
  const r = rats[id];
  if (!r.frames.includes(frame)) return;
  r.frame = frame;
  document.getElementById(`${id}Frame`).value = String(frame);
  r.info = null;
  loadRatMesh(id);
  if (id === 'rat1') currentFrame = frame;
  if (id === 'rat1' && !fromSync && document.getElementById('chkSync').checked && rats.rat2.frames.length) {
    const nearest = rats.rat2.frames.reduce((a, b) => Math.abs(b - frame) < Math.abs(a - frame) ? b : a);
    if (nearest !== rats.rat2.frame) setRatFrame('rat2', nearest, true);
  }
  updateFrameUi();
}

function stepRatFrame(id, delta) {
  const r = rats[id];
  if (r.frame == null) return;
  const i = r.frames.indexOf(r.frame);
  const j = Math.max(0, Math.min(r.frames.length - 1, i + delta));
  if (j !== i) setRatFrame(id, r.frames[j]);
}

function sliderRange() {
  const s = document.getElementById('frameSlider');
  return [parseInt(s.min, 10) || 0, parseInt(s.max, 10) || 0];
}

function updateFrameUi() {
  document.getElementById('frameVal').textContent = String(currentFrame);
  document.getElementById('frameMax').textContent = String(sliderRange()[1]);
  document.getElementById('timeVal').textContent = (currentFrame / fps).toFixed(2);
  updateMarkers();
}

function setFrame(f) {
  const [min, max] = sliderRange();
  currentFrame = Math.max(min, Math.min(max, Math.round(f)));
  document.getElementById('frameSlider').value = String(currentFrame);
  updateFrameUi();
}

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  lastTick = performance.now(); playAccum = 0;
}

function onKeyDown(ev) {
  if (ev.target && (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT')) return;
  if (ev.key === 'f' || ev.key === 'F') focusOnRats();
  if (ev.key === ' ') { ev.preventDefault(); togglePlay(); }
  if (ev.key === 'ArrowLeft') setFrame(currentFrame - (ev.shiftKey ? 10 : 1));
  if (ev.key === 'ArrowRight') setFrame(currentFrame + (ev.shiftKey ? 10 : 1));
}

// =============================================================================
// Analysis (from the masks, no per-frame placement)
// =============================================================================

function currentParams() {
  const num = (id) => parseFloat(document.getElementById(id).value);
  const params = {
    rest_ang_speed_dps: num('pAng'), rest_lin_speed_mps: num('pLin'), min_state_duration_s: num('pMinDur'),
    smoothing_s: num('pSmooth'), position_smoothing_s: num('pPosSmooth'),
    max_interp_gap_frames: Math.round(num('pGap')), min_confidence: num('pMinConf'),
  };
  const refRaw = document.getElementById('pRefFrame').value;
  if (refRaw !== '') params.reference_frame = Math.round(parseFloat(refRaw));
  return params;
}

function runAnalysis() {
  const frameStart = parseInt(document.getElementById('pFrameStart').value, 10);
  const frameEndRaw = document.getElementById('pFrameEnd').value;
  if (Number.isNaN(frameStart)) { setStatus('enter a first frame'); return; }
  if (frameEndRaw === '') { setStatus('enter a last frame'); return; }
  const frameEnd = parseInt(frameEndRaw, 10);
  if (frameEnd < frameStart + 2) { setStatus('need at least 3 frames in the window'); return; }
  // rat N = mask label N; each ticked rat gets its own run and rat<N>/ export folder
  const ratsToTrack = [1, 2].filter(n => document.getElementById(`chkTrack${n}`).checked);
  if (!ratsToTrack.length) { setStatus('tick rat 1 and/or rat 2 to track'); return; }

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  const logEl = document.getElementById('log');
  logEl.style.display = 'block'; logEl.textContent = 'starting…';
  const payload = { frame_start: frameStart, frame_end: frameEnd, fps, rats: ratsToTrack, params: currentParams() };
  fetch(API + '/api/analyze', { method: 'POST', body: JSON.stringify(payload) }).then(r => r.json())
    .then(res => {
      if (!res.ok) { logEl.textContent = 'could not start: ' + (res.error || '?'); btn.disabled = false; return; }
      // a failed poll (e.g. the network drive timing out on one request) is
      // retried rather than leaving the log stalled while the run goes on
      const retry = () => { setStatus('progress poll failed, retrying'); setTimeout(poll, 1000); };
      const poll = () => fetch(API + '/api/progress').then(r => r.json()).catch(() => null).then(p => {
        if (!p || !p.log) { retry(); return; }
        logEl.textContent = p.log.join('\n');
        logEl.scrollTop = logEl.scrollHeight;
        if (p.running) { setTimeout(poll, 500); return; }
        btn.disabled = false;
        if (p.error) { setStatus('analysis failed: ' + p.error); return; }
        if (p.result) showResult(p.result);
      });
      poll();
    }).catch(e => { console.error(e); btn.disabled = false; setStatus('analysis request failed'); });
}

function ratName(id) { return id === 'rat1' ? 'rat 1 (object 1, red)' : 'rat 2 (object 2, blue)'; }
function swatch(id) { return `<span class="swatch ${id}"></span>`; }
function summaryTable(rows) {
  return '<table class="summary">' + rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('') + '</table>';
}

function addPlots(container, relDir, plots, order) {
  for (const key of order) {
    if (!plots[key]) continue;
    const img = document.createElement('img');
    img.src = `${API}/exports/${relDir}/${plots[key]}?t=${Date.now()}`;
    img.alt = key;
    img.addEventListener('click', () => { document.getElementById('lightboxImg').src = img.src; document.getElementById('lightbox').style.display = 'flex'; });
    container.appendChild(img);
  }
}

// One rat's summary.json as table rows (same rows a single-rat run showed).
function ratSummaryRows(s) {
  const pct = (x) => (x * 100).toFixed(1) + ' %';
  const mf = s.mask_fusion || {};
  return [
    ['tracking', 'masks (no head-mounted IMU); mesh only as head fallback'],
    [`label ${mf.label}, min confidence`, `${mf.min_confidence}`],
    ['frames / duration', `${s.n_frames} / ${s.duration_s.toFixed(1)} s`],
    ['window', `${s.window.start_frame} → ${s.window.end_frame}`],
    ['present / reference frame', `${s.reference_frame}`],
    ['frames with mask position', `${s.position_coverage.frames_with_mesh} (${pct(s.position_coverage.fraction_with_position)} of frames covered incl. interpolation)`],
    ['frames with a fused heading', pct(mf.fraction_with_heading ?? 0)],
    ['mean cameras used, position / heading / head', `${(mf.mean_n_cams_position ?? 0).toFixed(1)} / ${(mf.mean_n_cams_heading ?? 0).toFixed(1)} / ${(mf.mean_n_cams_head ?? 0).toFixed(1)}`],
    ['head from masks / mesh / interpolated / missing', `${mf.frames_head_from_masks ?? 0} / ${mf.frames_head_from_mesh ?? 0} / ${mf.frames_head_interpolated ?? 0} / ${mf.frames_head_missing ?? 0} frames${mf.mesh_fallback_root ? '' : ' (no mesh fallback configured)'}`],
    ['head path length', `${s.path_length_m.head.toFixed(2)} m`],
    ['mean / max linear speed', s.linear_speed_mps.mean == null ? 'n/a' : `${s.linear_speed_mps.mean.toFixed(3)} / ${s.linear_speed_mps.max.toFixed(3)} m/s`],
    ['mean / max angular speed', s.angular_speed_dps.mean == null ? 'n/a' : `${s.angular_speed_dps.mean.toFixed(1)} / ${s.angular_speed_dps.max.toFixed(0)} deg/s`],
    ['rest', `${pct(s.state_fractions.rest)}, ${s.rest_bouts.count} bouts, mean ${s.rest_bouts.mean_duration_s.toFixed(1)} s, longest ${s.rest_bouts.longest_s.toFixed(1)} s`],
    ['head movement', pct(s.state_fractions.head_movement)],
    ['locomotion', pct(s.state_fractions.locomotion)],
  ];
}

// Results: the run folder's combined plots and rat-to-rat distances first,
// then each tracked rat's own summary and plots (from its rat<N>/ subfolder),
// each headed with its colour swatch.
function showResult(result) {
  lastResult = result;
  const panel = document.getElementById('results');
  panel.style.display = 'block';
  document.getElementById('resTitle').textContent = `${result.rel_dir}`;
  const between = (result.summary && result.summary.between_rats_m) || null;
  const cm = (x) => x == null ? 'n/a' : (x * 100).toFixed(1) + ' cm';
  const rows = [['rats tracked', result.rats.map(n => `${swatch(`rat${n}`)}rat ${n}`).join(' &nbsp;')]];
  if (between) {
    for (const [key, label] of [['head_to_head', 'head to head'], ['body_to_body', 'body centroid to body centroid']]) {
      const d = between[key];
      rows.push([`rat 1 to rat 2, ${label}: median / closest`, `${cm(d.median)} / ${cm(d.min)} (${d.n_frames} frames)`]);
    }
  }
  document.getElementById('resSummary').innerHTML = summaryTable(rows);
  const plots = document.getElementById('resPlots');
  plots.innerHTML = '';
  addPlots(plots, result.rel_dir, result.plots, ['path_top_view_both', 'camera_overlays']);
  const files = [`written to <b>${result.out_dir}</b><br>` + result.files.join(' &middot; ')];
  for (const [id, r] of Object.entries(result.per_rat)) {
    const block = document.createElement('div');
    block.className = 'rat-block';
    block.innerHTML = `<p class="rat-title">${swatch(id)}${ratName(id)}</p>` + summaryTable(ratSummaryRows(r.summary));
    plots.appendChild(block);
    addPlots(block, r.rel_dir, r.plots, ['path_top_view', 'head_orientation', 'head_kinematics', 'states']);
    files.push(`${swatch(id)}${r.rel_dir.split('/').pop()}/: ` + r.files.join(' &middot; '));
  }
  document.getElementById('resFiles').innerHTML = files.join('<br>');
  buildPaths(result);
  buildMarkers(result);
  // the scrubber now covers the analysed window, starting at the present / reference frame
  const slider = document.getElementById('frameSlider');
  const first = Object.values(result.per_rat)[0];
  slider.min = String(first.summary.window.start_frame);
  slider.max = String(first.summary.window.end_frame);
  setFrame(first.summary.reference_frame ?? first.summary.window.start_frame);
  setStatus(`analysis done: ${result.out_dir}`);
}

// Head paths in 3D, one per tracked rat, coloured by rat (red / blue) or by
// behavioural state ("colour paths by"); body-centroid paths in a lighter
// version of the rat's colour either way, so the two rats stay apart.
function buildPaths(result) {
  while (pathGroup.children.length) pathGroup.remove(pathGroup.children[0]);
  if (!result) return;
  const byState = document.getElementById('pathColor').value === 'state';
  for (const [id, r] of Object.entries(result.per_rat)) buildPath(r.viewer, id, byState);
  pathGroup.visible = document.getElementById('chkPath').checked;
}

function buildPath(viewer, id, byState) {
  if (!viewer) return;
  const headPts = [], headCols = [], bodyPts = [];
  const restStarts = [];
  const ratColor = new THREE.Color(RAT_COLORS[id]);
  let prevState = null;
  for (let i = 0; i < viewer.frame.length; i++) {
    const h = viewer.head[i], b = viewer.body[i];
    if (h[0] == null) { prevState = null; continue; }
    const p = remap(h);
    headPts.push(p);
    const c = byState ? new THREE.Color(STATE_COLORS[viewer.state[i]] || 0xffffff) : ratColor;
    headCols.push(c.r, c.g, c.b);
    if (b[0] != null) bodyPts.push(remap(b));
    if (viewer.state[i] === 'rest' && prevState !== 'rest') restStarts.push(p);
    prevState = viewer.state[i];
  }
  if (headPts.length > 1) {
    const geo = new THREE.BufferGeometry().setFromPoints(headPts);
    geo.setAttribute('color', new THREE.Float32BufferAttribute(headCols, 3));
    pathGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true })));
  }
  if (bodyPts.length > 1) {
    pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(bodyPts), new THREE.LineBasicMaterial({ color: ratLight(id) })));
  }
  if (byState && restStarts.length) {
    const m = new THREE.InstancedMesh(new THREE.SphereGeometry(0.008, 8, 8), new THREE.MeshBasicMaterial({ color: STATE_COLORS.rest }), restStarts.length);
    const mat = new THREE.Matrix4();
    restStarts.forEach((p, i) => { mat.makeTranslation(p.x, p.y, p.z); m.setMatrixAt(i, mat); });
    pathGroup.add(m);
  }
}

// One marker per tracked rat at the scrubber's frame: a sphere in the rat's
// colour (white rim) at its head, plus a white nose-direction arrow wherever
// the heading is known. Drawn on top of everything, since the head is often
// inside a mesh shown at another frame. Hidden outside the analysed window
// or where the head is unknown.
const markers = {}; // rat id -> { head: Group, nose: ArrowHelper, viewer }

function onTop(obj, order) {
  obj.traverse(o => { if (o.material) { o.material.depthTest = false; o.renderOrder = order; } });
  return obj;
}

function buildMarkers(result) {
  while (markerGroup.children.length) markerGroup.remove(markerGroup.children[0]);
  for (const id of Object.keys(markers)) delete markers[id];
  for (const [id, r] of Object.entries(result.per_rat)) {
    const head = new THREE.Group();
    head.add(onTop(new THREE.Mesh(new THREE.SphereGeometry(0.011, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.BackSide })), 10));
    head.add(onTop(new THREE.Mesh(new THREE.SphereGeometry(0.008, 16, 12), new THREE.MeshBasicMaterial({ color: RAT_COLORS[id] })), 11));
    const nose = onTop(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.05, 0xffffff, 0.016, 0.011), 12);
    markerGroup.add(head, nose);
    markers[id] = { head, nose, viewer: r.viewer };
  }
  updateMarkers();
}

// position in the (possibly downsampled) viewer arrays of a global frame, or -1
function viewerIndex(v, frame) {
  const n = v.frame.length;
  if (!n || frame < v.frame[0] || frame > v.frame[n - 1]) return -1;
  const step = n > 1 ? v.frame[1] - v.frame[0] : 1;
  return Math.min(n - 1, Math.round((frame - v.frame[0]) / step));
}

function updateMarkers() {
  for (const m of Object.values(markers)) {
    const i = viewerIndex(m.viewer, currentFrame);
    const h = i >= 0 ? m.viewer.head[i] : null;
    const f = i >= 0 ? m.viewer.fwd[i] : null;
    m.head.visible = m.nose.visible = false;
    if (!h || h[0] == null) continue;
    const p = remap(h);
    m.head.position.copy(p);
    m.head.visible = true;
    if (f && f[0] != null) {
      m.nose.position.copy(p);
      m.nose.setDirection(remap(f).normalize());
      m.nose.visible = true;
    }
  }
}

// =============================================================================
// Loop + wiring
// =============================================================================

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  if (playing) {
    const [min, max] = sliderRange();
    playAccum += (now - lastTick) / 1000 * fps * playSpeed;
    const steps = Math.floor(playAccum);
    if (steps > 0) {
      playAccum -= steps;
      let f = currentFrame + steps;
      if (f > max) f = min;
      setFrame(f);
    }
  }
  lastTick = now;
  orbit.update();
  renderer.render(scene, camera);
}

document.getElementById('focusBtn').addEventListener('click', focusOnRats);
document.getElementById('runBtn').addEventListener('click', runAnalysis);

document.getElementById('rat1Frame').addEventListener('change', (e) => setRatFrame('rat1', parseInt(e.target.value, 10)));
document.getElementById('rat2Frame').addEventListener('change', (e) => setRatFrame('rat2', parseInt(e.target.value, 10)));
document.getElementById('rat1Prev').addEventListener('click', () => stepRatFrame('rat1', -1));
document.getElementById('rat1Next').addEventListener('click', () => stepRatFrame('rat1', 1));
document.getElementById('rat2Prev').addEventListener('click', () => stepRatFrame('rat2', -1));
document.getElementById('rat2Next').addEventListener('click', () => stepRatFrame('rat2', 1));
document.getElementById('chkRat2').addEventListener('change', (e) => { if (e.target.checked) loadRatMesh('rat2'); else { clearRat('rat2'); } applyPlacementMode(); });
document.querySelectorAll('input[name="placeMode"]').forEach(r => r.addEventListener('change', () => { placeModeChosen = true; applyPlacementMode(); }));
document.getElementById('chkRig').addEventListener('change', (e) => { rigVisuals.visible = e.target.checked; });
document.getElementById('chkSync').addEventListener('change', (e) => { if (e.target.checked && rats.rat1.frame != null) setRatFrame('rat1', rats.rat1.frame); });
document.getElementById('chkPath').addEventListener('change', (e) => { pathGroup.visible = e.target.checked; });
document.getElementById('pathColor').addEventListener('change', () => buildPaths(lastResult));

document.getElementById('frameSlider').addEventListener('input', (e) => setFrame(parseInt(e.target.value, 10)));
document.getElementById('playBtn').addEventListener('click', togglePlay);
document.querySelectorAll('#panel button[data-speed]').forEach(b => b.addEventListener('click', () => {
  playSpeed = parseFloat(b.dataset.speed);
  document.querySelectorAll('#panel button[data-speed]').forEach(x => x.classList.toggle('active', x === b));
}));

window.__enterScene = init;
