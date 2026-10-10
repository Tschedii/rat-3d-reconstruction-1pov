import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

// Same origin as the page when served by server.py.
const API = location.protocol.startsWith('http') ? location.origin : 'http://127.0.0.1:8975';

// Same z-up (table frame) -> y-up (three.js) remap as GUI 5.1's app.js, reused
// verbatim for consistency: local = (x, z, -y), table = (x, -z, y). A z-up
// quaternion's vector part remaps the same way its w stays put.
function remapVec(v) { return new THREE.Vector3(v[0], v[2], -v[1]); }
function quatZupToThree(w, x, y, z) { return new THREE.Quaternion(x, z, -y, w); }

// rat n = object n in the masks. Rat 1 is red, rat 2 blue: skull, trajectory,
// head marker and legend, the same colours server.py tints the masks with
const RAT_HEX = { 1: 0xff2828, 2: 0x286eff };
const RAT_CSS = { 1: 'rgb(255,40,40)', 2: 'rgb(40,110,255)' };
const RAT_NAME = { 1: 'red', 2: 'blue' };

let meta = null;
// one entry per analysed rat: { id, info (its meta.rats entry), series, byFrame, inset }
let rats = [];
let selectedId = 1;               // the rat the head tools (click, rotate, chips, reach) act on
let currentFrame = 0, playing = false, playSpeed = 1, lastTick = 0, playAccum = 0;

const TRAIL_SECONDS = 3;
let alignDeg = [0, 0, 0];   // fixed mesh-to-head-frame offset, user-tunable (see #alignRow)
let saveAlignTimer = null;
let capGeometry = null, capUrl = null;   // the cap scan: loaded once, drawn in every rat's inset

function ratById(id) { return rats.find(r => r.id === id); }
function selectedRat() { return ratById(selectedId) || rats[0]; }

// one 3D inset per rat: the head's orientation with the cap mesh in the rat's colour
function makeInset(rat) {
  const wrap = document.createElement('div');
  wrap.className = 'inset';
  wrap.dataset.rat = String(rat.id);
  wrap.innerHTML = `<div class="insetLabel"></div><div class="insetLegend"><span class="insetSwatch"></span><span></span></div>`;
  wrap.querySelector('.insetSwatch').style.background = RAT_CSS[rat.id];
  wrap.querySelector('.insetLegend span:last-child').textContent = legendText(rat);
  document.getElementById('insets').appendChild(wrap);
  const w = wrap.clientWidth || 230, h = wrap.clientHeight || 230;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, w / h, 0.01, 10);
  // seen from the same side as the top-view video: from above, tilted from
  // the image's bottom edge, so table +x is screen right and table +y is
  // screen up (three (0, 0.23, 0.12) = table (0, -0.12, 0.23)); a view from
  // any other side makes the head's left/right tilt read mirrored
  camera.position.set(0, 0.23, 0.12);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);       // CSS sizes the canvas; a ResizeObserver keeps the buffer in step
  wrap.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.8); key.position.set(0.3, 0.4, 0.2); scene.add(key);
  const fill = new THREE.DirectionalLight(0xaac8ff, 0.5); fill.position.set(-0.2, 0.1, -0.3); scene.add(fill);
  scene.add(new THREE.GridHelper(0.16, 4, 0x4a4838, 0x2a2820));

  // head-local axes (x=nose, y=left, z=up), rotated live by the per-frame
  // world quaternion. Directions below are this group's local axes at
  // identity, chosen so they line up with the remapped table axes: local +x
  // = table +x = nose, local +y = table +z = up, local -z = table +y = left
  // (remap sends table +y to three -z, see remapVec above). Not red/blue:
  // those are the rats' colours.
  const axesGroup = new THREE.Group();
  const mkArrow = (dir, color) => new THREE.ArrowHelper(dir.normalize(), new THREE.Vector3(0, 0, 0), 0.055, color, 0.018, 0.01);
  axesGroup.add(mkArrow(new THREE.Vector3(1, 0, 0), 0xeda100));   // nose
  axesGroup.add(mkArrow(new THREE.Vector3(0, 1, 0), 0xd9d4c5));   // up
  axesGroup.add(mkArrow(new THREE.Vector3(0, 0, -1), 0x4ae37a));  // left
  scene.add(axesGroup);

  // the cap mesh (Cap_Cedric.stl, decimated + centred + mm->m, see the chat
  // that asked for this): capGroup gets the SAME live head quaternion as
  // axesGroup; alignGroup nested inside it is a fixed, user-tunable offset
  // (#alignRow) correcting for the scan's arbitrary native orientation --
  // there is no way to derive "which way is nose" from the cap's shape alone,
  // so this is calibrated by eye against the reference arrows, the same way
  // GUI 5.1's sensor gizmo itself is calibrated against the visible mesh.
  const capGroup = new THREE.Group();
  const alignGroup = new THREE.Group();
  capGroup.add(alignGroup);
  scene.add(capGroup);

  const trailMaxPts = Math.round(TRAIL_SECONDS * 60) + 5;
  const geo = new THREE.BufferGeometry();
  const trailPositions = new Float32Array(trailMaxPts * 3);
  geo.setAttribute('position', new THREE.BufferAttribute(trailPositions, 3));
  geo.setDrawRange(0, 0);
  const trailLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: RAT_HEX[rat.id], transparent: true, opacity: 0.85 }));
  trailLine.visible = document.getElementById('showTrail').checked;   // off by default: only the head's rotation
  scene.add(trailLine);

  const observer = new ResizeObserver(() => {
    const cw = wrap.clientWidth, ch = wrap.clientHeight;
    if (exporting || !cw || !ch) return;
    renderer.setSize(cw, ch, false); camera.aspect = cw / ch; camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  });
  observer.observe(wrap);

  rat.inset = { wrap, scene, camera, renderer, axesGroup, capGroup, alignGroup, trailLine, trailPositions, trailMaxPts,
                observer, label: wrap.querySelector('.insetLabel'), cap: null };
  setInsetLabel(rat);
  attachRotate(rat);
  if (capGeometry) addCap(rat);
  renderer.render(scene, camera);
}

function dropInset(rat) {
  const ins = rat.inset;
  if (!ins) return;
  ins.observer.disconnect();
  ins.renderer.dispose();
  ins.wrap.remove();
  rat.inset = null;
}

function addCap(rat) {
  const ins = rat.inset;
  if (!ins || ins.cap) return;
  ins.cap = new THREE.Mesh(capGeometry, new THREE.MeshStandardMaterial({ color: RAT_HEX[rat.id], roughness: 0.6, metalness: 0.05 }));
  ins.alignGroup.add(ins.cap);
}

function loadCap() {
  if (!meta.cap_mesh_url) return;
  if (capGeometry && capUrl === meta.cap_mesh_url) { rats.forEach(addCap); return; }
  capUrl = meta.cap_mesh_url;
  new STLLoader().load(`${API}${meta.cap_mesh_url}`, (geometry) => {
    // the unprocessed scan (Cap_Cedric.stl) is in mm and not centred: centre
    // it on its bounding box and, since a rat's cap is not a metre across,
    // read a size above 1 as mm
    geometry.computeBoundingBox();
    const centre = geometry.boundingBox.getCenter(new THREE.Vector3());
    const size = geometry.boundingBox.getSize(new THREE.Vector3());
    geometry.translate(-centre.x, -centre.y, -centre.z);
    if (Math.max(size.x, size.y, size.z) > 1) geometry.scale(0.001, 0.001, 0.001);
    geometry.computeVertexNormals();
    capGeometry = geometry;
    // arrives after the first render; while paused nothing else would redraw
    rats.forEach(r => { addCap(r); r.inset.renderer.render(r.inset.scene, r.inset.camera); });
  });
}

// nose heading (deg, table frame) of a head quaternion wxyz
function headingOfQ([w, x, y, z]) {
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) * 180 / Math.PI;
}

function tableToPx([x, y]) {
  const [xmin, xmax, ymin, ymax] = meta.view_box;
  const [W, H] = meta.out_size;
  const px = (x - xmin) / (xmax - xmin) * W;
  const py = H - (y - ymin) / (ymax - ymin) * H;
  return [px, py];
}

function updateOverlay(frame) {
  const canvas = document.getElementById('overlayCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawOverlay(ctx, frame);
}

// every rat's trajectory, head marker + nose arrow (and, on a clicked frame,
// the click), in output px; the selected rat last, on top. skipId: leave that
// rat out (it is being previewed)
function drawOverlay(ctx, frame, { markers = true, paths = showPaths(), skipId = null } = {}) {
  const order = rats.filter(r => r.id !== selectedId).concat(rats.filter(r => r.id === selectedId));
  if (paths) for (const r of order) if (r.id !== skipId) drawPath(ctx, r, frame);
  if (markers) for (const r of order) if (r.id !== skipId) drawMarker(ctx, r, frame);
}

function showPaths() { return document.getElementById('showPath').checked; }
function pathSeconds() { return Math.max(0, parseFloat(document.getElementById('pathSeconds').value) || 0); }

// the head's trajectory over the last pathSeconds(), where it appears in the video
function drawPath(ctx, rat, frame) {
  const s = rat.series, idx = rat.byFrame.get(frame);
  if (idx == null) return;
  const start = Math.max(0, idx - Math.round(pathSeconds() * meta.fps));
  ctx.save();
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  for (const [color, width] of [['rgba(0,0,0,0.45)', 4], [RAT_CSS[rat.id], 2]]) {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath();
    let pen = false;
    for (let i = start; i <= idx; i++) {
      const p = s.has_pos[i] ? s.head_top[i] : null;
      if (!p) { pen = false; continue; }
      const [x, y] = tableToPx(p);
      if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawArrow(ctx, x0, y0, x1, y1, color) {
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
}

function drawDot(ctx, x, y, fill, edge = '#0b0b0b') {
  ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fillStyle = fill; ctx.strokeStyle = edge; ctx.lineWidth = 1.5;
  ctx.fill(); ctx.stroke();
}

function drawMarker(ctx, rat, frame) {
  const s = rat.series, idx = rat.byFrame.get(frame);
  if (idx == null) return;
  const color = RAT_CSS[rat.id];
  const click = (rat.info.head_clicks || []).find(c => c.frame === frame);
  if (click && click.x != null) {
    const [cx, cy] = tableToPx([click.x, click.y]);
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx - 10, cy); ctx.lineTo(cx + 10, cy); ctx.moveTo(cx, cy - 10); ctx.lineTo(cx, cy + 10); ctx.stroke();
    const dir = click.heading_deg != null ? click.heading_deg
      : click.q ? headingOfQ(click.q) : null;
    if (dir != null) {
      const a = dir * Math.PI / 180;                           // table frame: y up, canvas y down
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * 60, cy - Math.sin(a) * 60); ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  if (!s.has_pos[idx] || !s.head_top[idx]) return;
  // head_top / nose_top are where the head point and a point 4 cm along the
  // nose APPEAR in the warped video (projected into the camera, then back
  // through the table-plane warp), not their true table xy: the head is
  // ~10 cm above the table, so its true xy would sit off the rat's image.
  const [px, py] = tableToPx(s.head_top[idx]);
  const [nx, ny] = tableToPx(s.nose_top[idx]);
  const L = 42, n = Math.hypot(nx - px, ny - py) || 1;
  drawArrow(ctx, px, py, px + (nx - px) / n * L, py + (ny - py) / n * L, color);
  // the rat the head tools edit gets a white rim while a tool is on
  drawDot(ctx, px, py, color, rat.id === selectedId && (clickMode || rotateMode) ? '#ffffff' : '#0b0b0b');
}

// draws the rat's head at this frame into its inset; false if its IMU has no row here
function updateInset(rat, frame) {
  const ins = rat.inset, s = rat.series;
  const idx = rat.byFrame.get(frame);
  ins.capGroup.visible = idx != null;
  ins.axesGroup.visible = idx != null && document.getElementById('showAxes').checked;
  if (idx == null) {
    ins.trailLine.geometry.setDrawRange(0, 0);
    ins.renderer.render(ins.scene, ins.camera);
    return false;
  }
  const q = s.q[idx];
  const threeQ = quatZupToThree(q[0], q[1], q[2], q[3]);
  ins.axesGroup.quaternion.copy(threeQ);
  ins.capGroup.quaternion.copy(threeQ);

  if (s.has_pos[idx]) {
    const span = Math.round(TRAIL_SECONDS * meta.fps);
    const start = Math.max(0, idx - span);
    const origin = s.head[idx];
    let n = 0;
    for (let i = start; i <= idx && n < ins.trailMaxPts; i++) {
      if (!s.has_pos[i]) continue;
      const h = s.head[i];
      const p = remapVec([h[0] - origin[0], h[1] - origin[1], h[2] - origin[2]]);
      ins.trailPositions[n * 3] = p.x; ins.trailPositions[n * 3 + 1] = p.y; ins.trailPositions[n * 3 + 2] = p.z;
      n++;
    }
    ins.trailLine.geometry.setDrawRange(0, n);
    ins.trailLine.geometry.attributes.position.needsUpdate = true;
  }
  ins.renderer.render(ins.scene, ins.camera);
  return true;
}

function redraw(frame = shownFrame ?? currentFrame) {
  updateOverlay(frame);
  rats.forEach(r => updateInset(r, frame));
}

// which rat a skull view shows, in its colour (= its mask colour)
function legendText(rat) { return `rat ${rat.id}: object ${rat.id} (${RAT_NAME[rat.id]})`; }

function setInsetLabel(rat) {
  rat.inset.label.textContent = rotateMode && rat.id === selectedId
    ? 'drag to rotate the head · Shift = about vertical' : `rat ${rat.id} head orientation`;
}

// the same legend drawn into an export frame; (x, y) = bottom-left of the
// view it belongs to, s = scale (1 = the 230 px inset)
function drawLegend(ctx, rat, x, y, s) {
  const text = legendText(rat);
  const fs = Math.round(10 * s), pad = Math.round(6 * s), sw = Math.round(10 * s);
  ctx.font = `${fs}px ui-monospace, Menlo, monospace`;
  const w = pad * 3 + sw + ctx.measureText(text).width, h = fs + pad * 2;
  ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(x, y - h, w, h);
  ctx.fillStyle = RAT_CSS[rat.id]; ctx.fillRect(x + pad, y - h / 2 - sw / 2, sw, sw);
  ctx.fillStyle = '#f1ede1'; ctx.fillText(text, x + pad * 2 + sw, y - pad - Math.round(fs * 0.15));
}

function frameUrl(frame) {
  const masks = document.getElementById('showMasks').checked ? 1 : 0;
  return `${API}/api/topview_frame.jpg?frame=${frame}&masks=${masks}`;
}

function setFrame(f) {
  currentFrame = Math.max(meta.frame_min, Math.min(meta.frame_max, Math.round(f)));
  document.getElementById('frameVal').textContent = String(currentFrame);
  document.getElementById('timeVal').textContent = (currentFrame / meta.fps).toFixed(2) + 's';
  document.getElementById('frameSlider').value = String(currentFrame);
  const img = document.getElementById('videoImg');
  img.dataset.frame = String(currentFrame);
  img.src = frameUrl(currentFrame);
  // overlay + insets are drawn in the image's load handler, for the frame
  // that is actually on screen: decoding/warping a video frame takes longer
  // than a playback step, so drawing here would run ahead of the picture
}

// fires only for the latest src (a superseded request never loads), so the
// markers and the 3D insets always describe the frame being shown
let shownFrame = null;   // the frame actually on screen (clicks refer to it)
document.getElementById('videoImg').addEventListener('load', (e) => {
  if (!rats.length || exporting || rot) return;
  shownFrame = parseInt(e.target.dataset.frame, 10);
  redraw(shownFrame);
});

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  lastTick = performance.now(); playAccum = 0;
}

function animate() {
  requestAnimationFrame(animate);
  if (!playing || !meta) return;
  const now = performance.now();
  playAccum += (now - lastTick) / 1000 * meta.fps * playSpeed;
  lastTick = now;
  const steps = Math.floor(playAccum);
  if (steps > 0) {
    playAccum -= steps;
    let f = currentFrame + steps;
    if (f > meta.frame_max) f = meta.frame_min;
    setFrame(f);
  }
}

function ratStatus(rat) {
  const s = rat.info.summary || {}, v = s.validation || {};
  const f2 = (x) => (x == null ? 'n/a' : x.toFixed(2));
  const f0 = (x) => (x == null ? '?' : x.toFixed(0));
  return `rat ${rat.id} (${RAT_NAME[rat.id]}, IMU ${rat.info.quat_file}, frames ${rat.info.frame_range[0]}–${rat.info.frame_range[1]}): ` +
    `position on ${s.frames_with_position}/${s.n_frames} frames, ${f0(v.cameras_used_median)} of ${s.tracking_cameras.length} cameras per frame, ` +
    `residual ${v.triangulation_residual_px_median.toFixed(1)} px, frame ${rat.info.anchor_frame} ` +
    `${v.triangulated_vs_mesh_centroid_cm != null ? v.triangulated_vs_mesh_centroid_cm.toFixed(1) + ' cm' : '?'} from the mesh\n  ` +
    (s.anchor_every_frames ? `heading re-aligned every ${s.anchor_every_frames} frames (${s.heading_anchors.length} anchors)` : 'heading from the eyes at the anchor only') +
    (s.mount_tilt && s.mount_tilt.mode !== 'off' ? ` · mounting tilt removed: roll ${s.mount_tilt.roll_deg.toFixed(1)}°, pitch ${s.mount_tilt.pitch_deg.toFixed(1)}° (${s.mount_tilt.mode})` : ' · no mounting tilt correction') +
    ` · agreement with walking direction: single anchor ${f2(v.heading_vs_travel_R_single_anchor)}, ` +
    `anchors held-out ${f2(v.heading_vs_travel_R_anchors_heldout)} · at frame ${rat.info.anchor_frame} ${f0(v.anchor_heading_minus_eyes_deg)}° off the eyes`;
}

async function enterPlayback(m, keepFrame = null) {
  meta = m;
  const canvas = document.getElementById('overlayCanvas');
  // canvas.width/height is the drawing-buffer resolution (matches the warped
  // image's own pixels, for crisp overlay drawing); #videoWrap's on-screen
  // size is handled entirely by CSS (responsive to the viewport) and both
  // #videoImg and this canvas stretch to fill it, so no inline sizing here.
  canvas.width = meta.out_size[0]; canvas.height = meta.out_size[1];

  const data = await (await fetch(`${API}/api/series`)).json();
  rats.forEach(dropInset);           // re-entered after "Change setup" or a re-run
  rats = meta.rats.map((info) => {
    const series = data.rats.find(s => s.rat === info.id);
    return { id: info.id, info, series, byFrame: new Map(series.frame.map((f, i) => [f, i])), inset: null };
  });
  if (!ratById(selectedId)) selectedId = rats[0].id;

  document.getElementById('status').textContent =
    `view: ${meta.view_mode === 'median' ? 'median of ' + meta.view_cameras.join(', ') : meta.view_mode} (videos in ${meta.video_root}) · ` +
    `floor at z = ${(meta.floor_z_m * 100).toFixed(1)} cm · frames ${meta.frame_min}–${meta.frame_max} @ ${meta.fps}fps\n` +
    rats.map(ratStatus).join('\n');

  const slider = document.getElementById('frameSlider');
  slider.min = String(meta.frame_min); slider.max = String(meta.frame_max);

  alignDeg = (meta.cap_align_deg || [0, 0, 0]).slice();
  ['alignX', 'alignY', 'alignZ'].forEach((id, i) => { document.getElementById(id).value = String(alignDeg[i]); });
  rats.forEach(makeInset);
  loadCap();
  applyAlignment();
  buildRatPicker();
  buildExportKinds();
  setSelected(selectedId);
  document.getElementById('expFrom').value = String(meta.frame_min);
  document.getElementById('expTo').value = String(meta.frame_max);
  if (!document.getElementById('expDir').value) document.getElementById('expDir').value = meta.export_dir;
  setFrame(keepFrame ?? (meta.position_frame_range ? meta.position_frame_range[0] : meta.frame_min));
}

// ---------------------------------------------------------------------------
// which rat the head tools edit
// ---------------------------------------------------------------------------
function buildRatPicker() {
  const box = document.getElementById('ratPick');
  box.innerHTML = '';
  for (const r of rats) {
    const b = document.createElement('button');
    b.className = 'ratBtn';
    b.dataset.rat = String(r.id);
    b.title = `edit rat ${r.id}'s head (key ${r.id})`;
    b.innerHTML = `<span class="ratSwatch"></span>rat ${r.id}`;
    b.querySelector('.ratSwatch').style.background = RAT_CSS[r.id];
    b.addEventListener('click', () => setSelected(r.id));
    box.appendChild(b);
  }
}

function setSelected(id) {
  if (!ratById(id)) return;
  selectedId = id;
  document.querySelectorAll('.ratBtn').forEach(b => b.classList.toggle('on', parseInt(b.dataset.rat, 10) === id));
  rats.forEach(r => { r.inset.wrap.classList.toggle('selected', r.id === id); setInsetLabel(r); });
  document.getElementById('headReach').value = String(selectedRat().info.head_click_reach_frames ?? 30);
  renderHeadChips();
  if (clickMode) setClickMode(true);
  if (rotateMode) setRotateMode(true);
  if (shownFrame != null && !exporting) updateOverlay(shownFrame);
}

// ---------------------------------------------------------------------------
// head position from clicks
// ---------------------------------------------------------------------------
let clickMode = false;

function setClickMode(on) {
  clickMode = on;
  document.getElementById('headClickBtn').classList.toggle('on', on);
  document.getElementById('videoWrap').classList.toggle('clickMode', on);
  if (on && playing) togglePlay();
  headMsg(on ? `click rat ${selectedId}'s head (${RAT_NAME[selectedId]}) = position; press on the head and drag towards the nose = position + direction (arrow keys step frames, 1 / 2 picks the rat)` : '');
  if (meta && shownFrame != null && !exporting) updateOverlay(shownFrame);
}

function headMsg(text, err = false) {
  const el = document.getElementById('headMsg');
  el.textContent = text; el.classList.toggle('err', err);
}

function fitText(rat) {
  const fit = rat.info.summary?.head_fit;
  const who = `rat ${rat.id}: `;
  if (!fit || (!fit.n_clicks && !fit.n_heading_clicks && !fit.n_orientation_clicks)) return who + 'no clicks: position from the marked eyes, orientation from the IMU';
  const nOri = (fit.n_heading_clicks || 0) + (fit.n_orientation_clicks || 0);
  if (!fit.n_clicks) return who + `${nOri} orientation fix${nOri > 1 ? 'es' : ''} saved; position from the marked eyes`;
  return who + `${fit.n_clicks} position click${fit.n_clicks > 1 ? 's' : ''} saved (${nOri} orientation fix${nOri === 1 ? '' : 'es'}): rigid-offset error ${fit.rms_before_cm.toFixed(1)} → ${fit.rms_after_cm.toFixed(1)} cm`
    + (rat.info.head_click_reach_frames > 0 ? ', exact at the clicks' : '');
}

function renderHeadChips() {
  const rat = selectedRat();
  const box = document.getElementById('headChips');
  box.innerHTML = '';
  for (const c of rat.info.head_clicks || []) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.style.borderColor = RAT_CSS[rat.id];
    chip.title = `rat ${rat.id}, jump to frame ${c.frame}: ` + [c.x != null ? 'position' : null,
      c.q ? 'orientation (3D view)' : c.heading_deg != null ? `direction ${c.heading_deg.toFixed(0)}°` : null].filter(Boolean).join(' + ');
    const icons = (c.x != null ? '' : ' ○') + (c.q ? ' ⟳' : c.heading_deg != null ? ' ➶' : '');
    chip.innerHTML = `${c.frame}${icons}<span class="x" title="remove">×</span>`;
    chip.addEventListener('click', (e) => {
      if (e.target.classList.contains('x')) {
        postClicks({ clicks: rat.info.head_clicks.filter(k => k.frame !== c.frame) });
      } else {
        if (playing) togglePlay();
        setFrame(c.frame);
      }
    });
    box.appendChild(chip);
  }
  if (!clickMode && !rotateMode) headMsg(fitText(rat));
}

function applyHeadResponse(r) {
  const rat = ratById(r.rat);
  rat.info.head_clicks = r.head_clicks;
  rat.info.head_click_reach_frames = r.head_click_reach_frames;
  rat.info.summary.head_fit = r.fit;
  const s = rat.series;
  s.head = r.head; s.head_top = r.head_top; s.nose_top = r.nose_top; s.q = r.q;
  if (rat.id === selectedId) renderHeadChips();
  redraw();
}

async function postClicks(body) {
  try {
    const r = await (await fetch(`${API}/api/head_clicks`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rat: selectedId, ...body }) })).json();
    if (!r.ok) { headMsg(r.error || 'failed', true); return; }
    applyHeadResponse(r);
  } catch (e) { headMsg(String(e), true); }
}

// press on the head = position; drag towards the nose = direction as well.
// While dragging, the marker and arrow are previewed where they will go.
let drag = null;
const DRAG_MIN_PX = 12;

function eventPx(e) {
  const rect = document.getElementById('videoImg').getBoundingClientRect();
  return [(e.clientX - rect.left) / rect.width * meta.out_size[0], (e.clientY - rect.top) / rect.height * meta.out_size[1]];
}

function drawDragPreview() {
  const canvas = document.getElementById('overlayCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawOverlay(ctx, drag.frame, { skipId: drag.rat.id });
  const color = RAT_CSS[drag.rat.id];
  const [x0, y0] = drag.from, [x1, y1] = drag.to;
  const len = Math.hypot(x1 - x0, y1 - y0);
  if (len >= DRAG_MIN_PX) {
    drawArrow(ctx, x0, y0, x0 + (x1 - x0) / len * 42, y0 + (y1 - y0) / len * 42, color);
  } else {
    // position only: keep today's direction, moved to the new spot
    const s = drag.rat.series, idx = drag.rat.byFrame.get(drag.frame);
    if (idx != null && s.head_top[idx] && s.nose_top[idx]) {
      const [hx, hy] = tableToPx(s.head_top[idx]), [nx, ny] = tableToPx(s.nose_top[idx]);
      const n = Math.hypot(nx - hx, ny - hy) || 1;
      drawArrow(ctx, x0, y0, x0 + (nx - hx) / n * 42, y0 + (ny - hy) / n * 42, color);
    }
  }
  drawDot(ctx, x0, y0, '#ffffff');
}

const videoImgEl = document.getElementById('videoImg');
videoImgEl.draggable = false;
videoImgEl.addEventListener('pointerdown', (e) => {
  if (!clickMode || !meta || exporting || e.button !== 0) return;
  e.preventDefault();
  try { videoImgEl.setPointerCapture(e.pointerId); } catch (err) { /* keep going without capture */ }
  const p = eventPx(e);
  drag = { from: p, to: p, frame: shownFrame ?? currentFrame, rat: selectedRat() };
  drawDragPreview();
});
videoImgEl.addEventListener('pointermove', (e) => {
  if (!drag) return;
  drag.to = eventPx(e);
  drawDragPreview();
});
videoImgEl.addEventListener('pointercancel', () => {
  if (!drag) return;
  const frame = drag.frame; drag = null;
  updateOverlay(frame);
});
videoImgEl.addEventListener('pointerup', async () => {
  if (!drag) return;
  const d = drag; drag = null;
  const [px, py] = d.from, [px2, py2] = d.to;
  const body = { rat: d.rat.id, frame: d.frame, px, py };
  if (Math.hypot(px2 - px, py2 - py) >= DRAG_MIN_PX) Object.assign(body, { px2, py2 });
  try {
    const r = await (await fetch(`${API}/api/head_click`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
    if (!r.ok) {
      headMsg(r.error || 'failed', true);
      updateOverlay(d.frame);
      return;
    }
    applyHeadResponse(r);
    headMsg(`rat ${d.rat.id} frame ${d.frame} saved (${body.px2 != null ? 'position + direction' : 'position'}) · ` + fitText(d.rat));
  } catch (err) { headMsg(String(err), true); }
});
document.getElementById('headClickBtn').addEventListener('click', () => setClickMode(!clickMode));
document.getElementById('headClearBtn').addEventListener('click', () => {
  const rat = selectedRat();
  const n = (rat.info.head_clicks || []).length;
  if (n && confirm(`Remove all ${n} head clicks of rat ${rat.id}?`)) postClicks({ clicks: [] });
});
document.getElementById('headReach').addEventListener('change', (e) => postClicks({ reach: Math.max(0, parseInt(e.target.value, 10) || 0) }));
window.addEventListener('keydown', (e) => {
  if (!meta || e.target.closest('input, select, textarea')) return;
  if (e.key === 'h' || e.key === 'H') { if (rotateMode) setRotateMode(false); setClickMode(!clickMode); }
  if (e.key === 'r' || e.key === 'R') setRotateMode(!rotateMode);
  if (e.key === '1' || e.key === '2') setSelected(parseInt(e.key, 10));
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    if (playing) togglePlay();
    setFrame(currentFrame + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 10 : 1));
    e.preventDefault();
  }
});

// ---------------------------------------------------------------------------
// rotate a head in its 3D inset = re-align that rat's gyroscope there
// ---------------------------------------------------------------------------
let rotateMode = false, rot = null;
const ROT_RAD_PER_PX = 0.01;

function setRotateMode(on) {
  rotateMode = on;
  document.getElementById('headRotateBtn').classList.toggle('on', on);
  rats.forEach(r => { r.inset.wrap.classList.toggle('rotating', on); setInsetLabel(r); });
  if (on) { if (playing) togglePlay(); if (clickMode) setClickMode(false); }
  headMsg(on ? `drag a head in its 3D view (rat ${selectedId}'s is selected; dragging the other one switches); on release it is saved for `
    + (document.getElementById('headRotateScope').value === 'mount' ? 'ALL frames (that rat\'s IMU mounting, re-runs the analysis)' : 'this frame ± reach') : '');
  if (!on) renderHeadChips();
  if (meta && shownFrame != null && !exporting) updateOverlay(shownFrame);
}

// live preview in the top view: marker + arrow along the rotated nose
function drawRotatePreview(rat, idx, q3) {
  const canvas = document.getElementById('overlayCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawOverlay(ctx, rat.series.frame[idx], { skipId: rat.id });
  if (!rat.series.head_top[idx]) return;
  const [px, py] = tableToPx(rat.series.head_top[idx]);
  const v = new THREE.Vector3(1, 0, 0).applyQuaternion(q3);        // three: table (x, -z, y)
  const dx = v.x, dy = -v.z, n = Math.hypot(dx, dy) || 1;
  drawArrow(ctx, px, py, px + dx / n * 42, py - dy / n * 42, RAT_CSS[rat.id]);
  drawDot(ctx, px, py, '#ffffff');
}

function attachRotate(rat) {
  const el = rat.inset.wrap;
  el.addEventListener('pointerdown', (e) => {
    if (!rotateMode || !meta || exporting || e.button !== 0) return;
    const frame = shownFrame ?? currentFrame, idx = rat.byFrame.get(frame);
    if (idx == null) { headMsg(`rat ${rat.id} has no IMU row at frame ${frame}`, true); return; }
    e.preventDefault();
    if (rat.id !== selectedId) setSelected(rat.id);
    try { el.setPointerCapture(e.pointerId); } catch (err) { /* fine without */ }
    const ins = rat.inset;
    rot = { rat, frame, idx, x: e.clientX, y: e.clientY, q0: ins.capGroup.quaternion.clone(), q: ins.capGroup.quaternion.clone(), moved: false };
  });
  el.addEventListener('pointermove', (e) => {
    if (!rot || rot.rat !== rat) return;
    const dx = e.clientX - rot.x, dy = e.clientY - rot.y, d = Math.hypot(dx, dy);
    if (d < 2) return;
    rot.moved = true;
    const ins = rat.inset;
    let dq;
    if (e.shiftKey) {                                   // about vertical (three +y) only
      dq = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), dx * ROT_RAD_PER_PX);
    } else {                                            // trackball: axis perpendicular to the drag, in view space
      const axis = new THREE.Vector3(dy / d, dx / d, 0).applyQuaternion(ins.camera.quaternion);
      dq = new THREE.Quaternion().setFromAxisAngle(axis, d * ROT_RAD_PER_PX);
    }
    rot.q = dq.multiply(rot.q0.clone());
    ins.axesGroup.quaternion.copy(rot.q); ins.capGroup.quaternion.copy(rot.q);
    ins.renderer.render(ins.scene, ins.camera);
    drawRotatePreview(rat, rot.idx, rot.q);
  });
  el.addEventListener('pointerup', () => endRotate(rat, false));
  el.addEventListener('pointercancel', () => endRotate(rat, true));
}

async function endRotate(rat, cancelled) {
  if (!rot || rot.rat !== rat) return;
  const r0 = rot; rot = null;
  if (cancelled || !r0.moved) { redraw(r0.frame); return; }
  const q = [r0.q.w, r0.q.x, -r0.q.z, r0.q.y];         // three (x, y, z, w) -> table wxyz
  const scope = document.getElementById('headRotateScope').value;
  try {
    const r = await (await fetch(`${API}/api/head_orient`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rat: rat.id, frame: r0.frame, q, scope }) })).json();
    if (!r.ok) { headMsg(r.error || 'failed', true); redraw(r0.frame); return; }
    if (r.rerun) {
      headMsg(`rat ${rat.id} IMU mounting set to roll ${r.mount_roll_deg.toFixed(1)}°, pitch ${r.mount_pitch_deg.toFixed(1)}° for all frames; re-running…`);
      await reloadAfterRerun(r0.frame);
      headMsg(`rat ${rat.id} IMU mounting saved: roll ${r.mount_roll_deg.toFixed(1)}°, pitch ${r.mount_pitch_deg.toFixed(1)}° (manual in the setup form)`
        + (Math.abs(r.dropped_yaw_deg) > 1 ? ` · the ${r.dropped_yaw_deg.toFixed(0)}° turn about the head's up axis is left to the heading anchors` : ''));
    } else {
      applyHeadResponse(r);
      headMsg(`rat ${rat.id} frame ${r0.frame} orientation saved · ` + fitText(rat));
    }
  } catch (err) { headMsg(String(err), true); }
}

async function reloadAfterRerun(frame) {
  playing = false; document.getElementById('playBtn').textContent = 'Play';
  for (;;) {
    await new Promise(r => setTimeout(r, 700));
    const p = await (await fetch(`${API}/api/progress`)).json();
    if (p.error) throw new Error(p.error);
    if (p.ready && !p.running) break;
  }
  const m = await (await fetch(`${API}/api/meta`)).json();
  const keepRotate = rotateMode;
  await enterPlayback(m, frame);
  if (keepRotate) setRotateMode(true);
}

document.getElementById('headRotateBtn').addEventListener('click', () => setRotateMode(!rotateMode));
document.getElementById('headRotateScope').addEventListener('change', () => { if (rotateMode) setRotateMode(true); });

// ---------------------------------------------------------------------------
// export: the page composites each frame exactly as shown (top view + markers
// + skull insets) and streams it to the server, which pipes it into ffmpeg
// ---------------------------------------------------------------------------
let exporting = false, exportStop = false;

function exportMsg(text, err = false) {
  const el = document.getElementById('exportMsg');
  el.textContent = text; el.classList.toggle('err', err);
}

function buildExportKinds() {
  const sel = document.getElementById('expKind');
  const keep = sel.value;
  sel.innerHTML = '<option value="video">top view + skull insets</option>'
    + (rats.length > 1 ? '<option value="skull">skulls only, side by side (rat 1 | rat 2)</option>' : '')
    + rats.map(r => `<option value="skull_rat${r.id}">skull only: rat ${r.id} (${RAT_NAME[r.id]}, full frame)</option>`).join('');
  if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
}

function exportRequest() {
  return {
    frame_from: parseInt(document.getElementById('expFrom').value, 10),
    frame_to: parseInt(document.getElementById('expTo').value, 10),
    out_dir: document.getElementById('expDir').value.trim(),
    kind: document.getElementById('expKind').value,
    trajectory: document.getElementById('expTraj').checked,
  };
}

function loadImage(url, tries = 3) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    // a long export occasionally loses one request; retry before giving up
    im.onerror = () => (tries > 1
      ? setTimeout(() => loadImage(url, tries - 1).then(res, rej), 300)
      : rej(new Error(`could not load ${url}`)));
    im.src = url;
  });
}

async function postJSON(path, body) {
  const r = await (await fetch(`${API}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
  if (!r.ok) throw new Error(r.error || `${path} failed`);
  return r;
}

function sizeInset(rat, w, h, pixelRatio) {
  const ins = rat.inset;
  ins.renderer.setPixelRatio(pixelRatio);
  ins.renderer.setSize(w, h, false);
  ins.camera.aspect = w / h; ins.camera.updateProjectionMatrix();
}

async function exportVideo() {
  if (exporting) return;
  if (playing) togglePlay();
  setClickMode(false);
  if (rotateMode) setRotateMode(false);
  const withInset = document.getElementById('expInset').checked;
  const withMarker = document.getElementById('expMarker').checked;
  const withPath = document.getElementById('expPath').checked;
  let job;
  try { job = await postJSON('/api/export/start', exportRequest()); }
  catch (e) { exportMsg(e.message, true); return; }
  exporting = true; exportStop = false;
  ['expVideoBtn', 'expTrajBtn'].forEach(id => { document.getElementById(id).disabled = true; });
  document.getElementById('expCancelBtn').style.display = '';

  const skull = job.kind !== 'video';
  // skull videos: every rat side by side ("skull") or one rat full frame
  const skullRats = job.kind === 'skull' ? rats : rats.filter(r => job.kind === `skull_rat${r.id}`);
  const [W, H] = meta.out_size;
  const cv = document.createElement('canvas'); cv.width = skull ? W * skullRats.length : W; cv.height = H;
  if (skull) skullRats.forEach(r => sizeInset(r, W, H, 1));
  const ctx = cv.getContext('2d');
  const insetPx = Math.round(W * 230 / 800), pad = Math.round(W * 10 / 800);
  const f0 = job.frame_from, f1 = job.frame_to, total = f1 - f0 + 1;
  const t0 = performance.now();
  let next = skull ? null : loadImage(frameUrl(f0)), sent = 0, error = null;
  try {
    for (let f = f0; f <= f1 && !exportStop; f++) {
      if (skull) {
        ctx.fillStyle = '#15140f'; ctx.fillRect(0, 0, cv.width, H);
        skullRats.forEach((r, k) => {
          if (updateInset(r, f)) ctx.drawImage(r.inset.renderer.domElement, k * W, 0, W, H);   // renders; copy it in the same task
          drawLegend(ctx, r, k * W + pad, pad + Math.round(H * 34 / 800), W / 400);   // top-left, clear of the frame label
        });
      } else {
        const im = await next;
        if (f < f1) next = loadImage(frameUrl(f + 1));          // decode the next frame meanwhile
        ctx.drawImage(im, 0, 0, W, H);
        if (withMarker || withPath) drawOverlay(ctx, f, { markers: withMarker, paths: withPath });
        if (withInset) {
          rats.forEach((r, k) => {                              // stacked down the right edge, rat 1 on top
            const y = pad + k * (insetPx + pad);
            ctx.fillStyle = 'rgba(21,20,15,0.82)'; ctx.fillRect(W - insetPx - pad, y, insetPx, insetPx);
            if (updateInset(r, f)) ctx.drawImage(r.inset.renderer.domElement, W - insetPx - pad, y, insetPx, insetPx);
            drawLegend(ctx, r, W - insetPx - pad + Math.round(8 * insetPx / 230), y + insetPx - Math.round(6 * insetPx / 230), insetPx / 230);
          });
        }
      }
      const states = rats.map(r => { const i = r.byFrame.get(f); return i != null ? `rat ${r.id} ${r.series.state[i]}` : null; }).filter(Boolean);
      const label = `frame ${f}  ${(f / meta.fps).toFixed(2)} s` + (states.length ? '  ' + states.join('  ') : '');
      ctx.font = `${Math.round(W * 13 / 800)}px ui-monospace, Menlo, monospace`;
      ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(pad, H - pad - 22, ctx.measureText(label).width + 12, 22);
      ctx.fillStyle = '#f1ede1';
      ctx.fillText(label, pad + 6, H - pad - 6);
      // toDataURL, not toBlob: the same JPEG, but toBlob took ~0.9 s per frame here
      const bin = atob(cv.toDataURL('image/jpeg', 0.92).split(',')[1]);
      const blob = new Uint8Array(bin.length);
      for (let k = 0; k < bin.length; k++) blob[k] = bin.charCodeAt(k);
      const r = await fetch(`${API}/api/export/frame?job=${job.job}`, { method: 'POST', body: blob });
      if (!r.ok) throw new Error((await r.json()).error || 'frame upload failed');
      sent++;
      if (sent % 10 === 0 || f === f1) {
        const el = (performance.now() - t0) / 1000, rate = sent / el;
        exportMsg(`${sent}/${total} frames · ${rate.toFixed(1)} fps · ${Math.round((total - sent) / rate)} s left`);
        document.getElementById('frameSlider').value = String(f);
      }
    }
  } catch (e) { error = e; }
  try {
    const r = await postJSON('/api/export/finish', { job: job.job });
    exportMsg((error ? `stopped by an error (${error.message}); ` : exportStop ? 'stopped; ' : '')
      + `wrote ${r.frames} frames to ${r.video}` + (r.trajectory.length ? ` + ${r.trajectory.map(p => p.split('/').pop()).join(', ')}` : ''), !!error);
  } catch (e) { exportMsg(e.message, true); }
  exporting = false;
  if (skull) {
    skullRats.forEach(r => sizeInset(r, r.inset.wrap.clientWidth || 230, r.inset.wrap.clientHeight || 230,
      Math.min(window.devicePixelRatio || 1, 2)));
  }
  ['expVideoBtn', 'expTrajBtn'].forEach(id => { document.getElementById(id).disabled = false; });
  document.getElementById('expCancelBtn').style.display = 'none';
  setFrame(currentFrame);
}

async function exportTrajectory() {
  try {
    const r = await postJSON('/api/export/trajectory', exportRequest());
    exportMsg(`wrote ${r.trajectory.slice(0, -1).join(', ')} (one CSV per rat, + .json with units and settings)`);
  } catch (e) { exportMsg(e.message, true); }
}

document.getElementById('expVideoBtn').addEventListener('click', exportVideo);
document.getElementById('expTrajBtn').addEventListener('click', exportTrajectory);
document.getElementById('expCancelBtn').addEventListener('click', () => { exportStop = true; });
document.getElementById('expRangeBtn').addEventListener('click', () => {
  document.getElementById('expFrom').value = String(currentFrame);
  document.getElementById('expTo').value = String(meta.frame_max);
});

function applyAlignment() {
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(alignDeg[0]),
    THREE.MathUtils.degToRad(alignDeg[1]),
    THREE.MathUtils.degToRad(alignDeg[2]),
    'XYZ',
  );
  for (const r of rats) {
    r.inset.alignGroup.quaternion.setFromEuler(euler);
    r.inset.renderer.render(r.inset.scene, r.inset.camera);
  }
}

function onAlignInput() {
  alignDeg = ['alignX', 'alignY', 'alignZ'].map(id => parseFloat(document.getElementById(id).value) || 0);
  applyAlignment();
  clearTimeout(saveAlignTimer);
  saveAlignTimer = setTimeout(() => {
    fetch(`${API}/api/align`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cap_align_deg: alignDeg }) }).catch(() => {});
  }, 500);
}
['alignX', 'alignY', 'alignZ'].forEach(id => document.getElementById(id).addEventListener('input', onAlignInput));
document.getElementById('showAxes').addEventListener('change', () => { if (rats.length) redraw(); });
document.getElementById('showTrail').addEventListener('change', (e) => {
  rats.forEach(r => { r.inset.trailLine.visible = e.target.checked; r.inset.renderer.render(r.inset.scene, r.inset.camera); });
});
document.getElementById('showMasks').addEventListener('change', () => { if (meta) setFrame(currentFrame); });
['showPath', 'pathSeconds'].forEach(id => document.getElementById(id).addEventListener('input', () => { if (rats.length && shownFrame != null) updateOverlay(shownFrame); }));

document.getElementById('frameSlider').addEventListener('input', (e) => {
  playing = false; document.getElementById('playBtn').textContent = 'Play';
  setFrame(parseInt(e.target.value, 10));
});
document.getElementById('playBtn').addEventListener('click', togglePlay);
document.querySelectorAll('.speedBtn').forEach(b => b.addEventListener('click', () => {
  playSpeed = parseFloat(b.dataset.speed);
  document.querySelectorAll('.speedBtn').forEach(x => x.classList.toggle('active', x === b));
}));

animate();
window.__enterPlayback = enterPlayback;
