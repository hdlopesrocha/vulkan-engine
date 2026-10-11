import { angleAxis, qmul, qnorm, qrot, eulerToQuat, V3 } from './math3d.js';

// Camera mirrors Vulkan: pos(-512,2048,0), yaw -90 (math/Camera + MyApp setup).
// Pose persists in localStorage across reloads (validated on load).
// Move speed is fixed at the Vulkan default (Camera::speed = 512).
export const CAM_STORE_KEY = 'chunkViewer.camera.v1';
export const CAM_DEFAULT = { pos: [-512, 2048, 0], yaw: -90 };
export const CAM_SPEED = 512;
export const cam = {
  pos: [...CAM_DEFAULT.pos],
  quat: eulerToQuat(CAM_DEFAULT.yaw, 0, 0),
  timer: 0,
  keys: new Set(),
};
export const NAV_KEYS = ['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE', 'KeyH', 'KeyF', 'KeyG', 'KeyT', 'KeyR', 'KeyY'];
export const camFlags = { dirty: false, lastSave: 0 };

export function saveCamera() {
  try {
    localStorage.setItem(CAM_STORE_KEY, JSON.stringify({ pos: cam.pos, quat: cam.quat }));
  } catch (e) { /* private mode etc. — pose just won't persist */ }
  camFlags.dirty = false;
  camFlags.lastSave = performance.now();
}

export function loadCamera() {
  let s = null;
  try { s = localStorage.getItem(CAM_STORE_KEY); } catch (e) { return; }
  if (!s) return;
  try {
    const o = JSON.parse(s);
    const ok3 = a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
    const ok4 = a => Array.isArray(a) && a.length === 4 && a.every(Number.isFinite);
    if (ok3(o.pos)) cam.pos = [...o.pos];
    if (ok4(o.quat)) cam.quat = qnorm([...o.quat]);
  } catch (e) { /* corrupt entry — keep defaults */ }
}

export function resetCamera() {
  cam.pos = [...CAM_DEFAULT.pos];
  cam.quat = eulerToQuat(CAM_DEFAULT.yaw, 0, 0);
  cam.timer = 0;
  saveCamera();
}

export function camForward() { return V3.norm(qrot(cam.quat, [0, 0, -1])); }
export function camUp() { return V3.norm(qrot(cam.quat, [0, 1, 0])); }
export function camRight() {
  const f = camForward(), u = camUp();
  return V3.norm(V3.cross(f, u));
}

export function updateCamera(dt) {
  const k = cam.keys;
  const fwd = camForward(), up = camUp(), right = camRight();
  const hasT = k.has('KeyW') || k.has('KeyS') || k.has('KeyA') || k.has('KeyD') || k.has('KeyQ') || k.has('KeyE');
  if (hasT) cam.timer += dt; else cam.timer = 0;
  const accel = (Math.exp(cam.timer) - 1) / (Math.exp(1) - 1);
  const vel = CAM_SPEED * dt * (hasT ? Math.max(accel, 1e-3) : 0);
  let mv = [0, 0, 0];
  if (k.has('KeyW')) mv = V3.add(mv, V3.scale(fwd, vel));
  if (k.has('KeyS')) mv = V3.add(mv, V3.scale(fwd, -vel));
  if (k.has('KeyD')) mv = V3.add(mv, V3.scale(right, vel));
  if (k.has('KeyA')) mv = V3.add(mv, V3.scale(right, -vel));
  if (k.has('KeyE')) mv = V3.add(mv, V3.scale(up, vel));
  if (k.has('KeyQ')) mv = V3.add(mv, V3.scale(up, -vel));
  cam.pos = V3.add(cam.pos, mv);
  const ang = 45 * dt; // deg/s, matches Camera::angularSpeedRad
  const D = Math.PI / 180;
  function yaw(d) { cam.quat = qnorm(qmul(angleAxis([0, 1, 0], d * D), cam.quat)); }
  function pitch(d) { cam.quat = qnorm(qmul(angleAxis(camRight(), d * D), cam.quat)); }
  function roll(d) { cam.quat = qnorm(qmul(angleAxis(camForward(), d * D), cam.quat)); }
  if (k.has('KeyH')) yaw(-ang);
  if (k.has('KeyF')) yaw(ang);
  if (k.has('KeyG')) pitch(-ang);
  if (k.has('KeyT')) pitch(ang);
  if (k.has('KeyR')) roll(-ang);
  if (k.has('KeyY')) roll(ang);
  if (NAV_KEYS.some(kk => k.has(kk))) camFlags.dirty = true;
}
