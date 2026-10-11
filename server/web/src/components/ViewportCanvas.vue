<template>
  <div class="viewport">
    <canvas id="gl"></canvas>
    <div class="help">
      <b>Camera (matches Vulkan):</b> W/S forward/back · A/D left/right · Q/E down/up ·
      H/F yaw−/+ · G/T pitch−/+ · R/Y roll−/+ · drag = yaw/pitch · click box = select + wireframe
      · <i>render all</i> draws every pushed chunk geometry (snapshot + live batches, no requests)
      (frustum-culled boxes, depth-tested wireframe · camera auto-saves, Reset camera restores start view)
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue';
import { cam, camFlags, loadCamera, saveCamera, updateCamera, NAV_KEYS, camForward, camUp, camRight } from '../camera.js';
import { perspective, lookAt, matMul, invertMat, extractPlanes, V3, qmul, qnorm, angleAxis } from '../math3d.js';
import { initGL, renderFrame, pick } from '../gl.js';
import { rendererOf, fps } from '../store.js';

let hasGL = false;
let raf = 0;
let down = null, moved = 0;

function onKeyDown(e) {
  const t = (e.target && e.target.tagName) || '';
  if (t === 'INPUT' || t === 'TEXTAREA') return;
  cam.keys.add(e.code);
}
function onKeyUp(e) { cam.keys.delete(e.code); }
function onBlur() { cam.keys.clear(); } // never stick keys when focus is lost
function onMouseDown(e) { down = [e.clientX, e.clientY]; moved = 0; }
function onMouseMove(e) {
  if (!down) return;
  const dx = e.movementX || 0, dy = e.movementY || 0;
  moved += Math.abs(dx) + Math.abs(dy);
  const s = 0.15;
  cam.quat = qnorm(qmul(angleAxis([0, 1, 0], -dx * s * Math.PI / 180), cam.quat));
  cam.quat = qnorm(qmul(angleAxis(camRight(), -dy * s * Math.PI / 180), cam.quat));
  if (dx || dy) camFlags.dirty = true;
}
function onMouseUp(e) {
  if (!down) return;
  const wasClick = moved < 5;
  down = null;
  if (!wasClick) return;
  const cv = document.getElementById('gl');
  const r = cv.getBoundingClientRect();
  const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
  const ny = -(((e.clientY - r.top) / r.height) * 2 - 1);
  const aspect = cv.width / cv.height;
  const fwd = camForward(), up = camUp();
  const vp = matMul(perspective(60, aspect, 0.1, 8092), lookAt(cam.pos, V3.add(cam.pos, fwd), up));
  const inv = invertMat(vp);
  if (!inv) return;
  function unproj(x, y, z) {
    const v = [x, y, z, 1];
    const o = [0, 0, 0, 0];
    for (let r2 = 0; r2 < 4; r2++) o[r2] = inv[r2] * v[0] + inv[4 + r2] * v[1] + inv[8 + r2] * v[2] + inv[12 + r2] * v[3];
    return [o[0] / o[3], o[1] / o[3], o[2] / o[3]];
  }
  pick(nx, ny, unproj);
}
function onResize() { /* canvas size syncs lazily in the frame loop */ }
function onPageHide() { saveCamera(); }

onMounted(() => {
  const cv = document.getElementById('gl');
  hasGL = initGL(cv);
  loadCamera();
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);
  cv.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mousemove', onMouseMove);
  window.addEventListener('mouseup', onMouseUp);
  window.addEventListener('resize', onResize);
  window.addEventListener('pagehide', onPageHide);
  let last = performance.now();
  let frames = 0, t0 = last;
  function loop(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    updateCamera(dt);
    const r = cv.parentElement.getBoundingClientRect();
    const w = Math.max(2, r.width | 0), h = Math.max(2, (r.height) | 0);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    if (hasGL) {
      const aspect = cv.width / cv.height;
      const fwd = camForward(), up = camUp();
      const proj = perspective(60, aspect, 0.1, 8092);
      const view = lookAt(cam.pos, V3.add(cam.pos, fwd), up);
      const vp = matMul(proj, view);
      const planes = extractPlanes(vp);
      const poseKey = cam.pos.map(x => x.toFixed(1)).join(',') + '|' + cam.quat.map(x => x.toFixed(4)).join(',');
      renderFrame(vp, planes, aspect, rendererOf, poseKey);
    }
    frames++;
    if (camFlags.dirty && now - camFlags.lastSave > 1000) saveCamera();
    if (now - t0 > 500) { fps.value = frames * 1000 / (now - t0); frames = 0; t0 = now; }
    raf = requestAnimationFrame(loop);
  }
  raf = requestAnimationFrame(loop);
});

onUnmounted(() => {
  cancelAnimationFrame(raf);
  window.removeEventListener('keydown', onKeyDown);
  window.removeEventListener('keyup', onKeyUp);
  window.removeEventListener('blur', onBlur);
  window.removeEventListener('mousemove', onMouseMove);
  window.removeEventListener('mouseup', onMouseUp);
  window.removeEventListener('resize', onResize);
  window.removeEventListener('pagehide', onPageHide);
});
</script>
