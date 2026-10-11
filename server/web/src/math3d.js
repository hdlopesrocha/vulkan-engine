// ---- math (quaternion camera matching math/Camera.cpp + KeyboardPublisher) ----
function angleAxis(axis, rad) {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const s = Math.sin(rad / 2) / n;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(rad / 2)];
}
function qmul(a, b) {
  return [
    a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
    a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
    a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
    a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]];
}
function qnorm(q) { const l = Math.hypot(q[0],q[1],q[2],q[3]) || 1; return [q[0]/l,q[1]/l,q[2]/l,q[3]/l]; }
function qrot(q, v) {
  const [x,y,z,w] = q;
  const uv = [y*v[2]-z*v[1], z*v[0]-x*v[2], x*v[1]-y*v[0]];
  const uuv = [y*uv[2]-z*uv[1], z*uv[0]-x*uv[2], x*uv[1]-y*uv[0]];
  return [v[0]+2*(w*uv[0]+uuv[0]), v[1]+2*(w*uv[1]+uuv[1]), v[2]+2*(w*uv[2]+uuv[2])];
}
function eulerToQuat(yawDeg, pitchDeg, rollDeg) {
  const qy = angleAxis([0,1,0], yawDeg*Math.PI/180);
  const qp = angleAxis([1,0,0], pitchDeg*Math.PI/180);
  const qr = angleAxis([0,0,1], rollDeg*Math.PI/180);
  return qnorm(qmul(qy, qmul(qp, qr)));
}
const V3 = {
  add:(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],
  sub:(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],
  scale:(a,s)=>[a[0]*s,a[1]*s,a[2]*s],
  cross:(a,b)=>[a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]],
  norm:(a)=>{const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];},
  dot:(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
};
function lookAt(eye, center, up) {
  let f = V3.norm(V3.sub(center, eye)), s = V3.norm(V3.cross(f, up)), u = V3.cross(s, f);
  return new Float32Array([s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0,
    -V3.dot(s,eye),-V3.dot(u,eye),V3.dot(f,eye),1]);
}
function perspective(fovDeg, aspect, near, far) {
  const f = 1/Math.tan(fovDeg*Math.PI/360), nf = 1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function matMul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  }
  return o;
}
function invertMat(m) {
  const o = new Float32Array(16);
  const a00=m[0],a01=m[1],a02=m[2],a03=m[3],a10=m[4],a11=m[5],a12=m[6],a13=m[7],
        a20=m[8],a21=m[9],a22=m[10],a23=m[11],a30=m[12],a31=m[13],a32=m[14],a33=m[15];
  const b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10,b03=a01*a12-a02*a11,
        b04=a01*a13-a03*a11,b05=a02*a13-a03*a12,b06=a20*a31-a21*a30,b07=a20*a32-a22*a30,
        b08=a20*a33-a23*a30,b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
  let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
  if (!det) return null;
  det=1/det;
  o[0]=(a11*b11-a12*b10+a13*b09)*det; o[1]=(a02*b10-a01*b11-a03*b09)*det;
  o[2]=(a31*b05-a32*b04+a33*b03)*det; o[3]=(a22*b04-a21*b05-a23*b03)*det;
  o[4]=(a12*b08-a10*b11-a13*b07)*det; o[5]=(a00*b11-a02*b08+a03*b07)*det;
  o[6]=(a32*b02-a30*b05-a33*b01)*det; o[7]=(a20*b05-a22*b02+a23*b01)*det;
  o[8]=(a10*b10-a11*b08+a13*b06)*det; o[9]=(a01*b08-a00*b10-a03*b06)*det;
  o[10]=(a30*b04-a31*b02+a33*b00)*det; o[11]=(a21*b02-a20*b04-a23*b00)*det;
  o[12]=(a11*b07-a10*b09-a12*b06)*det; o[13]=(a00*b09-a01*b07+a02*b06)*det;
  o[14]=(a31*b01-a30*b03-a32*b00)*det; o[15]=(a20*b03-a21*b01+a22*b00)*det;
  return o;
}
// ---- frustum culling (Gribb/Hartmann planes from column-major VP) ----
function extractPlanes(vp) {
  function plane(a, b, s) {
    const p = [vp[a] + s * vp[b], vp[4 + a] + s * vp[4 + b], vp[8 + a] + s * vp[8 + b], vp[12 + a] + s * vp[12 + b]];
    const l = Math.hypot(p[0], p[1], p[2]) || 1;
    return [p[0] / l, p[1] / l, p[2] / l, p[3] / l];
  }
  // left, right, bottom, top, near, far (inward-facing)
  return [plane(3, 0, 1), plane(3, 0, -1), plane(3, 1, 1), plane(3, 1, -1), plane(3, 2, 1), plane(3, 2, -1)];
}
function aabbVisible(planes, mn, mx) {
  for (const pl of planes) {
    const px = pl[0] >= 0 ? mx[0] : mn[0];
    const py = pl[1] >= 0 ? mx[1] : mn[1];
    const pz = pl[2] >= 0 ? mx[2] : mn[2];
    if (pl[0] * px + pl[1] * py + pl[2] * pz + pl[3] < 0) return false;
  }
  return true;
}
export { angleAxis, qmul, qnorm, qrot, eulerToQuat, V3, lookAt, perspective,
  matMul, invertMat, extractPlanes, aabbVisible };
