// M5 (perf report 28): shared impostor placement core (center -> quad ->
// offset -> finalPos -> UV crop) for the color + depth vertex twins. Zero
// runtime delta, one spelling: the same float ops in the same order as the
// twin-local blocks this replaces (byte-comparable SPIR-V modulo names).
//
// Contract: the includer provides worldPos, billboardScale (push), hs,
// toCamera, layerIdx and the inCornerUV / inInstanceNormal attributes; this
// block defines the placement locals (surfN, center, right/up, offset,
// finalPos) and writes outTexCoord / outWorldPos / gl_Position. Prologues
// stay separate (different cull sources, density context, H8 trig in color
// only), as do the divergent outputs (depth writes outInstanceOffset after
// including; the color-only faceNorm copy is deleted per M5).
//
// Match the capture setup: the plant was captured at heightScale=1.0 with a
// fixed-size square framebuffer. The plant occupies only 34.6% of the image
// height (top/bottom 32.7% are clear). The quad and its centre scale
// uniformly with the runtime height scale (hs), exactly like the billboard
// (VegetationRenderer.vert scales its corners by billboardScale * hs), so the
// impostor is the same size as the plant it replaces. The UV crop maps the
// captured bbox (a fixed fraction of the image) to the whole quad, so the
// texture scales with the quad; it must NOT scale with hs itself.
// The quad stays glued to sloped ground by rising along the surface normal.
vec3 surfN = inInstanceNormal;
float surfNLen2 = dot(surfN, surfN);
surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
vec3 center = worldPos + surfN * (billboardScale * hs * 0.5);
vec3 worldUp = vec3(0.0, 1.0, 0.0);

vec3 right;
float sinElev = length(vec2(toCamera.x, toCamera.z));
if (sinElev > 0.001) {
    right = normalize(cross(worldUp, toCamera));
} else {
    right = vec3(1.0, 0.0, 0.0);
}
vec3 upDir = normalize(cross(toCamera, right));

// Quad size: the captured plant bbox in world units, scaled by hs.
// (The old code scaled the width by hs but left the height at
// billboardScale and scaled uFrac by hs, which cancelled out and rendered
// every impostor at the capture size - tall plants shrank and short ones
// were cropped at the hand-off.)
float quadHalfW = 0.75 * billboardScale * hs;
float quadHalfH = 0.5  * billboardScale * hs;
right = right * quadHalfW;
vec3 up = upDir * quadHalfH;

float u = inCornerUV.x;
float v = inCornerUV.y;
vec3 offset = (u - 0.5) * 2.0 * right + (0.5 - v) * 2.0 * up;
vec3 finalPos = center + offset;

// Crop UV to the plant's bounding box within the captured image.
// The plant occupies UV.V in [0.327, 0.673] (vertical) and
// UV.U in [0.5 ± 1.5/2.8867] (horizontal), independent of hs.
float uFrac = 1.5 / 2.886751346;      // 1.5 / (2 * 2.5 * tan(30°))
float vFrac = 1.0 / 2.886751346;      // billboardScale / (2 * 2.5 * billboardScale * tan(30°))
float vOff  = 0.5 - 0.5 / 2.886751346;
outTexCoord = vec3(0.5 + (inCornerUV.x - 0.5) * uFrac,
                   inCornerUV.y * vFrac + vOff,
                   float(layerIdx));
outWorldPos = finalPos;
gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
