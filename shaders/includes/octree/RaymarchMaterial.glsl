#ifndef OCTREE_RAYMARCH_MATERIAL_GLSL
#define OCTREE_RAYMARCH_MATERIAL_GLSL

// Node-payload helpers shared by the raymarch fragment shader
// (shaders/renderer/octree/OctreeRaymarch.frag) and the once-per-revision
// payload derive pass (shaders/computation/octree/OctreeDerive.comp), so the
// cached payload and the march's own fallback cannot disagree.
//
// Includers must provide SdfCell.glsl (sdfIsFinite, sdfCellNormal) and
// LandBrush.glsl (LAND_BRUSH_INDEX, LAND_UNDERGROUND, LAND_ROCK, NORMAL_BRUSH,
// landMaterialIndex) before this header.

// Shading "not a surface" sentinel (same value as RAYMARCH_DISCARD_BRUSH in
// OctreeRaymarch.frag).
const int RM_MATERIAL_DISCARD = -1;

// Resolve a node's shading material. `sdf`/`mn`/`sz` describe the node cube so
// a caller that already solved the QEF can pass its position and normal
// (`qefPos`, `n`) instead of recomputing them. Same rule as the original
// rmResolveMaterialAt(): only sentinel brushes need the classifier.
int rmResolveMaterialAtCorners(int raw, float sdf[8], vec3 mn, float sz,
                               vec3 qefPos, vec3 n) {
    if (raw != LAND_BRUSH_INDEX && raw != NORMAL_BRUSH) return raw;
    if (raw == LAND_BRUSH_INDEX) {
        // Mirror writeVertexOnce(): LAND_UNDERGROUND is a shading fallback,
        // not a topology discard.
        int land = landMaterialIndex(qefPos, n);
        return (land == LAND_UNDERGROUND) ? LAND_ROCK : land;
    }
    // NormalBrush keeps only +Y-facing cells, like the old CPU painter.
    return (dot(normalize(n), vec3(0.0, 1.0, 0.0)) > 0.0) ? 0 : RM_MATERIAL_DISCARD;
}

// Conservative Lipschitz bound of the node's trilinear reconstruction
// (docs/raymarch-octree.tex section 6.2). Per-axis maximum edge difference of
// the eight corner samples over the cell edge length; a step of |f| / L cannot
// cross the isosurface. Any non-finite corner returns 2.0 (the classical safe
// factor), so sentinel cells keep the fixed 0.5 step.
float rmLipBound(float sdf[8], float sz) {
    for (int i = 0; i < 8; ++i) {
        if (!sdfIsFinite(sdf[i])) return 2.0;
    }
    // Engine corner order: bit2 = x, bit1 = y, bit0 = z (SdfCell.glsl).
    float dx = max(max(abs(sdf[4] - sdf[0]), abs(sdf[5] - sdf[1])),
                   max(abs(sdf[6] - sdf[2]), abs(sdf[7] - sdf[3])));
    float dy = max(max(abs(sdf[2] - sdf[0]), abs(sdf[3] - sdf[1])),
                   max(abs(sdf[6] - sdf[4]), abs(sdf[7] - sdf[5])));
    float dz = max(max(abs(sdf[1] - sdf[0]), abs(sdf[3] - sdf[2])),
                   max(abs(sdf[5] - sdf[4]), abs(sdf[7] - sdf[6])));
    return sqrt(dx * dx + dy * dy + dz * dz) / max(sz, 1e-8);
}

#endif // OCTREE_RAYMARCH_MATERIAL_GLSL
