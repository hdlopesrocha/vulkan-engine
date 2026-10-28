#include "Simplifier.hpp"
#include "NodeOperationResult.hpp"
#include "Octree.hpp"
#include "../sdf/SDF.hpp"
#include <cmath>


Simplifier::Simplifier(float tm) {
	this->toleranceMultiplier = tm;
}	


SimplificationResult Simplifier::simplify(
        const BoundingCube &cube,
        const float * sdf,
        NodeOperationResult * children,
        const BoundingCube& chunkCube) const 
{
    SimplificationResult res(0u);
    int brushIndex = DISCARD_BRUSH_INDEX;

    // --- 1. Brush consistency ---
    // Every surface child must share the same brush index: a collapse that
    // mixes materials would lose the material boundary, so full detail is
    // preserved there.
    for(uint i = 0; i < 8; ++i) {
        NodeOperationResult * child = &children[i];
        if(child && child->resultType == SpaceType::Surface) {
            if(brushIndex == DISCARD_BRUSH_INDEX) {
                brushIndex = child->brushIndex;
            } else if(child->brushIndex != brushIndex) {
                return res;   // material boundary: preserve full detail
            }
        }
    }


    // --- 2. Chunk-border guard ---
    // Direct children of a chunk root (side = chunkSize / 2) must never be
    // simplified.  Chunk roots are the GPU-upload boundaries; collapsing their
    // immediate children would produce LOD seams at chunk edges because two
    // adjacent chunks may reach different simplification decisions at that level.
    {
        const float directChildLen = chunkCube.getLengthX() * 0.5f;
        if(glm::abs(cube.getLengthX() - directChildLen) < cube.getLengthX() * 1e-3f) {
            return res;
        }
    }
    // --- 3. Coarse-surface crossing guard ---
    // A collapsed cell is meshed as one Surface-Nets cell, and its vertex is
    // solved from the coarse corner samples alone. If those eight samples are
    // all on the same side, there is no crossing at the coarse level (the
    // solver parks the vertex at the cell centre), while the children can still
    // contain the real surface: a thin wall/rim, or any feature that fits
    // between the coarse corners. Keeping such a cell terminal turns its centre
    // into a surface vertex and the mesh bridges from there to the real surface
    // with long spike triangles. Only collapse cells whose own coarse samples
    // actually straddle the surface; otherwise let the children stay meshed.
    //
    // SDF-only: no corner is converted to a Cartesian position.
    if(SDF::eval(sdf) != SpaceType::Surface) {
        return res;
    }

  
    // --- 4. SDF re-interpolation error check with adaptive tolerance ---
    // Re-interpolate the parent SDF at every corner of the child lattice and
    // compare against the child's actual stored values.  The union of the eight
    // children's corners is the full 3x3x3 half-cell lattice, so this samples
    // the whole cell, not just its surface children.  Solid/empty children must
    // be checked too: a cliff, crease or thin feature can cross their corners
    // (the coarse interpolation is then wrong there), and skipping them lets
    // the simplifier collapse the cell and place a QEF vertex on the wrong side
    // of the feature, which the mesh bridges with long spike triangles.
    const float threshold = cube.getLengthX() * toleranceMultiplier;
    for(uint i = 0; i < 8; ++i) {
        NodeOperationResult * child = &children[i];
        if(child == nullptr) {
            continue;
        }
        BoundingCube childCube = cube.getChild(i);
        for(int j = 0; j < 8; ++j) {
            glm::vec3 corner = childCube.getCorner(j);
            float d = SDF::interpolate(sdf, corner, cube);
            float stored = child->resultSDF[j];
            // Sentinels (unknown regions) carry no error information.
            if(!std::isfinite(d) || !std::isfinite(stored)) {
                continue;
            }
            float dif = glm::abs(d - stored);
            if(dif > threshold) {
                return res;
            }
        }
    }

    res.isSimplified = true;
    return res;
}
