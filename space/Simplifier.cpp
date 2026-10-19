#include "Simplifier.hpp"
#include "NodeOperationResult.hpp"
#include "Octree.hpp"
#include "../sdf/SDF.hpp"


Simplifier::Simplifier(float angle_, float distance_, bool texturing_) {
	this->angle = angle_;
	this->distance = distance_;
	this->texturing = texturing_;
}	


SimplificationResult Simplifier::simplify(
        const BoundingCube &cube,
        const float * sdf,
        NodeOperationResult * children,
        const BoundingCube& chunkCube) const 
{
    SimplificationResult res(0u);
    int brushIndex = DISCARD_BRUSH_INDEX;

    // --- 1. Brush consistency and simplification-chain check (unchanged) ---
    // With texturing enabled all surface children must share the same brush index
    // and must themselves be simplified (ensures the LOD chain is consistent).
    if(texturing) {
        for(uint i = 0; i < 8; ++i) {
            NodeOperationResult * child = &children[i];
            if(child && child->resultType == SpaceType::Surface) {
                if(brushIndex == DISCARD_BRUSH_INDEX) {
                    brushIndex = child->brushIndex;
                } else if(child->brushIndex != DISCARD_BRUSH_INDEX
                           && child->brushIndex != brushIndex) {
                    return res;   // material boundary: preserve full detail
                }
                if(child->node && !child->node->isLeaf()) {
                    return res;   // unsimplified (non-collapsed) child: cannot collapse parent
                }
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
    // Default to the full `distance` tolerance (flat-surface path).
    float toleranceMultiplier = distance;
  
    // --- 3. SDF re-interpolation error check with adaptive tolerance ---
    // For each surface child re-interpolate the parent SDF at the child's 8
    // corners and compare against the child's actual stored values.  If the
    // error exceeds the adaptive threshold the coarser representation would
    // lose too much geometric detail at this level.
    const float threshold = cube.getLengthX() * toleranceMultiplier;
    for(uint i = 0; i < 8; ++i) {
        NodeOperationResult * child = &children[i];
        if(child && child->resultType == SpaceType::Surface) {
            BoundingCube childCube = cube.getChild(i);
            for(int j = 0; j < 8; ++j) {
                glm::vec3 corner = childCube.getCorner(j);
                float d = SDF::interpolate(sdf, corner, cube);
                float dif = glm::abs(d - child->resultSDF[j]);
                if(dif > threshold) {
                    return res;
                }
            }
        }
    }

    res.isSimplified = true;
    return res;
}
