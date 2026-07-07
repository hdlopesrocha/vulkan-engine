#include "Simplifier.hpp"
#include "NodeOperationResult.hpp"
#include "Octree.hpp"
#include "../sdf/SDF.hpp"


Simplifier::Simplifier(float angle, float distance, bool texturing) {
	this->angle = angle;
	this->distance = distance;
	this->texturing = texturing;
}	


SimplificationResult Simplifier::simplify(
        const BoundingCube cube,
        const float * sdf,
        NodeOperationResult * children,
        const BoundingCube& chunkCube)
{
    SimplificationResult res(false, DISCARD_BRUSH_INDEX);
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
                if(!child->isSimplified) {
                    return res;   // unsimplified child: cannot collapse parent
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

    // --- 3. Flatness detection from child surface normals ---
    // Compute the average surface normal across all surface children.  If every
    // child's normal lies within `angle` (cosine of max allowed deviation) of
    // the average, the patch is considered flat and a generous SDF tolerance
    // (`distance * cube_size`) is applied — allowing aggressive simplification
    // of extended flat or gently sloped terrain.
    // Curved patches (high normal variance) get half the tolerance so sharp
    // ridges and concave features are not smoothed away.
    glm::vec3 avgNormal(0.0f);
    int surfaceCount = 0;
    for(uint i = 0; i < 8; ++i) {
        NodeOperationResult * child = &children[i];
        if(child && child->resultType == SpaceType::Surface && child->node) {
            avgNormal += child->node->vertex.normal;
            ++surfaceCount;
        }
    }

    // Default to the full `distance` tolerance (flat-surface path).
    float toleranceMultiplier = distance;
    if(surfaceCount >= 2) {
        const float len = glm::length(avgNormal);
        if(len > 1e-6f) {
            avgNormal /= len;
        }

        // Find the child whose normal deviates most from the average.
        float minDot = 1.0f;
        for(uint i = 0; i < 8; ++i) {
            NodeOperationResult * child = &children[i];
            if(child && child->resultType == SpaceType::Surface && child->node) {
                const glm::vec3 n = child->node->vertex.normal;
                const float nLen = glm::length(n);
                if(nLen > 1e-6f) {
                    minDot = glm::min(minDot, glm::dot(avgNormal, n / nLen));
                }
            }
        }

        // `angle` is the cosine threshold (e.g. 0.95 ≈ 18°).  If any child's
        // normal falls outside this cone the surface is curved: tighten the
        // tolerance to half to preserve geometric detail.
        if(minDot < angle) {
            toleranceMultiplier = distance * 0.5f;
        }
    }

    // --- 4. SDF re-interpolation error check with adaptive tolerance ---
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
    res.brushIndex = brushIndex;
    return res;
}
