#include "Simplifier.hpp"
#include "NodeOperationResult.hpp"
#include "Octree.hpp"
#include "../sdf/SDF.hpp"


Simplifier::Simplifier(float angle, float distance, bool texturing) {
	this->angle = angle;
	this->distance = distance;
	this->texturing = texturing;
}	


SimplificationResult Simplifier::simplify(const BoundingCube cube, const float * sdf, NodeOperationResult * children){
	SimplificationResult res(false, DISCARD_BRUSH_INDEX);
	int brushIndex = DISCARD_BRUSH_INDEX;

	//uint mask = 0xff;
	if(texturing) {
		for(uint i=0; i < 8 ; ++i) {
			NodeOperationResult * child = &children[i];
			if(child && child->resultType == SpaceType::Surface) {
				if(brushIndex == DISCARD_BRUSH_INDEX) {
					brushIndex = child->brushIndex;
				} else if(child->brushIndex != DISCARD_BRUSH_INDEX 
					&& child->brushIndex != brushIndex) {
					return res;    
				}
						
				if(!child->isSimplified) {
					return res;
				}
			}
		}
	}
	// for leaf nodes shouldn't loop
	for(uint i=0; i < 8 ; ++i) {
		NodeOperationResult * child = &children[i];
		if(child && child->resultType == SpaceType::Surface) {
			BoundingCube childCube = cube.getChild(i);

			for(int j = 0 ; j < 8 ; ++j) {
				glm::vec3 corner = childCube.getCorner(j);
				float d = SDF::interpolate(sdf, corner , cube);
				float dif = glm::abs(d - child->resultSDF[j]);

				if(dif > cube.getLengthX() * 0.1f) {
					return res;
				}
			}			
		}
	}
	res.isSimplified = true;
	res.brushIndex = brushIndex;
	return res;
}
