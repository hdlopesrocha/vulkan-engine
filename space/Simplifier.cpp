#include "Simplifier.hpp"
#include "NodeOperationResult.hpp"
#include "Octree.hpp"
#include "../sdf/SDF.hpp"


Simplifier::Simplifier(float angle, float distance, bool texturing) {
	this->angle = angle;
	this->distance = distance;
	this->texturing = texturing;
}	


std::pair<bool,int> Simplifier::simplify(const BoundingCube chunkCube, const BoundingCube cube, const float * sdf, NodeOperationResult * children){	
	int brushIndex = DISCARD_BRUSH_INDEX;
	if(!chunkCube.contains(BoundingCube(cube.getMin() - cube.getLength(), cube.getLengthX()))) {
		return {false, brushIndex};
	}

	//uint mask = 0xff;
	int nodeCount=0;
	for(uint i=0; i < 8 ; ++i) {
		NodeOperationResult * child = &children[i];
		if(child && child->resultType == SpaceType::Surface) {
			if(!child->isSimplified) {
				return {false, brushIndex};	
			}
			if(texturing && brushIndex != DISCARD_BRUSH_INDEX && child->brushIndex != brushIndex) {
				return {false, brushIndex};	
			}
			brushIndex = child->brushIndex;
			break;
		}
	}


	// for leaf nodes shouldn't loop
	for(uint i=0; i < 8 ; ++i) {
		NodeOperationResult * child = &children[i];
		if(child->resultType == SpaceType::Surface) {
			BoundingCube childCube = cube.getChild(i);

			for(int j = 0 ; j < 8 ; ++j) {
				glm::vec3 corner = childCube.getCorner(j);
				float d = SDF::interpolate(sdf, corner , cube);
				float dif = glm::abs(d - child->resultSDF[j]);

				if(dif > cube.getLengthX() * 0.05) {
					return {false, brushIndex};
				}
			}

			++nodeCount;
			
		}
	}
	
	if(nodeCount > 0) {	
		return {true, brushIndex};
	}
	
	return {false, brushIndex};
}
