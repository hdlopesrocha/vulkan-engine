#include "OctreeVisibilityChecker.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "../math/Frustum.hpp"

OctreeVisibilityChecker::OctreeVisibilityChecker() {
	visibleNodes.reserve(1024);
}

void OctreeVisibilityChecker::update(glm::mat4 m) {
	frustum = Frustum(m);
	viewDir = glm::normalize(-glm::vec3(m[2]));
}

bool OctreeVisibilityChecker::iterate(const Octree &tree, OctreeNodeData &params) {
	ContainmentType containmentType = frustum.test(params.cube);
	if(containmentType == ContainmentType::Disjoint) {
		return false;
	}

	if(params.node->isChunk() || params.node->isSimplified()) {
		if(params.node->getType() == SpaceType::Surface) {
			std::lock_guard<std::mutex> lock(mutex);
			visibleNodes.push_back(params);
		} 
		return false;
	}
	return true;
	
}

void OctreeVisibilityChecker::getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]){
    std::pair<float, uint> internalSortingVector[8]={};

    for(uint i = 0; i < 8; ++i){
        glm::vec3 center = params.cube.getChildCenter(i);
	    float key = -glm::dot(center, viewDir);
        internalSortingVector[i] = std::pair<float, uint>(key, i);
    }

    std::sort(std::begin(internalSortingVector), std::end(internalSortingVector),
        [&](const std::pair<float, uint>& a, const std::pair<float, uint>& b) {
            return a.first < b.first;
    });

    for(uint i = 0 ; i < 8 ; ++i) {
        order[i] = internalSortingVector[i].second;
    }
}