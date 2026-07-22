#include "SolidSpaceChangeHandler.hpp"


SolidSpaceChangeHandler::SolidSpaceChangeHandler(
		const NodeDataCallback& onNodeUpdatedCallback_,
    	const NodeDataCallback& onNodeDeletedCallback_
	)
    : onNodeUpdatedCallback(onNodeUpdatedCallback_),
      onNodeDeletedCallback(onNodeDeletedCallback_) {
}

void SolidSpaceChangeHandler::onNodeAdded(const OctreeNodeData& nodeData) const {
    onNodeUpdatedCallback(nodeData);
};

void SolidSpaceChangeHandler::onNodeDeleted(const OctreeNodeData& nodeData) const {
    onNodeDeletedCallback(nodeData);
};
