#include "SolidSpaceChangeHandler.hpp"


SolidSpaceChangeHandler::SolidSpaceChangeHandler(
		const NodeDataCallback& onNodeUpdatedCallback,
    	const NodeDataCallback& onNodeDeletedCallback
	)
    : onNodeUpdatedCallback(onNodeUpdatedCallback),
      onNodeDeletedCallback(onNodeDeletedCallback) {
}

void SolidSpaceChangeHandler::onNodeAdded(const OctreeNodeData& nodeData) const {
    onNodeUpdatedCallback(nodeData);
};

void SolidSpaceChangeHandler::onNodeDeleted(const OctreeNodeData& nodeData) const {
    onNodeDeletedCallback(nodeData);
};
