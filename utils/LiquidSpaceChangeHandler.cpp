#include "LiquidSpaceChangeHandler.hpp"

// Default empty callback used to initialize reference members
static NodeDataCallback emptyNodeDataCallback;


LiquidSpaceChangeHandler::LiquidSpaceChangeHandler(
		const NodeDataCallback& onNodeUpdatedCallback,
    	const NodeDataCallback& onNodeDeletedCallback
	)
    : onNodeUpdatedCallback(onNodeUpdatedCallback),
      onNodeDeletedCallback(onNodeDeletedCallback) {
}

void LiquidSpaceChangeHandler::onNodeAdded(const OctreeNodeData& nodeData) const {
    onNodeUpdatedCallback(nodeData);
};

void LiquidSpaceChangeHandler::onNodeDeleted(const OctreeNodeData& nodeData) const {
    onNodeDeletedCallback(nodeData);
};

