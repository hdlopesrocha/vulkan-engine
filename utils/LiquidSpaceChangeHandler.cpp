#include "LiquidSpaceChangeHandler.hpp"

// Default empty callback used to initialize reference members
static NodeDataCallback emptyNodeDataCallback;

LiquidSpaceChangeHandler::LiquidSpaceChangeHandler(
		const NodeDataCallback & onNodeUpdated,
		const NodeDataCallback & onNodeErased
	) : onNodeUpdated(onNodeUpdated), onNodeErased(onNodeErased) {
};


void LiquidSpaceChangeHandler::onNodeAdded(const OctreeNodeData& nodeData) const {
    if (onNodeUpdated) {
        onNodeUpdated(nodeData);
    }
};

void LiquidSpaceChangeHandler::onNodeDeleted(const OctreeNodeData& nodeData) const {
    if (onNodeErased) {
        onNodeErased(nodeData);
    }
};
