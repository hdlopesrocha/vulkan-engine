#include "SolidSpaceChangeHandler.hpp"

// Default empty callback used to initialize reference members
static NodeDataCallback emptyNodeDataCallback;

SolidSpaceChangeHandler::SolidSpaceChangeHandler(
		const NodeDataCallback & onNodeUpdated,
		const NodeDataCallback & onNodeErased
	) : onNodeUpdated(onNodeUpdated), onNodeErased(onNodeErased) {
};

void SolidSpaceChangeHandler::onNodeAdded(const OctreeNodeData& nodeData) const {
    if (onNodeUpdated) {
        onNodeUpdated(nodeData);
    }
};

void SolidSpaceChangeHandler::onNodeDeleted(const OctreeNodeData& nodeData) const {
    if (onNodeErased) {
        onNodeErased(nodeData);
    }
};
