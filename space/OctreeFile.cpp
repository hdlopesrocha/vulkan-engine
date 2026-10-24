#include "OctreeFile.hpp"
#include "OctreeSerialized.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "OctreeNodeFile.hpp"
#include "../math/Math.hpp"
#include <fstream>
#include <sstream>
#include <iostream>



OctreeFile::OctreeFile(Octree * tree_, std::string filename_) {
	this->tree = tree_;
	this->filename = filename_;
}

void OctreeFile::readFromStream(std::istream& in) {
	OctreeSerialized octreeSerialized;
	in.read(reinterpret_cast<char*>(&octreeSerialized), sizeof(OctreeSerialized));

	size_t size = 0;
	in.read(reinterpret_cast<char*>(&size), sizeof(size_t));

	std::vector<OctreeNodeSerialized> nodes;
	nodes.resize(size);
	in.read(reinterpret_cast<char*>(nodes.data()), size * sizeof(OctreeNodeSerialized));

	tree->setMin(octreeSerialized.min);
	tree->setLength(octreeSerialized.length);
	tree->chunkSize = octreeSerialized.chunkSize;
	if (nodes.empty()) {
		tree->root = nullptr;
		return;
	}
	// Force full-tree recursive reconstruction from in-stream node array.
	tree->root = OctreeNodeFile::loadSubtree(tree, NULL, 0, *tree, &nodes, true, 0.0f, filename, "");
}

void OctreeFile::writeToStream(std::ostream& out) {
	std::vector<OctreeNodeSerialized> nodes;

	if (tree->root != nullptr) {
		// Force full-tree recursive flattening into the in-stream node array.
		OctreeNodeFile::saveSubtree(tree, tree->root, &nodes, true, 0.0f, filename, *tree, "");
	}

	OctreeSerialized octreeSerialized;
	octreeSerialized.min = tree->getMin();
	octreeSerialized.length = tree->getLengthX();
	octreeSerialized.chunkSize = tree->chunkSize;

	out.write(reinterpret_cast<const char*>(&octreeSerialized), sizeof(OctreeSerialized));
	size_t size = nodes.size();
	out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
	if (size > 0) {
		out.write(reinterpret_cast<const char*>(nodes.data()), nodes.size() * sizeof(OctreeNodeSerialized));
	}
}

