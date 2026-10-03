#include "OctreeNodeFile.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "OctreeAllocator.hpp"
#include "ChildBlock.hpp"
#include "../sdf/SDF.hpp"
#include "../math/Math.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace {

std::string getChunkName(const BoundingCube &cube) {
    glm::vec3 p = cube.getMin();
    return std::to_string(cube.getLengthX()) + "_" + std::to_string(p.x) + "_" +  std::to_string(p.y) + "_" + std::to_string(p.z);
}

OctreeNodeSerialized packNode(OctreeNode * node) {
    OctreeNodeSerialized n = OctreeNodeSerialized();
    n.brushIndex = node->vertex.brushIndex;
    n.hsv = node->vertex.hsv;
    n.bits = node->bits;
    SDF::copySDF(node->sdf, n.sdf);
    n.lod = node->getLod();
    n.chunkLod = node->getChunkLod();
    return n;
}

OctreeNode * allocateNode(Octree * tree, OctreeNodeSerialized &serialized, const BoundingCube &cube) {
    glm::vec3 position = SDF::getPosition(serialized.sdf, cube);
    glm::vec3 normal = SDF::getNormalFromPosition(serialized.sdf, cube, position);
    Vertex vertex(position, normal, glm::vec2(0), serialized.brushIndex);
    vertex.hsv = serialized.hsv;
    OctreeNode * node = tree->allocator->allocate()->init(vertex);
    node->setSDF(serialized.sdf);
    node->bits = serialized.bits;
    node->setLod(serialized.lod);
    node->setChunkLod(serialized.chunkLod);
    node->setBrush(serialized.brushIndex);
    return node;
}

} // namespace

OctreeNodeFile::OctreeNodeFile(Octree * tree_, OctreeNode * node_, std::string filename_) {
	this->node = node_;
	this->filename = filename_;
	this->tree = tree_;
}

OctreeNode * OctreeNodeFile::loadSubtree(Octree * tree, OctreeNode * workingNode, int i,
                                         const BoundingCube &cube,
                                         std::vector<OctreeNodeSerialized> * nodes,
                                         bool chunked, float chunkSize,
                                         const std::string &filename_,
                                         const std::string &baseFolder) {
	OctreeNodeSerialized serialized = nodes->at(i);
	if(workingNode == NULL) {
		workingNode = allocateNode(tree, serialized, cube);
	}

	bool isLeaf = true;
	for(int j=0; j < 8; ++j) {
		if(serialized.children[j] != 0) {
			isLeaf = false;
			break;
		}
	}
	ChildBlock * block = isLeaf ? NULL : workingNode->allocate(*tree->allocator)->init();
	if(chunked && cube.getLengthX() <= chunkSize) {
		std::string chunkName = getChunkName(cube);
		OctreeNodeFile file(tree, workingNode, baseFolder + "/" + filename_ + "_" + chunkName + ".bin");
		file.load(baseFolder, cube);
	} else {
		for(int j=0 ; j <8 ; ++j){
			int index = serialized.children[j];
			if(index != 0) {
				BoundingCube c = cube.getChild(j);
				block->set(j, loadSubtree(tree, NULL, index, c, nodes, chunked, chunkSize, filename_, baseFolder), *tree->allocator);
			}
		}
	}

	return workingNode;
}

void OctreeNodeFile::load(std::string baseFolder, const BoundingCube &cube) {
	std::ifstream file = std::ifstream(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    std::stringstream decompressed = gzipDecompressFromIfstream(file);

	size_t size;
	decompressed.read(reinterpret_cast<char*>(&size), sizeof(size_t) );

	std::vector<OctreeNodeSerialized> nodes;
	nodes.resize(size);

   	decompressed.read(reinterpret_cast<char*>(nodes.data()), size * sizeof(OctreeNodeSerialized));
	loadSubtree(tree, node, 0, cube, &nodes, false, 0.0f, filename, "");
    file.close();
	nodes.clear();
}

uint OctreeNodeFile::saveSubtree(Octree * tree, OctreeNode * inNode,
                                 std::vector<OctreeNodeSerialized> * nodes,
                                 bool chunked, float chunkSize,
                                 const std::string &filename_,
                                 const BoundingCube &cube,
                                 const std::string &baseFolder) {
	if(inNode != NULL) {
		OctreeNodeSerialized n = packNode(inNode);

		uint index = nodes->size();
		nodes->push_back(n);

		if(chunked && cube.getLengthX() <= chunkSize) {
			std::string chunkName = getChunkName(cube);
			OctreeNodeFile file(tree, inNode, baseFolder + "/" + filename_ + "_" + chunkName + ".bin");
			file.save(baseFolder);
		} else {
			OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
			inNode->getChildren(*tree->allocator, children);

			for(int i=0; i < 8; ++i) {
				BoundingCube c = cube.getChild(i);
				(*nodes)[index].children[i] = saveSubtree(tree, children[i], nodes, chunked, chunkSize, filename_, c, baseFolder);
			}
		}
		return index;
	}
	return 0;
}

void OctreeNodeFile::save(std::string baseFolder){
    std::vector<OctreeNodeSerialized> nodes;

	std::ofstream file = std::ofstream(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

	saveSubtree(tree, node, &nodes, false, 0.0f, filename, BoundingCube(), "");

	size_t size = nodes.size();

    std::ostringstream decompressed;
	decompressed.write(reinterpret_cast<const char*>(&size), sizeof(size_t) );
	decompressed.write(reinterpret_cast<const char*>(nodes.data()), nodes.size() * sizeof(OctreeNodeSerialized) );

	std::istringstream inputStream(decompressed.str());
 	gzipCompressToOfstream(inputStream, file);
	file.close();

	nodes.clear();
}
