#include "OctreeFile.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "OctreeAllocator.hpp"
#include "ChildBlock.hpp"
#include "OctreeNodeFile.hpp"
#include "../sdf/SDF.hpp"
#include "../math/Math.hpp"



OctreeFile::OctreeFile(Octree * tree, std::string filename) {
	this->tree = tree;
	this->filename = filename;
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
	tree->root = loadRecursive(0, &nodes, 0.0f, filename, *tree, "");
}

void OctreeFile::writeToStream(std::ostream& out) {
	std::vector<OctreeNodeSerialized> nodes;

	if (tree->root != nullptr) {
		// Force full-tree recursive flattening into the in-stream node array.
		saveRecursive(tree->root, &nodes, 0.0f, filename, *tree, "");
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

std::string getChunkName(BoundingCube cube) {
	glm::vec3 p = cube.getMin();
	return std::to_string(cube.getLengthX()) + "_" + std::to_string(p.x) + "_" +  std::to_string(p.y) + "_" + std::to_string(p.z);
}

OctreeNode * OctreeFile::loadRecursive(int i, std::vector<OctreeNodeSerialized> * nodes, float chunkSize, std::string filename, BoundingCube cube, std::string baseFolder) {
	OctreeNodeSerialized serialized = nodes->at(i);
	glm::vec3 position = SDF::getAveragePosition(serialized.sdf, cube);
	glm::vec3 normal = SDF::getNormalFromPosition(serialized.sdf, cube, position);
	Vertex vertex(position, normal, glm::vec2(0), serialized.brushIndex);

	OctreeNode * node = tree->allocator->allocate()->init(vertex);
	node->setSDF(serialized.sdf);
	node->bits = serialized.bits;
	bool isLeaf = true;
	for(int j=0; j < 8; ++j) {
		if(serialized.children[j] != 0) {
			isLeaf = false;
			break;
		}
	}
	ChildBlock * block = isLeaf ? NULL : node->allocate(*tree->allocator)->init();
	if(cube.getLengthX() > chunkSize) {
		for(int j=0 ; j <8 ; ++j){
			int index = serialized.children[j];
			if(index != 0) {
				BoundingCube c = cube.getChild(j);
				block->set(j , loadRecursive(index, nodes, chunkSize, filename, c,baseFolder), *tree->allocator);
			}
		}
	} else {
		std::string chunkName = getChunkName(cube);
		OctreeNodeFile * file = new OctreeNodeFile(tree, node, baseFolder + "/" + filename+ "_" + chunkName + ".bin");
		//NodeInfo info(INFO_TYPE_FILE, file, NULL, true);
		//node->info.push_back(info);
		file->load(baseFolder, cube);
		delete file;
	}
	return node;
}


void OctreeFile::load(std::string baseFolder, float chunkSize) {
	std::string filePath = baseFolder + "/" + filename+".bin";
	std::ifstream file = std::ifstream(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading: " << filePath << std::endl;
        return;
    }

    std::stringstream decompressed = gzipDecompressFromIfstream(file);

	if (chunkSize == 0.0f) {
		readFromStream(decompressed);
	} else {
		OctreeSerialized octreeSerialized;
		decompressed.read(reinterpret_cast<char*>(&octreeSerialized), sizeof(OctreeSerialized) );

		size_t size;
		decompressed.read(reinterpret_cast<char*>(&size), sizeof(size_t) );

		std::vector<OctreeNodeSerialized> nodes;
		nodes.resize(size);
	   	decompressed.read(reinterpret_cast<char*>(nodes.data()), size * sizeof(OctreeNodeSerialized));

		tree->setMin(octreeSerialized.min);
		tree->setLength(octreeSerialized.length);
		tree->chunkSize = octreeSerialized.chunkSize;
		tree->root = loadRecursive(0,&nodes, chunkSize, filename, *tree, baseFolder);
	}

    file.close();

	std::cout << "OctreeFile::load('" << filePath <<"') Ok!" << std::endl;
}


uint OctreeFile::saveRecursive(OctreeNode * node, std::vector<OctreeNodeSerialized> * nodes, float chunkSize, std::string filename, BoundingCube cube, std::string baseFolder) {
	if(node!=NULL) {
		OctreeNodeSerialized n = OctreeNodeSerialized();
		n.brushIndex = node->vertex.texIndex;
		n.bits = node->bits;
		SDF::copySDF(node->sdf, n.sdf);

		uint index = nodes->size(); 
		nodes->push_back(n);

		if(cube.getLengthX() > chunkSize) {
			OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
			node->getChildren(*tree->allocator, children);

			for(int i=0; i < 8; ++i) {
				BoundingCube c = cube.getChild(i);
				(*nodes)[index].children[i] = saveRecursive(children[i], nodes, chunkSize, filename, c, baseFolder);
			}
		} else {
			std::string chunkName = getChunkName(cube);
			OctreeNodeFile file(tree, node, baseFolder + "/" + filename + "_" + chunkName + ".bin");
			file.save(baseFolder);
		}
		return index;
	}
	return 0;
}

void OctreeFile::save(std::string baseFolder, float chunkSize){
	ensureFolderExists(baseFolder);
    std::vector<OctreeNodeSerialized> nodes;
	std::string filePath = baseFolder + "/" + filename+".bin";
	std::ofstream file = std::ofstream(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

	saveRecursive(tree->root, &nodes, chunkSize, filename, *tree, baseFolder);

    std::ostringstream decompressed;
	if (chunkSize == 0.0f) {
		writeToStream(decompressed);
	} else {
		OctreeSerialized  octreeSerialized;
		octreeSerialized.min = tree->getMin();
		octreeSerialized.length = tree->getLengthX();
		octreeSerialized.chunkSize = tree->chunkSize;

		decompressed.write(reinterpret_cast<const char*>(&octreeSerialized), sizeof(OctreeSerialized));

		size_t size = nodes.size();
		decompressed.write(reinterpret_cast<const char*>(&size), sizeof(size_t) );
		decompressed.write(reinterpret_cast<const char*>(nodes.data()), nodes.size() * sizeof(OctreeNodeSerialized) );
	}
	
	std::istringstream inputStream(decompressed.str());
 	gzipCompressToOfstream(inputStream, file);
	file.close();

	nodes.clear();

	std::cout << "OctreeFile::save('" << filePath <<"') Ok!" << std::endl;

}

AbstractBoundingBox& OctreeFile::getBox(){
	return *this->tree;
}
