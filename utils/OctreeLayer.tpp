#pragma once
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include "../math/InstanceGeometry.tpp"
#include "../space/OctreeNode.hpp"
#include "NodeInfo.tpp"

template <typename T> struct OctreeLayer {
	private:
	using MapType   = std::unordered_map<OctreeNode*, NodeInfo<T>>;
	using Iterator  = typename MapType::iterator;
	MapType info;
    std::shared_mutex infoMutex;

	
	public:
	void erase(OctreeNode* node) {
		if(node!=NULL) {
			std::unique_lock<std::shared_mutex> lock(infoMutex);
			info.erase(node);
		}
	};

	size_t size() {
		return info.size();
	}

	NodeInfo<T>* find(OctreeNode * node) {
		std::unique_lock<std::shared_mutex> lock(infoMutex);
		Iterator it = info.find(node);
		Iterator end = info.end();
		if (it == end) {
			return NULL;
		}
		return &it->second;
	}

	std::pair<Iterator, bool> tryInsert(OctreeNode * node, InstanceGeometry<T>* loadable) {
		std::unique_lock<std::shared_mutex> lock(infoMutex);
		return info.try_emplace(node, loadable);
	}
};
