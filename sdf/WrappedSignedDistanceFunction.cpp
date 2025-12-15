#include "WrappedSignedDistanceFunction.hpp"

WrappedSignedDistanceFunction::WrappedSignedDistanceFunction(SignedDistanceFunction * function)
: function(function) {}

WrappedSignedDistanceFunction::~WrappedSignedDistanceFunction() = default;

SdfType WrappedSignedDistanceFunction::getType() const {
	return function->getType();
}

SignedDistanceFunction * WrappedSignedDistanceFunction::getFunction() {
	return function;
}

float WrappedSignedDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
	if(cacheEnabled){
		mtx.lock();
		auto it = cacheSDF.find(p);
		if (it != cacheSDF.end()) {
			float value = it->second;
			mtx.unlock();
			return value;
		}
		mtx.unlock();
		float result = function->distance(p, model);
		mtx.lock();
		cacheSDF[p] = result;
		mtx.unlock();
		return result;
	} else {
	   return function->distance(p, model);
	}
}

glm::vec3 WrappedSignedDistanceFunction::getCenter(const Transformation &model) const {
	return function->getCenter(model);
}
