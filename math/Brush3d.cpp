#include "Brush3d.hpp"
#include "Camera.hpp"
#include "Math.hpp"

Brush3d::Brush3d() {
   	this->translationSensitivity = 32.0f;
    this->enabled =true;
}

void Brush3d::reset(Camera * camera) {
	// TODO: Reset brush position based on camera direction
    (void)camera;
}
