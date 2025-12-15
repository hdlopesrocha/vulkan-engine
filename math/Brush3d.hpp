#pragma once
#include "Camera.hpp"

class Brush3d {
public:
    float translationSensitivity;
    bool enabled;
    Brush3d();
    void reset(Camera * camera);
};

 
