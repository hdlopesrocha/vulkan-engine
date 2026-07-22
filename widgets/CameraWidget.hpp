#pragma once

#include "Widget.hpp"
#include "../math/Camera.hpp"
#include <imgui.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

class CameraWidget : public Widget {
public:
    CameraWidget(Camera* cam);
    
    void render() override;

private:
    Camera* camera;
};
