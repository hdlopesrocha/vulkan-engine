#pragma once

#include "Widget.hpp"
#include "../math/Camera.hpp"
#include <imgui.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

class CameraWidget : public Widget {
public:
    CameraWidget(Camera* camera);
    
    void render() override;

private:
    Camera* camera;
};
