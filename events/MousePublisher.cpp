#include "MousePublisher.hpp"
#include "EventManager.hpp"
#include "../math/Camera.hpp"

#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "ControllerInput.hpp"
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "../utils/Brush3dManager.hpp"
#include <cmath>

MousePublisher* MousePublisher::s_instance = nullptr;

void MousePublisher::scrollCallback(GLFWwindow* w, double xoffset, double yoffset) {
    if (s_instance) s_instance->scrollAccum += static_cast<float>(yoffset);
    // Chain to the previous callback (e.g. ImGui's) so UI scrolling keeps working.
    if (s_instance && s_instance->prevScrollFun)
        s_instance->prevScrollFun(w, xoffset, yoffset);
}

void MousePublisher::attachWindow(GLFWwindow* window_) {
    this->window = window_;
    s_instance = this;
    // Chain the existing scroll callback (ImGui installs one via glfwSetScrollCallback).
    prevScrollFun = glfwSetScrollCallback(window, scrollCallback);
}

void MousePublisher::update(EventManager* em, const Camera& cam, float deltaTime,
                            ControllerManager* cm, Brush3dManager* brushManager,
                            bool imguiWantsMouse) {
    if (!window || !em || !cm) return;
    (void)deltaTime;

    ControllerContext& mctx = cm->mouseContext;

    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    if (!haveLast) { lastX = x; lastY = y; haveLast = true; }
    double dx = x - lastX;
    double dy = y - lastY;

    bool lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    // Suppress mouse-camera events when on a non-propagating page OR when ImGui
    // is capturing the mouse, so ImGui windows keep working perfectly.
    if (mctx.isNoPropagate() || imguiWantsMouse) {
        scrollAccum = 0.0f;
        lastX = x; lastY = y;
        return;
    }

    ControllerAction action;
    glm::vec3 forward = cam.getForward();
    glm::vec3 up = cam.getUp();
    glm::vec3 right = cam.getRight();

    const PageCategory cat = mctx.activeCategory();

    if (cat == PageCategory::CAMERA) {
        float panSens = cam.speed * 0.0005f;       // world units per pixel
        float scrollDist = scrollAccum * cam.speed * 0.1f; // world units per notch

        // Combined Transform subpage: left-drag rotates, scroll dollies along
        // forward, right-drag pans sideways / up. (The UI subpage is handled by
        // the non-propagating early return above.)
        if (lmb) {
            action.rotateDeg.x += static_cast<float>(-dx * rotateSens);
            action.rotateDeg.y += static_cast<float>(-dy * rotateSens);
        }
        if (scrollAccum != 0.0f) action.translate += forward * scrollDist;
        if (rmb) {
            action.translate += right * static_cast<float>(dx * panSens);
            action.translate += up * static_cast<float>(-dy * panSens);
        }
    }
    // BRUSH manipulation with the mouse is intentionally not implemented yet.

    applyControllerAction(mctx, em, brushManager, action);

    scrollAccum = 0.0f;
    lastX = x; lastY = y;
}
