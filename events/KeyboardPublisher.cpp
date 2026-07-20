#include "KeyboardPublisher.hpp"
#include "EventManager.hpp"
#include "../math/Camera.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "CloseWindowEvent.hpp"
#include "ToggleFullscreenEvent.hpp"
#include "RebuildBrushEvent.hpp"
#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "ControllerInput.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"
#include <algorithm>

KeyboardPublisher::KeyboardPublisher(float moveSpeed_, float angularSpeedDeg_)
    : moveSpeed(moveSpeed_), angularSpeedDeg(angularSpeedDeg_) {}

// Returns true on the rising edge of a key press (press, not held).
static bool edgePressed(GLFWwindow* w, int key, bool& prev) {
    bool now = glfwGetKey(w, key) == GLFW_PRESS;
    bool pressed = now && !prev;
    prev = now;
    return pressed;
}

void KeyboardPublisher::update(GLFWwindow* window, EventManager* em, const Camera& cam, float deltaTime, ControllerManager* cm, Brush3dManager* brushManager, bool flipRotation) {
    if (!window || !em || !cm) return;

    ControllerContext& kctx = cm->keyboardContext;
    const ControllerParameters& cp = *cm->getParameters();

    // Use Camera's configured speeds so UI changes in CameraWidget take effect.
    float velocity = cam.speed * deltaTime;
    float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
    float rotSign = flipRotation ? -1.0f : 1.0f;

    // ---- Page navigation (edge-triggered) ----
    // Keyboard's own pages: F1/F2 page, F3/F4 subpage.
    if (edgePressed(window, GLFW_KEY_F1, k1Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::KEYBOARD, PageNavigationEvent::Action::PREV_PAGE));
    if (edgePressed(window, GLFW_KEY_F2, k2Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::KEYBOARD, PageNavigationEvent::Action::NEXT_PAGE));
    if (edgePressed(window, GLFW_KEY_F3, k3Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::KEYBOARD, PageNavigationEvent::Action::PREV_SUBPAGE));
    if (edgePressed(window, GLFW_KEY_F4, k4Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::KEYBOARD, PageNavigationEvent::Action::NEXT_SUBPAGE));
    // Mouse pages controlled from the keyboard: F5/F6 page, F7/F8 subpage.
    if (edgePressed(window, GLFW_KEY_F5, k5Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::MOUSE, PageNavigationEvent::Action::PREV_PAGE));
    if (edgePressed(window, GLFW_KEY_F6, k6Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::MOUSE, PageNavigationEvent::Action::NEXT_PAGE));
    if (edgePressed(window, GLFW_KEY_F7, k7Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::MOUSE, PageNavigationEvent::Action::PREV_SUBPAGE));
    if (edgePressed(window, GLFW_KEY_F8, k8Prev)) em->publish(std::make_shared<PageNavigationEvent>(ControllerId::MOUSE, PageNavigationEvent::Action::NEXT_SUBPAGE));

    // ---- Map raw keys into a controller-agnostic action based on active page ----
    ControllerAction action;
    glm::vec3 forward = cam.getForward();
    glm::vec3 up = cam.getUp();
    glm::vec3 right = cam.getRight();

    const PageCategory cat = kctx.activeCategory();
    const PageControl ctrl = kctx.activeControl();

    if (cat == PageCategory::CAMERA) {
        // Camera: WASD/QE translate and H/F/G/T/R/Y rotate are always active,
        // matching the original single-camera-page behaviour. Brush rotation is
        // likewise always active; the active Brush subpage only gates the
        // translate/scale/texture/attribute controls, and the mouse controller.
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) action.translate += forward * velocity;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) action.translate -= forward * velocity;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) action.translate += right * velocity;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) action.translate -= right * velocity;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) action.translate += up * velocity;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) action.translate -= up * velocity;
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) action.rotateDeg.x += rotSign * -angDeg;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) action.rotateDeg.x += rotSign *  angDeg;
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) action.rotateDeg.y += rotSign * -angDeg;
        if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) action.rotateDeg.y += rotSign *  angDeg;
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) action.rotateDeg.z += rotSign * -angDeg;
        if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) action.rotateDeg.z += rotSign *  angDeg;
    } else { // BRUSH
        float mSpeed = cp.cameraMoveSpeed * deltaTime;
        float aSpeed = cp.cameraAngularSpeedDeg * deltaTime;

        // Rotation (H/F/G/T/R/Y) is always active on the brush, exactly like the
        // camera, so the brush orients the same way regardless of subpage.
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) action.rotateDeg.x += rotSign * -aSpeed;
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) action.rotateDeg.x += rotSign *  aSpeed;
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) action.rotateDeg.y += rotSign * -aSpeed;
        if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) action.rotateDeg.y += rotSign *  aSpeed;
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) action.rotateDeg.z += rotSign * -aSpeed;
        if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) action.rotateDeg.z += rotSign *  aSpeed;

        // Scale (J/U = X, K/I = Y, L/O = Z) is always active on the brush.
        if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) action.scaleDelta.x += 0.5f * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) action.scaleDelta.x -= 0.5f * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) action.scaleDelta.y += 0.5f * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) action.scaleDelta.y -= 0.5f * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) action.scaleDelta.z += 0.5f * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) action.scaleDelta.z -= 0.5f * deltaTime;

        if (ctrl == PageControl::TRANSLATE) {
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) action.translate += forward * mSpeed;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) action.translate -= forward * mSpeed;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) action.translate += right * mSpeed;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) action.translate -= right * mSpeed;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) action.translate += up * mSpeed;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) action.translate -= up * mSpeed;
        } else if (ctrl == PageControl::SCALE) {
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) action.scaleDelta.x += 0.5f * deltaTime;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) action.scaleDelta.x -= 0.5f * deltaTime;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) action.scaleDelta.y += 0.5f * deltaTime;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) action.scaleDelta.y -= 0.5f * deltaTime;
        } else if (ctrl == PageControl::TEXTURE) {
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) action.textureDelta += 1;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) action.textureDelta -= 1;
        } else if (ctrl == PageControl::ATTRIBUTE) {
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) action.attributeDelta += 1;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) action.attributeDelta -= 1;
        }
    }

    bool brushChanged = applyControllerAction(kctx, em, brushManager, action);

    // Toggle fullscreen: F11 on key-down
    if (edgePressed(window, GLFW_KEY_F11, f11Prev)) {
        em->publish(std::make_shared<ToggleFullscreenEvent>());
    }
    // Close window: ESC on key-down
    if (edgePressed(window, GLFW_KEY_ESCAPE, escPrev)) {
        em->publish(std::make_shared<CloseWindowEvent>());
    }

    if (brushChanged) {
        em->queue(std::make_shared<RebuildBrushEvent>());
    }
}
