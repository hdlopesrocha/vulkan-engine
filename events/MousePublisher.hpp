#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class EventManager;
class Camera;

// Polls the mouse each frame and publishes camera events according to the
// active Mouse controller page (owned by ControllerManager). The mouse has its
// own page tree so it can be on a different page than the keyboard/gamepad.
//
//   - "UI" page (non-propagating): raw mouse is NOT turned into app events, so
//     ImGui / other UI can consume the mouse without interference.
//   - Camera > Rotate page:
//       * left  drag  -> rotate camera (yaw/pitch)
//       * scroll wheel -> translate camera along +/- camera forward
//       * right drag   -> pan camera sideways / up
//   - Camera > Translate page: left drag pans, scroll dollies forward/back.
//
// Brush manipulation with the mouse is not implemented yet.
class ControllerManager;
class Brush3dManager;

class MousePublisher {
public:
    MousePublisher() = default;

    // Wire up the GLFW window (chains the existing scroll callback so ImGui
    // keeps working) and start tracking the cursor.
    void attachWindow(GLFWwindow* window);

    // Call each frame to inspect mouse state and publish events.
    // - em: EventManager to publish to
    // - cam: reference to Camera for axis vectors
    // - deltaTime: frame delta seconds
    // - imguiWantsMouse: when true (ImGui is capturing the mouse), mouse-camera
    //   events are suppressed so ImGui windows keep working perfectly.
    void update(EventManager* em, const Camera& cam, float deltaTime,
                ControllerManager* cm, Brush3dManager* brushManager,
                bool imguiWantsMouse);

    void setRotateSensitivity(float v) { rotateSens = v; }

private:
    GLFWwindow* window = nullptr;
    double lastX = 0.0, lastY = 0.0;
    bool haveLast = false;
    float scrollAccum = 0.0f;
    GLFWscrollfun prevScrollFun = nullptr;

    float rotateSens = 0.3f; // degrees per pixel

    static MousePublisher* s_instance;
    static void scrollCallback(GLFWwindow* w, double xoffset, double yoffset);
};
