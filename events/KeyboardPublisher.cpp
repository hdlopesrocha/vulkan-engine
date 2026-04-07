#include "KeyboardPublisher.hpp"
#include "EventManager.hpp"
#include "../math/Camera.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "CloseWindowEvent.hpp"
#include "ToggleFullscreenEvent.hpp"
#include "ControllerManager.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"
#include <algorithm>

KeyboardPublisher::KeyboardPublisher(float moveSpeed_, float angularSpeedDeg_)
    : moveSpeed(moveSpeed_), angularSpeedDeg(angularSpeedDeg_) {}

void KeyboardPublisher::update(GLFWwindow* window, EventManager* em, const Camera& cam, float deltaTime, ControllerManager* controllerManager, Brush3dManager* brushManager, bool flipRotation) {
    if (!window || !em) return;

    // compute camera axes (camera provides normalized vectors)
    glm::vec3 forward = cam.getForward();
    glm::vec3 up = cam.getUp();
    glm::vec3 right = cam.getRight();

    // Use Camera's configured speeds so UI changes in CameraWidget take effect.
    float velocity = cam.speed * deltaTime;
    float angDeg = glm::degrees(cam.angularSpeedRad) * deltaTime;
    float rotSign = flipRotation ? -1.0f : 1.0f;

    // Controller override (if present)
    ControllerParameters* cp = nullptr;
    if (controllerManager) cp = controllerManager->getParameters();
    if (cp) {
        // If numeric keys pressed, switch pages (1..5)
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) cp->currentPage = ControllerParameters::CAMERA;
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) cp->currentPage = ControllerParameters::BRUSH_POSITION;
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) cp->currentPage = ControllerParameters::BRUSH_SCALE;
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) cp->currentPage = ControllerParameters::BRUSH_ROTATION;
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) cp->currentPage = ControllerParameters::BRUSH_PROPERTIES;
    }

    // Translation keys (continuous while held)
    // Translation keys (continuous while held)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        if (cp && cp->currentPage != ControllerParameters::CAMERA) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->translate += forward * (cp->cameraMoveSpeed * deltaTime);
            }
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(forward * velocity));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        if (cp && cp->currentPage != ControllerParameters::CAMERA) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->translate -= forward * (cp->cameraMoveSpeed * deltaTime);
            }
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(-forward * velocity));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_SCALE) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->scale.x = std::max(0.001f, be->scale.x - 0.5f * deltaTime);
            }
        } else if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->yaw -= (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime;
            }
        } else if (cp) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->translate -= right * (cp->cameraMoveSpeed * deltaTime);
            }
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(-right * velocity));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_SCALE) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->scale.x += 0.5f * deltaTime;
            }
        } else if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->yaw += (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime;
            }
        } else if (cp) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->translate += right * (cp->cameraMoveSpeed * deltaTime);
            }
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(right * velocity));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_SCALE) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->scale.y += 0.5f * deltaTime;
            }
        } else if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->roll += (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime;
            }
        } else if (cp) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->translate += up * (cp->cameraMoveSpeed * deltaTime);
            }
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(up * velocity));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_SCALE) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->scale.y = std::max(0.001f, be->scale.y - 0.5f * deltaTime);
            }
        } else if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->roll -= (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime;
            }
        } else if (cp) {
            if (brushManager) {
                BrushEntry* be = brushManager->getSelectedEntry();
                if (be) be->translate -= up * (cp->cameraMoveSpeed * deltaTime);
            }
        } else {
            em->publish(std::make_shared<TranslateCameraEvent>(-up * velocity));
        }
    }

    // Rotation keys (continuous while held)
    // yaw: H (left), F (right) around world up
    // Rotation keys (continuous while held)
    // H/F -> yaw, G/T -> pitch, R/Y -> roll
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) { BrushEntry* be = brushManager->getSelectedEntry(); if (be) be->yaw -= (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime * rotSign; }
        } else {
            em->publish(std::make_shared<RotateCameraEvent>(rotSign * -angDeg, 0.0f, 0.0f));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) { BrushEntry* be = brushManager->getSelectedEntry(); if (be) be->yaw += (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime * rotSign; }
        } else {
            em->publish(std::make_shared<RotateCameraEvent>(rotSign * angDeg, 0.0f, 0.0f));
        }
    }
    // pitch: G (down), T (up) around camera right
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) { BrushEntry* be = brushManager->getSelectedEntry(); if (be) be->pitch -= (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime * rotSign; }
        } else {
            em->publish(std::make_shared<RotateCameraEvent>(0.0f, rotSign * -angDeg, 0.0f));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) { BrushEntry* be = brushManager->getSelectedEntry(); if (be) be->pitch += (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime * rotSign; }
        } else {
            em->publish(std::make_shared<RotateCameraEvent>(0.0f, rotSign * angDeg, 0.0f));
        }
    }
    // roll: R (left), Y (right) around forward
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) { BrushEntry* be = brushManager->getSelectedEntry(); if (be) be->roll -= (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime * rotSign; }
        } else {
            em->publish(std::make_shared<RotateCameraEvent>(0.0f, 0.0f, rotSign * -angDeg));
        }
    }
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) {
        if (cp && cp->currentPage == ControllerParameters::BRUSH_ROTATION) {
            if (brushManager) { BrushEntry* be = brushManager->getSelectedEntry(); if (be) be->roll += (cp ? cp->cameraAngularSpeedDeg : glm::degrees(cam.angularSpeedRad)) * deltaTime * rotSign; }
        } else {
            em->publish(std::make_shared<RotateCameraEvent>(0.0f, 0.0f, rotSign * angDeg));
        }
    }

    // Toggle fullscreen: F11 on key-down
    bool f11Now = (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS);
    if (f11Now && !f11Prev) {
        em->publish(std::make_shared<ToggleFullscreenEvent>());
    }
    f11Prev = f11Now;

    // Close window: ESC on key-down
    bool escNow = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS);
    if (escNow && !escPrev) {
        em->publish(std::make_shared<CloseWindowEvent>());
    }
    escPrev = escNow;
}
