#include "RadialMenuHandler.hpp"
#include "EventManager.hpp"
#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "../widgets/RadialMenu.hpp"
#include "../widgets/RadialMenuIcons.hpp"
#include "NunchukPublisher.hpp"
#include "GamepadPublisher.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"
#include "../vulkan/TextureArrayManager.hpp"
#include "../math/Light.hpp"
#include "SetBrushTextureEvent.hpp"
#include "SetBrushControlEvent.hpp"
#include "SetBrushPaintModeEvent.hpp"
#include "SetBrushDragModeEvent.hpp"
#include "SetBrushHSVEvent.hpp"
#include "SetBrushSdfTypeEvent.hpp"
#include "SetLightEvent.hpp"
#include "SetPageEvent.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <cmath>

RadialMenuHandler::RadialMenuHandler(
    GLFWwindow* window,
    EventManager* eventManager,
    RadialMenu* radialMenu,
    NunchukPublisher* nunchuk,
    GamepadPublisher* gamepad,
    ControllerManager* controllerManager,
    Brush3dManager* brushManager,
    TextureArrayManager* textureArrayManager,
    Light* light)
    : window_(window)
    , em_(eventManager)
    , menu_(radialMenu)
    , nunchuk_(nunchuk)
    , gamepad_(gamepad)
    , cm_(controllerManager)
    , brush_(brushManager)
    , texArray_(textureArrayManager)
    , light_(light)
{
}

void RadialMenuHandler::setupPages() {
    pages_.clear();
    {
        Page cam;
        cam.label = ICON_FA_CAMERA;
        cam.textLabel = "Camera";
        cam.subPages.push_back({ICON_FA_ARROWS_ALT, "Translate"});
        cam.subPages.push_back({ICON_FA_STAR, "UI"});
        pages_.push_back(cam);
    }
    {
        Page brush;
        brush.label = ICON_FA_PAINTBRUSH;
        brush.textLabel = "Brush";
        brush.subPages.push_back({ICON_FA_COG, "Control"});
        brush.subPages.push_back({ICON_FA_SLIDERS_H, "Mode"});
        brush.subPages.push_back({ICON_FA_ARROWS_V, "Drag Mode"});
        brush.subPages.push_back({ICON_FA_SHAPES, "Shape"});
        brush.subPages.push_back({ICON_FA_TH, "Texture"});
        brush.subPages.push_back({ICON_FA_EYE, "Attributes"});
        brush.subPages.push_back({ICON_FA_PALETTE, "Color"});
        pages_.push_back(brush);
    }
    {
        Page lgt;
        lgt.label = ICON_FA_LIGHTBULB;
        lgt.textLabel = "Light";
        lgt.subPages.push_back({ICON_FA_GLOBE, "Azimuth"});
        lgt.subPages.push_back({ICON_FA_MOUNTAIN, "Elevation"});
        pages_.push_back(lgt);
    }
    menu_->SetPages(pages_);
}

void RadialMenuHandler::detectToggle() {
    // Home key toggle (edge-triggered)
    bool homeNow = glfwGetKey(window_, GLFW_KEY_HOME) == GLFW_PRESS;
    if (homeNow && !homePrev) {
        menu_->SetVisible(!menu_->IsVisible());
        if (menu_->IsVisible())
            glfwSetCursorPos(window_, 640, 360); // will be overridden by feedInputVector center
    }
    homePrev = homeNow;

    // Middle mouse button toggle (edge-triggered)
    bool middleMouseNow = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    if (middleMouseNow && !middleMousePrev) {
        menu_->SetVisible(!menu_->IsVisible());
        if (menu_->IsVisible())
            glfwSetCursorPos(window_, 640, 360);
    }
    middleMousePrev = middleMouseNow;

    // Wiimote Home button toggle (edge-triggered)
    if (nunchuk_->isConnected()) {
        if (nunchuk_->homeButtonPressed()) {
            menu_->SetVisible(!menu_->IsVisible());
            if (menu_->IsVisible())
                glfwSetCursorPos(window_, 640, 360);
        }
    }

    // Gamepad START button toggle (edge-triggered)
    if (gamepad_->isConnected()) {
        if (gamepad_->startButtonPressed()) {
            menu_->SetVisible(!menu_->IsVisible());
            if (menu_->IsVisible())
                glfwSetCursorPos(window_, 640, 360);
        }
    }
}

void RadialMenuHandler::feedInputVector() {
    int w, h;
    glfwGetWindowSize(window_, &w, &h);
    menu_->SetCenter(ImVec2(w * 0.5f, h * 0.5f));

    auto getScale = [&]() {
        float scale = 120.0f;
        auto activeType = menu_->GetActiveRingType();
        if (activeType == RadialMenu::RingType::TEXTURE || activeType == RadialMenu::RingType::LABEL)
            scale = 170.0f;
        else if (activeType == RadialMenu::RingType::HSV_SLIDER)
            scale = 250.0f;
        return scale;
    };

    ImVec2 vec(0, 0);
    bool usedAnalog = false;

    // Nunchuk joystick — use only when outside deadzone
    if (nunchuk_->isConnected()) {
        WiimoteState ws = nunchuk_->getState();
        constexpr float deadzone = 0.15f;
        if (std::abs(ws.joystickX) > deadzone || std::abs(ws.joystickY) > deadzone) {
            float scale = getScale();
            vec = ImVec2(ws.joystickX * scale, -ws.joystickY * scale);
            usedAnalog = true;
        }
    }

    // Gamepad left stick — use only when outside deadzone
    if (!usedAnalog && gamepad_->isConnected()) {
        float lx = gamepad_->getLeftStickX();
        float ly = gamepad_->getLeftStickY();
        if (lx != 0.0f || ly != 0.0f) {
            float scale = getScale();
            vec = ImVec2(lx * scale, ly * scale);
            usedAnalog = true;
        }
    }

    // Mouse fallback — use when no analog stick is active
    if (!usedAnalog) {
        double mx, my;
        glfwGetCursorPos(window_, &mx, &my);
        vec = ImVec2(
            static_cast<float>(mx) - w * 0.5f,
            static_cast<float>(my) - h * 0.5f
        );
    }

    menu_->SetInputVector(vec);
}

void RadialMenuHandler::detectSelectBack() {
    bool selectNow = false;
    bool backNow = false;
    if (nunchuk_->isConnected()) {
        selectNow = nunchuk_->cButtonPressed();
        backNow = nunchuk_->zButtonPressed();
    } else if (gamepad_->isConnected()) {
        selectNow = gamepad_->aButtonPressed();
        backNow = gamepad_->bButtonPressed();
    } else {
        selectNow = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        backNow = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    }
    textureSelectPrev = selectNow && !textureSelectPrev;
    backPrev = backNow && !backPrev;
}

bool RadialMenuHandler::update(uint32_t loadedTextureLayers) {
    if (!menu_) return false;

    detectToggle();

    if (menu_->IsVisible()) {
        feedInputVector();

        int hp = menu_->GetHoveredPage();
        int hs = menu_->GetHoveredSubPage();

        // Detect select/back edges — prefer analog buttons when pressed,
        // otherwise fall through to mouse so mouse can navigate freely.
        bool selectNow = false;
        bool backNow = false;
        if (nunchuk_->isConnected()) {
            selectNow = nunchuk_->cButtonPressed();
            backNow = nunchuk_->zButtonPressed();
        }
        if (!selectNow && !backNow && gamepad_->isConnected()) {
            selectNow = gamepad_->aButtonPressed();
            backNow = gamepad_->bButtonPressed();
        }
        if (!selectNow && !backNow) {
            selectNow = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
            backNow = (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        }
        bool selectEdge = selectNow && !textureSelectPrev;
        bool backEdge = backNow && !backPrev;
        textureSelectPrev = selectNow;
        backPrev = backNow;

        auto activeType = menu_->GetActiveRingType();

        // --- Active ring handling ---
        if (activeType == RadialMenu::RingType::TEXTURE) {
            textureScratch_.clear();
            for (uint32_t i = 0; i < loadedTextureLayers; ++i) {
                textureScratch_.push_back(texArray_->getImTexture(i, 0));
            }
            menu_->SetTextures(textureScratch_);

            if (selectEdge) {
                int ht = menu_->GetHoveredTexture();
                if (ht >= 0) {
                    menu_->SetSelectedIndex(ht);
                    em_->queue(std::make_shared<SetBrushTextureEvent>(ht));
                } else if (menu_->GetHoveredNavPrev()) {
                    menu_->SetTexturePage(menu_->GetTexturePage() - 1);
                } else if (menu_->GetHoveredNavNext()) {
                    menu_->SetTexturePage(menu_->GetTexturePage() + 1);
                }
            }
        } else if (activeType == RadialMenu::RingType::LABEL) {
            int hl = menu_->GetHoveredLabel();
            if (hl >= 0 && selectEdge) {
                menu_->SetSelectedIndex(hl);
                if (labelRingKind == LabelRingKind::HSV) {
                    const char* compNames[] = { "Hue", "Saturation", "Value" };
                    float compVals[] = { brush_->getHue(), brush_->getSaturation(), brush_->getValue() };
                    float compMax[] = { 360.0f, 100.0f, 100.0f };
                    menu_->PushHSVSliderRing(compNames[hl], compVals[hl], 0.0f, compMax[hl]);
                    textureSelectPrev = true;
                } else if (labelRingKind == LabelRingKind::PAINT) {
                    BrushPaintMode pm = BrushPaintMode::ADD;
                    if (hl == 0) pm = BrushPaintMode::ADD;
                    else if (hl == 1) pm = BrushPaintMode::REMOVE;
                    else if (hl == 2) pm = BrushPaintMode::PAINT;
                    em_->queue(std::make_shared<SetBrushPaintModeEvent>(pm));
                    menu_->PopRing();
                } else if (labelRingKind == LabelRingKind::DRAG) {
                    BrushDragMode dm = BrushDragMode::DRAG;
                    if (hl == 0) dm = BrushDragMode::DRAG;
                    else if (hl == 1) dm = BrushDragMode::CLICK;
                    em_->queue(std::make_shared<SetBrushDragModeEvent>(dm));
                    menu_->PopRing();
                } else if (labelRingKind == LabelRingKind::SHAPE) {
                    em_->queue(std::make_shared<SetBrushSdfTypeEvent>(hl));
                    menu_->PopRing();
                } else {
                    BrushControlMode mode = BrushControlMode::TRANSLATE;
                    if (hl == 0) mode = BrushControlMode::TRANSLATE;
                    else if (hl == 1) mode = BrushControlMode::AIM;
                    else if (hl == 2) mode = BrushControlMode::SCALE;
                    em_->queue(std::make_shared<SetBrushControlEvent>(mode));
                    menu_->PopRing();
                }
            }
        } else if (activeType == RadialMenu::RingType::HSV_SLIDER) {
            float val = menu_->GetHSVSliderValue();
            std::string comp = menu_->GetHSVSliderName();
            if (comp == "Azimuth" || comp == "Elevation") {
                em_->queue(std::make_shared<SetLightEvent>(comp, val));
            } else {
                em_->queue(std::make_shared<SetBrushHSVEvent>(comp, val));
            }
            if (selectEdge) {
                menu_->PopRing();
            }
        } else if (activeType == RadialMenu::RingType::NONE) {
            if (selectEdge && hp >= 0 && hp < static_cast<int>(pages_.size())) {
                menu_->PushSubpageRing(hp);
            }
        } else if (activeType == RadialMenu::RingType::SUBPAGE) {
            int stackPage = menu_->GetStackPageIndex();
            if (selectEdge && stackPage >= 0 && hs >= 0
                && stackPage < static_cast<int>(pages_.size()))
            {
                const auto& subPages = pages_[stackPage].subPages;
                if (hs < static_cast<int>(subPages.size())) {
                    menu_->SetSelectedIndex(hs);

                    // Camera page (index 0): subpage 0=Translate, 1=UI
                    if (stackPage == 0 && hs == 0) {
                        PageCategory cat = PageCategory::CAMERA;
                        PageControl pc = PageControl::TRANSLATE;
                        BrushControlMode bm = BrushControlMode::TRANSLATE;
                        queueSetPageEvent(cat, pc, bm);
                        menu_->SetVisible(false);
                    } else if (stackPage == 0 && hs == 1) {
                        PageCategory cat = PageCategory::CAMERA;
                        PageControl pc = PageControl::UI;
                        BrushControlMode bm = BrushControlMode::TRANSLATE;
                        queueSetPageEvent(cat, pc, bm);
                        menu_->SetVisible(false);
                    }
                    // Brush page (index 1): subpage 0=Control, 1=Mode, 2=DragMode,
                    // 3=Shape, 4=Texture, 5=Attributes, 6=Color
                    else if (stackPage == 1 && hs == 0) {
                        labelRingKind = LabelRingKind::CONTROL;
                        menu_->PushLabelRing({ICON_FA_MOVE, ICON_FA_CROSSHAIRS, ICON_FA_EXPAND_ARROWS},
                                             {"Translate", "Aim", "Scale"});
                        int ci = 0;
                        if (brush_->controlMode == BrushControlMode::AIM) ci = 1;
                        else if (brush_->controlMode == BrushControlMode::SCALE) ci = 2;
                        menu_->SetCurrentItem(ci);
                    } else if (stackPage == 1 && hs == 1) {
                        labelRingKind = LabelRingKind::PAINT;
                        menu_->PushLabelRing({ICON_FA_PLUS, ICON_FA_MINUS, ICON_FA_PAINT_BRUSH},
                                             {"Add", "Remove", "Paint"});
                        int ci = 0;
                        if (brush_->paintMode == BrushPaintMode::REMOVE) ci = 1;
                        else if (brush_->paintMode == BrushPaintMode::PAINT) ci = 2;
                        menu_->SetCurrentItem(ci);
                    } else if (stackPage == 1 && hs == 2) {
                        labelRingKind = LabelRingKind::DRAG;
                        menu_->PushLabelRing({ICON_FA_HAND_POINTER, ICON_FA_STOP},
                                             {"Drag", "Click"});
                        int ci = (brush_->dragMode == BrushDragMode::CLICK) ? 1 : 0;
                        menu_->SetCurrentItem(ci);
                    } else if (stackPage == 1 && hs == 3) {
                        labelRingKind = LabelRingKind::SHAPE;
                        menu_->PushLabelRing({"Sphere", "Box", "Capsule", "Octahedron", "Pyramid",
                                              "Torus", "Cone", "Cylinder", "Tapered Cylinder", "Tapered Capsule"},
                                             {"Sphere", "Box", "Capsule", "Octahedron", "Pyramid",
                                              "Torus", "Cone", "Cylinder", "Tapered Cylinder", "Tapered Capsule"});
                        BrushEntry* be = brush_->getSelectedEntry();
                        menu_->SetCurrentItem(be ? be->sdfType : -1);
                    } else if (stackPage == 1 && hs == 4) {
                        menu_->ResetTexturePage();
                        menu_->PushTextureRing({});
                    } else if (stackPage == 1 && hs == 5) {
                        PageCategory cat = PageCategory::BRUSH;
                        PageControl pc = PageControl::ATTRIBUTE;
                        BrushControlMode bm = BrushControlMode::ATTRIBUTE;
                        queueSetPageEvent(cat, pc, bm);
                        menu_->SetVisible(false);
                    } else if (stackPage == 1 && hs == 6) {
                        labelRingKind = LabelRingKind::HSV;
                        menu_->PushLabelRing({ICON_FA_TINT, ICON_FA_ADJUST, ICON_FA_SUN},
                                             {"Hue", "Saturation", "Value"});
                        menu_->SetCurrentItem(-1);
                    }
                    // Light page (index 2): subpage 0=Azimuth, 1=Elevation
                    else if (stackPage == 2 && hs == 0) {
                        float azi, ele;
                        light_->getSpherical(azi, ele);
                        menu_->PushHSVSliderRing("Azimuth", azi + 180.0f, 0.0f, 360.0f);
                        textureSelectPrev = true;
                    } else if (stackPage == 2 && hs == 1) {
                        float azi, ele;
                        light_->getSpherical(azi, ele);
                        menu_->PushHSVSliderRing("Elevation", ele + 90.0f, 0.0f, 180.0f);
                        textureSelectPrev = true;
                    }
                }
            }
        }

        // Z = back: pop last ring or close menu
        if (backEdge && !selectEdge) {
            if (menu_->GetStackDepth() > 0)
                menu_->PopRing();
            else
                menu_->SetVisible(false);
        }
    } else {
        textureSelectPrev = false;
    }

    return menu_->IsVisible();
}

void RadialMenuHandler::queueSetPageEvent(PageCategory cat, PageControl ctrl, BrushControlMode bm) {
    em_->queue(std::make_shared<SetPageEvent>(cat, ctrl, bm));
}
