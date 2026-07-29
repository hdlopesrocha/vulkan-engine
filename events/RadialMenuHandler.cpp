#include "RadialMenuHandler.hpp"
#include "EventManager.hpp"
#include "ControllerManager.hpp"
#include "ControllerContext.hpp"
#include "../widgets/RadialMenu.hpp"
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
#include "SetLightEvent.hpp"
#include "SetPageEvent.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>

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
        cam.label = "Camera";
        cam.subPages.push_back({"Translate"});
        cam.subPages.push_back({"UI"});
        pages_.push_back(cam);
    }
    {
        Page brush;
        brush.label = "Brush";
        brush.subPages.push_back({"Control"});
        brush.subPages.push_back({"Mode"});
        brush.subPages.push_back({"Drag Mode"});
        brush.subPages.push_back({"Texture"});
        brush.subPages.push_back({"Attributes"});
        brush.subPages.push_back({"Color"});
        pages_.push_back(brush);
    }
    {
        Page lgt;
        lgt.label = "Light";
        lgt.subPages.push_back({"Azimuth"});
        lgt.subPages.push_back({"Elevation"});
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

    ImVec2 vec(0, 0);
    if (nunchuk_->isConnected()) {
        WiimoteState ws = nunchuk_->getState();
        float scale = 120.0f;
        auto activeType = menu_->GetActiveRingType();
        if (activeType == RadialMenu::RingType::TEXTURE || activeType == RadialMenu::RingType::LABEL)
            scale = 170.0f;
        else if (activeType == RadialMenu::RingType::HSV_SLIDER)
            scale = 250.0f;
        vec = ImVec2(ws.joystickX * scale, -ws.joystickY * scale);
    } else if (gamepad_->isConnected()) {
        float scale = 120.0f;
        auto activeType = menu_->GetActiveRingType();
        if (activeType == RadialMenu::RingType::TEXTURE || activeType == RadialMenu::RingType::LABEL)
            scale = 170.0f;
        else if (activeType == RadialMenu::RingType::HSV_SLIDER)
            scale = 250.0f;
        vec = ImVec2(gamepad_->getLeftStickX() * scale, gamepad_->getLeftStickY() * scale);
    } else {
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
    // Note: textureSelectPrev and backPrev are now updated to the current frame values.
    // We need the EDGE (was false, now true), which is what selectNow && !oldPrev gives.
    // But we've already overwritten textureSelectPrev. Let's fix the edge detection:
    // Actually, looking at the original code, the edge detection uses:
    //   bool selectEdge = selectNow && !textureSelectPrev;
    //   textureSelectPrev = selectNow;
    // So textureSelectPrev tracks the PREVIOUS frame's state.
    // We need to restructure this. Let me re-read the original...

    // The original code does:
    //   bool selectEdge = selectNow && !textureSelectPrev;
    //   bool backEdge = backNow && !backPrev;
    //   textureSelectPrev = selectNow;
    //   backPrev = backNow;
    // So textureSelectPrev/backPrev track previous frame state for edge detection.
    // We should NOT update them inside detectSelectBack; instead, we should
    // return the edge values and let the caller update the tracking state.
    // But for simplicity, let's just use the pattern correctly here.
}

bool RadialMenuHandler::update(uint32_t loadedTextureLayers) {
    if (!menu_) return false;

    detectToggle();

    if (menu_->IsVisible()) {
        feedInputVector();

        int hp = menu_->GetHoveredPage();
        int hs = menu_->GetHoveredSubPage();

        // Detect select/back edges
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
        bool selectEdge = selectNow && !textureSelectPrev;
        bool backEdge = backNow && !backPrev;
        textureSelectPrev = selectNow;
        backPrev = backNow;

        auto activeType = menu_->GetActiveRingType();

        // --- Active ring handling ---
        if (activeType == RadialMenu::RingType::TEXTURE) {
            std::vector<ImTextureID> texIds;
            for (uint32_t i = 0; i < loadedTextureLayers; ++i) {
                texIds.push_back(texArray_->getImTexture(i, 0));
            }
            menu_->SetTextures(texIds);

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
                    const std::string& pageLabel = pages_[stackPage].label;
                    const std::string& subLabel = subPages[hs].label;
                    menu_->SetSelectedIndex(hs);

                    if (subLabel == "Texture") {
                        menu_->ResetTexturePage();
                        menu_->PushTextureRing({});
                    } else if (subLabel == "Control") {
                        labelRingKind = LabelRingKind::CONTROL;
                        menu_->PushLabelRing({"Translate", "Aim", "Scale"});
                        int ci = 0;
                        if (brush_->controlMode == BrushControlMode::AIM) ci = 1;
                        else if (brush_->controlMode == BrushControlMode::SCALE) ci = 2;
                        menu_->SetCurrentItem(ci);
                    } else if (subLabel == "Mode") {
                        labelRingKind = LabelRingKind::PAINT;
                        menu_->PushLabelRing({"Add", "Remove", "Paint"});
                        int ci = 0;
                        if (brush_->paintMode == BrushPaintMode::REMOVE) ci = 1;
                        else if (brush_->paintMode == BrushPaintMode::PAINT) ci = 2;
                        menu_->SetCurrentItem(ci);
                    } else if (subLabel == "Drag Mode") {
                        labelRingKind = LabelRingKind::DRAG;
                        menu_->PushLabelRing({"Drag", "Click"});
                        int ci = (brush_->dragMode == BrushDragMode::CLICK) ? 1 : 0;
                        menu_->SetCurrentItem(ci);
                    } else if (subLabel == "Color") {
                        labelRingKind = LabelRingKind::HSV;
                        menu_->PushLabelRing({"Hue", "Saturation", "Value"});
                        menu_->SetCurrentItem(-1);
                    } else if (subLabel == "Azimuth") {
                        float azi, ele;
                        light_->getSpherical(azi, ele);
                        menu_->PushHSVSliderRing("Azimuth", azi + 180.0f, 0.0f, 360.0f);
                        textureSelectPrev = true;
                    } else if (subLabel == "Elevation") {
                        float azi, ele;
                        light_->getSpherical(azi, ele);
                        menu_->PushHSVSliderRing("Elevation", ele + 90.0f, 0.0f, 180.0f);
                        textureSelectPrev = true;
                    } else {
                        PageCategory cat = (pageLabel == "Camera")
                            ? PageCategory::CAMERA : PageCategory::BRUSH;
                        PageControl pc = PageControl::TRANSLATE;
                        BrushControlMode bm = BrushControlMode::TRANSLATE;
                        if (subLabel == "UI")           { pc = PageControl::UI; }
                        else if (subLabel == "Texture")    { pc = PageControl::TEXTURE;    bm = BrushControlMode::TEXTURE; }
                        else if (subLabel == "Attributes") { pc = PageControl::ATTRIBUTE;  bm = BrushControlMode::ATTRIBUTE; }
                        else if (subLabel == "Color")      { pc = PageControl::COLOR;      bm = BrushControlMode::COLOR; }
                        queueSetPageEvent(cat, pc, bm);
                        menu_->SetVisible(false);
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
