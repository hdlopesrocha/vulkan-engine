#pragma once

#include <glm/glm.hpp>
#include "ControllerContext.hpp"
#include "EventManager.hpp"
#include "TranslateCameraEvent.hpp"
#include "RotateCameraEvent.hpp"
#include "../utils/Brush3dManager.hpp"
#include "../utils/Brush3dEntry.hpp"

// Aggregated input a publisher wants to apply this frame, expressed in a
// controller-agnostic way. The publisher is responsible for mapping its raw
// inputs (keys, sticks, mouse drag, scroll) into these deltas according to the
// active control; this function only decides WHERE they go (camera vs brush)
// based on the active page of the context. Reused by every controller.
struct ControllerAction {
    glm::vec3 translate = glm::vec3(0.0f); // world-space delta this frame
    glm::vec3 rotateDeg = glm::vec3(0.0f); // yaw,pitch,roll in degrees this frame
    glm::vec3 scaleDelta = glm::vec3(0.0f);// per-axis scale add this frame
    int textureDelta = 0;                  // material index increment
    int attributeDelta = 0;                // generic attribute increment
};

namespace {
    inline bool nonzero(const glm::vec3 &v) {
        return glm::abs(v.x) > 1e-12f || glm::abs(v.y) > 1e-12f || glm::abs(v.z) > 1e-12f;
    }
}

// Route the action to the camera or to the selected brush entry depending on
// the context's active page. Returns true if the brush was modified (caller
// should queue RebuildBrushEvent). Does nothing when the active page is a
// non-propagating (UI) page.
inline bool applyControllerAction(const ControllerContext &ctx, EventManager *em,
                                  Brush3dManager *brush, const ControllerAction &a) {
    if (ctx.isNoPropagate()) return false;
    if (!em) return false;

    if (ctx.activeCategory() == PageCategory::CAMERA) {
        if (nonzero(a.translate))
            em->publish(std::make_shared<TranslateCameraEvent>(a.translate));
        if (nonzero(a.rotateDeg))
            em->publish(std::make_shared<RotateCameraEvent>(a.rotateDeg.x, a.rotateDeg.y, a.rotateDeg.z));
        return false;
    }

    // BRUSH category
    if (!brush) return false;
    BrushEntry *be = brush->getSelectedEntry();
    if (!be) return false;

    // Translate, rotate and scale are applied from whatever deltas are present,
    // independent of the active subpage, so the Brush > Transform subpage
    // combines all three operations at once.
    bool changed = false;
    if (nonzero(a.translate)) { be->translate += a.translate; changed = true; }
    if (nonzero(a.rotateDeg)) {
        be->yaw += a.rotateDeg.x;
        be->pitch += a.rotateDeg.y;
        be->roll += a.rotateDeg.z;
        changed = true;
    }
    if (nonzero(a.scaleDelta)) {
        be->scale = glm::max(be->scale + a.scaleDelta, glm::vec3(0.001f));
        changed = true;
    }

    // Texture / Attribute remain on their dedicated subpages.
    const PageControl ctrl = ctx.activeControl();
    if (ctrl == PageControl::TEXTURE && a.textureDelta != 0) {
        be->materialIndex = std::max(0, be->materialIndex + a.textureDelta);
        changed = true;
    }
    if (ctrl == PageControl::ATTRIBUTE && a.attributeDelta != 0) {
        be->sdfType = (be->sdfType + a.attributeDelta) % 8;
        if (be->sdfType < 0) be->sdfType += 8;
        changed = true;
    }
    return changed;
}
