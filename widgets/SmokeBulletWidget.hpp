#pragma once

#include "Widget.hpp"
#include "../vulkan/renderer/SdfRenderer.hpp"

class Camera;

// Counter-Strike-style smoke bomb with bullet interaction, rendered by the
// generic SDF volume renderer. All tuning lives in SdfEffectConfig owned by
// the renderer; the controls below bind per-frame locals initialized from
// it and commit through the renderer setters (clamp + upload/rebuild
// routing), so renderer and widget share one copy.
class SmokeBulletWidget : public Widget {
public:
    SmokeBulletWidget(SdfRenderer* sdf, Camera* cam);
    void render() override;

private:
    SdfRenderer* sdfRenderer = nullptr;
    Camera* camera = nullptr;
};
