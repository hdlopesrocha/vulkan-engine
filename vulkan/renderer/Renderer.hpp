#pragma once

class VulkanApp;
struct CommandBufferState;

// Common lifecycle interface implemented by every renderer in the engine.
//
// The concrete renderers (SolidRenderer, WaterRenderer, ShadowRenderer, ...)
// keep their own, specialised drawing APIs — the interface only captures the
// two lifecycle hooks every renderer shares, so owners (SceneRenderer) can
// drive init/teardown wiring uniformly:
//
//   - cleanup(VulkanApp*)  — release GPU resources while the device is still
//     valid. Must be called exactly once during teardown (never in a
//     destructor, unless passing nullptr is safe for that renderer).
//   - setCmdState(...)     — attach the shared per-frame command buffer state
//     tracker used to deduplicate vkCmdBindPipeline calls. Renderers that
//     record from a worker thread (e.g. the async back-face task) must keep
//     cmdState = nullptr to avoid a data race on the shared tracker.
class Renderer {
public:
    virtual ~Renderer() = default;

    virtual void cleanup(VulkanApp* app) = 0;

    // Attach the shared per-frame command buffer state tracker. Default
    // implementation stores the pointer in `cmdState`; renderers that own
    // sub-renderers override this to cascade to them as well.
    virtual void setCmdState(CommandBufferState* state) { cmdState = state; }

protected:
    // Shared per-frame pipeline-bind tracker, consumed by the renderer's
    // command recording. nullptr until setCmdState() is called (and
    // deliberately never set for renderers used by the async back-face task).
    CommandBufferState* cmdState = nullptr;
};
