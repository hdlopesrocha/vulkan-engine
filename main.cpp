// Standard library includes first
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <stdexcept>
#include <mutex>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "Uniforms.hpp"
#include "vulkan/VulkanApp.hpp"
#include "vulkan/SceneRenderer.hpp"
#include "utils/LocalScene.hpp"
#include "widgets/SettingsWidget.hpp"
#include "widgets/SkyWidget.hpp"
#include "widgets/WaterWidget.hpp"
#include "widgets/RenderPassDebugWidget.hpp"
#include "widgets/BillboardWidget.hpp"
#include "utils/MainSceneLoader.hpp"
// ...existing includes...
#include "widgets/BillboardWidgetManager.hpp"
#include "widgets/WidgetManager.hpp"
#include "math/Camera.hpp"
#include "math/Light.hpp"
#include "events/EventManager.hpp"
#include "events/KeyboardPublisher.hpp"
#include "events/CloseWindowEvent.hpp"
#include "events/ToggleFullscreenEvent.hpp"
#include "vulkan/TextureArrayManager.hpp"
#include "vulkan/MaterialManager.hpp"

class MyApp : public VulkanApp, public IEventHandler {
public:
    std::unique_ptr<SceneRenderer> sceneRenderer;
    std::unique_ptr<LocalScene> mainScene;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    static constexpr uint32_t QUERY_COUNT = 12;
    float timestampPeriod = 0.0f;
    bool profilingEnabled = true;
    float profileShadowCull = 0.0f;
    float profileShadowDraw = 0.0f;
    float profileMainCull = 0.0f;
    float profileDepthPrepass = 0.0f;
    float profileMainDraw = 0.0f;
    float profileSky = 0.0f;
    float profileImGui = 0.0f;
    float profileWater = 0.0f;
    float profileCpuUpdate = 0.0f;
    float profileCpuRecord = 0.0f;
    VkDescriptorSet shadowPassDescriptorSet = VK_NULL_HANDLE;
    UniformObject uboStatic = {};
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::shared_ptr<SettingsWidget> settingsWidget;
    std::shared_ptr<SkyWidget> skyWidget;
    std::shared_ptr<WaterWidget> waterWidget;
    std::shared_ptr<RenderPassDebugWidget> renderPassDebugWidget;
    std::shared_ptr<BillboardWidget> billboardWidget;
    std::unique_ptr<BillboardWidgetManager> billboardWidgetManager;
    WidgetManager widgetManager;
    // Camera at original working position
    // FIXED: Camera was at Z=-750 looking at -Z, but terrain is at +Z (behind camera!)
    // Now positioned at Z=2000 looking at -Z to see terrain at Z=795
    Camera camera = Camera(glm::vec3(3456, 915, 2000), Math::eulerToQuat(0, 0, 0));
    Light light = Light(glm::vec3(0.0f, -1.0f, 0.0f));
    EventManager eventManager;
    KeyboardPublisher keyboardPublisher;
    bool sceneLoading = false;  // True during initial loadScene, false after (enables incremental uploads)

    MyApp()
        : sceneRenderer(std::make_unique<SceneRenderer>(this)),
          mainScene(std::make_unique<LocalScene>())
    {}

    void setup() override {
        // Restore previous scene loading logic
        settingsWidget = std::make_shared<SettingsWidget>();
        skyWidget = std::make_shared<SkyWidget>();
        // Bind water widget to the SceneRenderer's WaterRenderer so it edits the canonical params
        waterWidget = std::make_shared<WaterWidget>(sceneRenderer->waterRenderer.get());
        renderPassDebugWidget = std::make_shared<RenderPassDebugWidget>(this, nullptr);
        billboardWidget = std::make_shared<BillboardWidget>();
        billboardWidgetManager = std::make_unique<BillboardWidgetManager>(billboardWidget, nullptr, nullptr);
        widgetManager.addWidget(settingsWidget);
        widgetManager.addWidget(skyWidget);
        widgetManager.addWidget(waterWidget);
        widgetManager.addWidget(renderPassDebugWidget);
        widgetManager.addWidget(billboardWidget);
        // Scene loading
        MainSceneLoader mainSceneLoader(&mainScene->transparentLayerChangeHandler, &mainScene->opaqueLayerChangeHandler);
        
        // Initialize scene renderer (creates pipelines, descriptor sets, water targets, etc.)
        if (sceneRenderer) {
            sceneRenderer->init(this, skyWidget.get());

            // Helper to wire up change handler callbacks for a layer
            // During initial load (sceneLoading=true), just accumulate to CPU arrays
            // After initial load, use uploadMesh() for incremental GPU updates
            auto wireUpChangeHandler = [this](auto& changeHandler, Layer layer, IndirectRenderer& indirectRenderer, const char* name, bool& loadingFlag) {
                changeHandler.setOnNodeUpdated([this, layer, &indirectRenderer, name, &loadingFlag](const OctreeNodeData& nodeData) {
                    uint32_t nodeId = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(nodeData.node) & 0xFFFFFFFF);
                    
                    mainScene->requestModel3D(layer, const_cast<OctreeNodeData&>(nodeData), [this, &indirectRenderer, nodeId, name, &loadingFlag](const Geometry& geom) {
                        if (!geom.vertices.empty() && !geom.indices.empty()) {
                            indirectRenderer.addMesh(this, geom, glm::mat4(1.0f), nodeId);
                            if (!loadingFlag) {
                                // Runtime change: try incremental upload, fall back to rebuild if needed
                                if (!indirectRenderer.uploadMesh(this, nodeId)) {
                                    printf("[%s] uploadMesh failed for node %u, triggering rebuild\n", name, nodeId);
                                    indirectRenderer.rebuild(this);
                                }
                            }
                        }
                    });
                });
                
                changeHandler.setOnNodeErased([this, &indirectRenderer, name, &loadingFlag](const OctreeNodeData& nodeData) {
                    uint32_t nodeId = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(nodeData.node) & 0xFFFFFFFF);
                    if (!loadingFlag) {
                        // Runtime removal: zero out GPU command before removing from CPU
                        indirectRenderer.eraseMeshFromGPU(this, nodeId);
                    }
                    indirectRenderer.removeMesh(nodeId);
                });
            };

            // Wire up opaque layer -> SolidRenderer's IndirectRenderer
            if (sceneRenderer->solidRenderer) {
                wireUpChangeHandler(mainScene->opaqueLayerChangeHandler, Layer::LAYER_OPAQUE,
                                    sceneRenderer->solidRenderer->getIndirectRenderer(), "OpaqueChangeHandler", sceneLoading);
            }
            
            // Wire up transparent layer -> WaterRenderer's IndirectRenderer
            if (sceneRenderer->waterRenderer) {
                wireUpChangeHandler(mainScene->transparentLayerChangeHandler, Layer::LAYER_TRANSPARENT,
                                    sceneRenderer->waterRenderer->getIndirectRenderer(), "TransparentChangeHandler", sceneLoading);
            }
        }
        
        // Load scene data into octrees (this populates the octree structure)
        // Change handlers are called automatically by octree add/del and will build meshes
        sceneLoading = true;
        mainScene->loadScene(mainSceneLoader);
        sceneLoading = false;
        
        // Batch rebuild: upload all accumulated meshes to GPU in one call per renderer
        if (sceneRenderer) {
            if (sceneRenderer->solidRenderer) {
                sceneRenderer->solidRenderer->getIndirectRenderer().rebuild(this);
                printf("[MyApp::setup] Rebuilt opaque IndirectRenderer\n");
            }
            if (sceneRenderer->waterRenderer) {
                sceneRenderer->waterRenderer->getIndirectRenderer().rebuild(this);
                printf("[MyApp::setup] Rebuilt water IndirectRenderer\n");
            }
        }
        
        // Subscribe event handlers
        eventManager.subscribe(&camera);  // Camera handles translate/rotate events
        eventManager.subscribe(this);     // MyApp handles close/fullscreen events
        
        // Set up camera projection matrix
        float aspectRatio = static_cast<float>(getWidth()) / static_cast<float>(getHeight());
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspectRatio, 0.1f, 10000.0f);
        proj[1][1] *= -1; // Vulkan Y-flip
        camera.setProjection(proj);
        
        // Position camera to view the terrain
        printf("[Camera Setup] Final Position: (%.1f, %.1f, %.1f)\n", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
        printf("[Camera Setup] Forward: (%.3f, %.3f, %.3f)\n", camera.getForward().x, camera.getForward().y, camera.getForward().z);
    }

    void update(float deltaTime) override {
        // Poll keyboard input and publish events
        keyboardPublisher.update(getWindow(), &eventManager, camera, deltaTime, false);
        eventManager.processQueued();
        
        // Advance water animation time (owned by WaterRenderer)
        if (sceneRenderer && sceneRenderer->waterRenderer) {
            sceneRenderer->waterRenderer->advanceTime(deltaTime);
        }
    }

    void preRenderPass(VkCommandBuffer &commandBuffer) override {
        // Shadow pass (uses separate command buffer internally)
        sceneRenderer->shadowPass(commandBuffer, queryPool, shadowPassDescriptorSet, uboStatic, true, false);
        
        // Run GPU frustum culling for opaque geometry
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        if (sceneRenderer && sceneRenderer->solidRenderer) {
            sceneRenderer->solidRenderer->getIndirectRenderer().prepareCull(commandBuffer, viewProj);
        }
    }

    void renderImGui() override {
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit")) requestClose();
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Demo", NULL, &imguiShowDemo);
                ImGui::MenuItem("Show Profiling", NULL, &profilingEnabled);
                if (ImGui::MenuItem("Fullscreen", "F11", isFullscreen)) {
                    toggleFullscreen();
                }
                ImGui::EndMenu();
            }
            // Widget menu
            widgetManager.renderMenu();
            ImGui::EndMainMenuBar();

            // Small top-left overlay under the main menu bar showing FPS and visible count
            {
                ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
                ImGui::SetNextWindowBgAlpha(0.35f);
                float padding = 10.0f;
                float y = ImGui::GetFrameHeight() + 6.0f; // position just under the main menu bar
                ImGui::SetNextWindowPos(ImVec2(padding, y), ImGuiCond_Always);
                if (ImGui::Begin("##stats_overlay", nullptr, flags)) {
                    ImGui::Text("FPS: %.1f (%.2f ms)", imguiFps, imguiFps > 0 ? 1000.0f / imguiFps : 0.0f);
                    ImGui::Text("Loaded: %zu meshes", sceneRenderer && sceneRenderer->solidRenderer ? sceneRenderer->solidRenderer->getRegisteredModelCount() : 0);
                    
                    if (profilingEnabled) {
                        ImGui::Separator();
                        ImGui::Text("--- GPU Timing (ms) ---");
                        float gpuTotal = profileShadowCull + profileShadowDraw + profileMainCull +
                                         profileDepthPrepass + profileMainDraw + profileSky + profileImGui;
                        ImGui::Text("Shadow Cull:   %.2f", profileShadowCull);
                        ImGui::Text("Shadow Draw:   %.2f", profileShadowDraw);
                        ImGui::Text("Main Cull:     %.2f", profileMainCull);
                        ImGui::Text("Depth Prepass: %.2f", profileDepthPrepass);
                        ImGui::Text("Main Draw:     %.2f", profileMainDraw);
                        ImGui::Text("Sky:           %.2f", profileSky);
                        ImGui::Text("ImGui:         %.2f", profileImGui);
                        ImGui::Text("GPU Total:     %.2f", gpuTotal);
                        ImGui::Separator();
                        ImGui::Text("--- CPU Timing (ms) ---");
                        ImGui::Text("Update:        %.2f", profileCpuUpdate);
                        ImGui::Text("Record:        %.2f", profileCpuRecord);
                    }
                }
                ImGui::End();
            }
        }

        if (imguiShowDemo) ImGui::ShowDemoWindow(&imguiShowDemo);

        // Render all widgets
        widgetManager.renderAll();
    }

    void draw(VkCommandBuffer &commandBuffer, VkRenderPassBeginInfo &renderPassInfo) override {
        // Only record draw commands; command buffer and render pass are already active
        if (commandBuffer == VK_NULL_HANDLE) {
            fprintf(stderr, "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE, skipping draw.\n");
            return;
        }
        if (!sceneRenderer) {
            fprintf(stderr, "[MyApp::draw] Error: sceneRenderer is nullptr, skipping draw.\n");
            return;
        }
        if (!mainScene) {
            fprintf(stderr, "[MyApp::draw] Error: mainScene is nullptr, skipping draw.\n");
            return;
        }
        
        // Update UBO with current camera and light data
        glm::mat4 viewProj = camera.getViewProjectionMatrix();
        uboStatic.viewProjection = viewProj;
        uboStatic.viewPos = glm::vec4(camera.getPosition(), 1.0f);
        uboStatic.lightDir = glm::vec4(light.getDirection(), 0.0f);
        uboStatic.lightColor = glm::vec4(1.0f, 1.0f, 0.9f, 1.0f);
        uboStatic.lightSpaceMatrix = light.getViewProjectionMatrix();
        
        static int frameCount = 0;
        if (frameCount++ < 3) {
            printf("[Frame %d] Camera pos=(%.1f, %.1f, %.1f) forward=(%.3f, %.3f, %.3f)\n", 
                   frameCount, camera.getPosition().x, camera.getPosition().y, camera.getPosition().z,
                   camera.getForward().x, camera.getForward().y, camera.getForward().z);
            printf("[Frame %d] Uploading UBO: buffer=%p, size=%zu, mainDescriptorSet=%p\n", 
                   frameCount, (void*)sceneRenderer->mainUniformBuffer.buffer, sizeof(UniformObject),
                   (void*)getMainDescriptorSet());
        }
        
        // Upload UBO to GPU
        void* data;
        vkMapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory, 0, sizeof(UniformObject), 0, &data);
        memcpy(data, &uboStatic, sizeof(UniformObject));
        vkUnmapMemory(getDevice(), sceneRenderer->mainUniformBuffer.memory);
        
        // Main pass (inside render pass - sky, solid, vegetation)
        sceneRenderer->mainPass(commandBuffer, renderPassInfo, 0, true, getMainDescriptorSet(), sceneRenderer->mainUniformBuffer, false, profilingEnabled, queryPool,
            viewProj, uboStatic, sceneRenderer->waterRenderer->getParams(), sceneRenderer->waterRenderer->getTime(), true, false, true, 0, 0.0f, 0.0f);
        // Water pass
        sceneRenderer->waterPass(commandBuffer, renderPassInfo, 0, getMainDescriptorSet(), profilingEnabled, queryPool, sceneRenderer->waterRenderer->getParams(), sceneRenderer->waterRenderer->getTime());
        //fprintf(stderr, "[MyApp::draw] waterPass returned. Rendering ImGui...\n");
        // ImGui rendering
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (!draw_data) {
            fprintf(stderr, "[MyApp::draw] Warning: ImGui::GetDrawData() returned nullptr, skipping ImGui rendering.\n");
        } else if (commandBuffer == VK_NULL_HANDLE) {
            fprintf(stderr, "[MyApp::draw] Error: commandBuffer is VK_NULL_HANDLE before ImGui rendering, skipping ImGui.\n");
        } else {
            //fprintf(stderr, "[MyApp::draw] ImGui::GetDrawData() valid, calling ImGui_ImplVulkan_RenderDrawData...\n");
            ImGui_ImplVulkan_RenderDrawData(draw_data, commandBuffer);
        }
    }

    void clean() override {
        // Cleanup scene renderer and all sub-renderers
        if (sceneRenderer) {
            sceneRenderer->cleanup();
        }
    }

    void onEvent(const EventPtr &event) override {
        if (auto closeEvent = std::dynamic_pointer_cast<CloseWindowEvent>(event)) {
            requestClose();
        } else if (auto fullscreenEvent = std::dynamic_pointer_cast<ToggleFullscreenEvent>(event)) {
            toggleFullscreen();
        }
    }
};

int main(int argc, char** argv) {
    try {
        MyApp app;
        app.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}


