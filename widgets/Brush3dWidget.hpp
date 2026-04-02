#pragma once

#include "Widget.hpp"
#include "../sdf/SdfType.hpp"
#include "../math/BrushMode.hpp"
#include "../utils/Scene.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <functional>

class TextureArrayManager;
class SceneRenderer;
class VulkanApp;
class LocalScene;

// Represents one SDF brush entry with all parameters needed to populate the scene
struct BrushEntry {
    // SDF primitive: 0=Sphere,1=Box,2=Capsule,3=Octahedron,4=Pyramid,5=Torus,6=Cone,7=Cylinder
    int sdfType = 0;
    // Operation: 0=ADD, 1=REMOVE
    int brushMode = 0;
    // Target layer: 0=OPAQUE, 1=TRANSPARENT
    int targetLayer = 0;
    // Material (texture index for SimpleBrush)
    int materialIndex = 0;
    // Transform
    glm::vec3 scale = glm::vec3(128.0f);
    glm::vec3 translate = glm::vec3(0.0f);
    float yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
    // Octree min size
    float minSize = 30.0f;
    // Optional SDF effect
    bool useEffect = false;
    // Effect type: 0=PerlinDistort, 1=PerlinCarve, 2=SineDistort, 3=VoronoiCarve
    int effectType = 0;
    float effectAmplitude = 48.0f;
    float effectFrequency = 0.003f;
    float effectThreshold = 0.1f;   // PerlinCarve
    float effectCellSize = 64.0f;   // Voronoi
    float effectBrightness = 0.0f;
    float effectContrast = 1.0f;
    // Capsule-specific
    glm::vec3 capsuleA = glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 capsuleB = glm::vec3(0.0f, 1.0f, 0.0f);
    float capsuleRadius = 0.5f;
    // Torus-specific
    glm::vec2 torusRadii = glm::vec2(0.5f, 0.25f);
};

class Brush3dWidget : public Widget {
public:
    using RebuildCallback = std::function<void()>;

    Brush3dWidget(TextureArrayManager* texMgr, uint32_t loadedLayers);

    void render() override;

    void setRebuildCallback(RebuildCallback cb) { rebuildCallback = cb; }

    const std::vector<BrushEntry>& getEntries() const { return entries; }
    bool isDirty() const { return dirty; }
    void clearDirty() { dirty = false; }

private:
    void renderEntry(int index);
    void renderMaterialPicker(BrushEntry& entry);

    std::vector<BrushEntry> entries;
    TextureArrayManager* textureArrayManager;
    uint32_t loadedTextureLayers;
    bool dirty = false;
    RebuildCallback rebuildCallback;

    static const char* sdfTypeNames[];
    static const char* brushModeNames[];
    static const char* layerNames[];
    static const char* effectTypeNames[];
};
