#pragma once
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <cstdint>
#include <vector>
#include "../vulkan/resources/Buffer.hpp"
#include "OctreeTopologyTypes.hpp"

class VulkanApp;

// GPU water-droplet particle simulation.
//
// Dispatches shaders/computation/water/DropletSim.comp once per substep with
// several passes (spawn / integrate / spatial-hash build / contacts+solve /
// finalize). All state lives in device-local storage buffers; the host never
// touches particle data during simulation — it only uploads the cropped octree
// SDF table used for world-space terrain collision and (optionally) reads the
// contact counters back for diagnostics.
//
// Collision geometry is the SAME SDF the terrain mesh is generated from: the
// region table carries the identical node layout (OctreeGpuNode) the topology
// pass consumes, so a droplet collides with the trilinear interpolant of the
// octree corners — no triangle mesh is involved and the rendering mode never
// changes the collision shape.
//
// The simulation is a Jacobi-style solver: every invocation reads its
// neighbours but writes only its own record, so the result is independent of
// workgroup scheduling (see the shader header for the momentum argument).
class DropletSim {
public:
    // Particle capacity of the fixed state pool. `Settings::dropletCount` is
    // clamped to this; the buffer is allocated once.
    static constexpr uint32_t kMaxParticles = 8192;
    // Spatial hash shape (mirrors the shader constants).
    static constexpr uint32_t kGridBuckets = 8192;
    static constexpr uint32_t kGridMaxPerBucket = 64;
    // Fixed simulation region cap (nodes) so a bad region can never allocate
    // unbounded memory; the region builder reports truncation instead.
    static constexpr size_t kMaxRegionNodes = 262144;

    // Per-frame simulation parameters (mirrors Settings + the shader push
    // constants; kept independent of the rendering configuration).
    struct Config {
        uint32_t particleCount = 2048;
        float spawnRadius = 128.0f;
        glm::vec3 spawnCenter{0.0f, 2000.0f, 0.0f};
        float gravity = 9.81f;
        float restitution = 0.25f;
        float friction = 0.35f;
        float damping = 0.02f;
        float waterLevel = 0.0f;
        float particleRadius = 1.5f;
        int solverIterations = 3;
        int substeps = 4;
        glm::vec3 regionMin{0.0f};
        glm::vec3 regionMax{0.0f};
    };

    // Diagnostics counters (one uvec4 in gridExtra[0]).
    struct Stats {
        uint32_t gridOverflow = 0;        // particles dropped from full buckets
        uint32_t pairContacts = 0;        // particle-particle contacts solved
        uint32_t terrainAndEscaped = 0;   // SDF contacts + particles out of region
        uint32_t maxContactsPerParticle = 0;
    };

    void init(VulkanApp* app);
    void destroy(VulkanApp* app);

    // Uploads the cropped octree node table (same layout as the topology
    // pass). Safe to call again to move/resize the simulation region.
    void uploadRegion(const std::vector<OctreeGpuNode>& nodes);
    // Seeds every particle's radius/mass (host pre-fill, one buffer copy).
    void setParticleShape(float radius, float mass);
    // Re-seeds all particles on the next dispatch (deterministic hash seed).
    void requestReset();

    // Records this frame's simulation passes into `cmd` (graphics queue, before
    // any consumer of the droplet state). `dtSeconds` is the fixed substep.
    void dispatch(VkCommandBuffer cmd, float substepSeconds, const Config& cfg);

    // Terrain surface height at (x, z) from the uploaded region table: marches
    // the interpolated SDF upward from the region floor until it turns positive
    // (outside the solid). Used to place the spawn volume just above the height
    // map. Returns the region floor when no surface is found.
    float terrainHeightAt(float x, float z) const;

    // Blocking counter readback (diagnostics only, never in the hot path).
    Stats readStats();

    // Diagnostics/tests: copies the particle state back and summarizes it
    // (vertical extent + active count) plus the first `probeCount` positions
    // and velocities. Blocking readback — never used in the normal frame path.
    struct Probe {
        float minY = 0.0f;
        float maxY = 0.0f;
        float meanY = 0.0f;
        uint32_t activeCount = 0;
        uint32_t probed = 0;
        glm::vec4 positions[8]{}; // xyz + radius
        glm::vec4 velocities[8]{};// xyz + speed
    };
    Probe probeParticles();

    bool ready() const { return initialized_ && !nodes_.empty(); }
    size_t regionNodeCount() const { return nodes_.size(); }
    VkBuffer stateBuffer() const { return state_.buffer; }

private:
    void createPipeline();
    void createBuffers();
    void ensureNodeTables(size_t nodeCount);
    void writeDescriptors();
    // One compute->compute dependency between two dispatches.
    static void computeBarrier(VkCommandBuffer cmd);

    VulkanApp* app_ = nullptr;
    bool initialized_ = false;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;

    Buffer state_{};       // Droplet[kMaxParticles], device-local
    Buffer shapeStage_{};  // host-visible radius/mass source for the init copy
    Buffer gridCounts_{};  // uint[kGridBuckets]
    Buffer gridEntries_{}; // uint[kGridBuckets * kGridMaxPerBucket]
    Buffer gridExtra_{};   // uvec4[1] counters
    Buffer statsRead_{};   // host-visible mirror for readStats()
    Buffer probeStage_{};  // host-visible mirror for probeParticles()

    // Octree node tables (host-visible coherent, same split as the topology
    // pass: sdf0/1, cube, meta, children).
    Buffer bSdf0_{}, bSdf1_{}, bCube_{}, bMeta_{}, bChild03_{}, bChild47_{};
    std::vector<OctreeGpuNode> nodes_;
    glm::vec3 regionMin_{0.0f};
    glm::vec3 regionMax_{0.0f};
    size_t nodeCap_ = 0;
    bool nodesDirty_ = false;

    uint32_t resetSeed_ = 1u;
    bool resetPending_ = true;
    bool shapeDirty_ = true;
    float shapeRadius_ = 1.5f;
    float shapeMass_ = 1.0f;
};
