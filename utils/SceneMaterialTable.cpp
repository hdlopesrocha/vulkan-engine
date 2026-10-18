#include "SceneMaterialTable.hpp"

std::vector<SceneTextureTriple> sceneTextureTriples() {
    return {
        { "textures/Wall_Stone_010_COLOR.jpg", "textures/Wall_Stone_010_NORMAL.jpg", "textures/Wall_Stone_010_HEIGHT.jpg", "textures/Wall_Stone_010_ROUGH.jpg", "textures/Wall_Stone_010_OCC.jpg" },
        { "textures/Ground_Dirt_007_COLOR.jpg", "textures/Ground_Dirt_007_NORMAL.jpg", "textures/Ground_Dirt_007_HEIGHT.jpg", "textures/Ground_Dirt_007_ROUGH.jpg", "textures/Ground_Dirt_007_OCC.jpg" },
        { "textures/Dead_leaves_001_COLOR.jpg", "textures/Dead_leaves_001_NORMAL.jpg", "textures/Dead_leaves_001_HEIGHT.jpg", "textures/Dead_leaves_001_ROUGH.jpg", "textures/Dead_leaves_001_OCC.jpg" },
        { "textures/Grass_001_COLOR.jpg", "textures/Grass_001_NORMAL.jpg", "textures/Grass_001_HEIGHT.jpg", "textures/Grass_001_ROUGH.jpg", "textures/Grass_001_OCC.jpg" },
        { "textures/Lava_005_COLOR.jpg", "textures/Lava_005_NORMAL.jpg", "textures/Lava_005_HEIGHT.jpg", "textures/Lava_005_ROUGH.jpg", "textures/Lava_005_OCC.jpg" },
        { "textures/Metal_Pattern_008_COLOR.jpg", "textures/Metal_Pattern_008_NORMAL.jpg", "textures/Metal_Pattern_008_HEIGHT.jpg", "textures/Metal_Pattern_008_ROUGH.jpg", "textures/Metal_Pattern_008_OCC.jpg" },
        { "textures/Linoleum_Floor_001_COLOR.jpg", "textures/Linoleum_Floor_001_NORMAL.jpg", "textures/Linoleum_Floor_001_HEIGHT.jpg", "textures/Linoleum_Floor_001_ROUGH.jpg", "textures/Linoleum_Floor_001_OCC.jpg" },
        { "textures/Rough_rock_021_COLOR.jpg", "textures/Rough_rock_021_NORMAL.jpg", "textures/Rough_rock_021_HEIGHT.jpg", "textures/Rough_rock_021_ROUGH.jpg", "textures/Rough_rock_021_OCC.jpg" },
        { "textures/Sand_007_COLOR.jpg", "textures/Sand_007_NORMAL.jpg", "textures/Sand_007_HEIGHT.jpg", "textures/Sand_007_ROUGH.jpg", "textures/Sand_007_OCC.jpg" },
        { "textures/Snow_001_COLOR.jpg", "textures/Snow_001_NORMAL.jpg", "textures/Snow_001_HEIGHT.jpg", "textures/Snow_001_ROUGH.jpg", "textures/Snow_001_OCC.jpg" },
        { "textures/Sand_002_COLOR.jpg", "textures/Sand_002_NORMAL.jpg", "textures/Sand_002_HEIGHT.jpg", "textures/Sand_002_ROUGH.jpg", "textures/Sand_002_OCC.jpg" },
        { "textures/Bark_001_COLOR.jpg", "textures/Bark_001_NORMAL.jpg", "textures/Bark_001_HEIGHT.jpg", "textures/Bark_001_ROUGH.jpg", "textures/Bark_001_OCC.jpg" },
        { "textures/Concrete_Blocks_013_COLOR.jpg", "textures/Concrete_Blocks_013_NORMAL.jpg", "textures/Concrete_Blocks_013_HEIGHT.jpg", "textures/Concrete_Blocks_013_ROUGH.jpg", "textures/Concrete_Blocks_013_OCC.jpg" },
        { "textures/Asphalt_001_COLOR.jpg", "textures/Asphalt_001_NORMAL.jpg", "textures/Asphalt_001_HEIGHT.jpg", "textures/Asphalt_001_ROUGH.jpg", "textures/Asphalt_001_OCC.jpg" },
        { "textures/Stone_Floor_002_COLOR.jpg", "textures/Stone_Floor_002_NORMAL.jpg", "textures/Stone_Floor_002_HEIGHT.jpg", "textures/Stone_Floor_002_ROUGH.jpg", "textures/Stone_Floor_002_OCC.jpg" },
        { "textures/Canyon_Rock_001_COLOR.jpg", "textures/Canyon_Rock_001_NORMAL.jpg", "textures/Canyon_Rock_001_HEIGHT.jpg", "textures/Canyon_Rock_001_ROUGH.jpg", "textures/Canyon_Rock_001_OCC.jpg" },
        { "textures/Sapphire_001_COLOR.jpg", "textures/Sapphire_001_NORMAL.jpg", "textures/Sapphire_001_HEIGHT.jpg", "textures/Sapphire_001_ROUGH.jpg", "textures/Sapphire_001_OCC.jpg" },
        { "textures/Rough_rock_006_COLOR.jpg", "textures/Rough_rock_006_NORMAL.jpg", "textures/Rough_rock_006_HEIGHT.jpg", "textures/Rough_rock_006_ROUGH.jpg", "textures/Rough_rock_006_OCC.jpg" },
        { "textures/Crystal_Metal_001_COLOR.jpg", "textures/Crystal_Metal_001_NORMAL.jpg", "textures/Crystal_Metal_001_HEIGHT.jpg", "textures/Crystal_Metal_001_ROUGH.jpg", "textures/Crystal_Metal_001_OCC.jpg" },
        { "textures/Sci-fi_Armor_001_COLOR.jpg", "textures/Sci-fi_Armor_001_NORMAL.jpg", "textures/Sci-fi_Armor_001_HEIGHT.jpg", "textures/Sci-fi_Armor_001_ROUGH.jpg", "textures/Sci-fi_Armor_001_OCC.jpg" },
        { "textures/Greeble_Techno_002_COLOR.jpg", "textures/Greeble_Techno_002_NORMAL.jpg", "textures/Greeble_Techno_002_HEIGHT.jpg", "textures/Greeble_Techno_002_ROUGH.jpg", "textures/Greeble_Techno_002_OCC.jpg" },
    };
}

std::vector<SceneMixerDesc> sceneMixerDescs() {
    const uint32_t firstTarget = static_cast<uint32_t>(sceneTextureTriples().size());
    std::vector<SceneMixerDesc> mixers;
    mixers.reserve(kSceneMixerCount);
    // LandBrush uses generated blend layers for the slope/height transition
    // bands (grass/sand, grass/snow, rock/grass, rock/snow, rock/sand).
    const uint32_t sources[kSceneMixerCount][2] = {
        {3u, 8u}, // grassMixSand
        {3u, 9u}, // grassMixSnow
        {7u, 3u}, // rockMixGrass
        {7u, 9u}, // rockMixSnow
        {7u, 8u}, // rockMixSand
    };
    for (uint32_t i = 0; i < kSceneMixerCount; ++i) {
        SceneMixerDesc d;
        d.target = firstTarget + i;
        d.primary = sources[i][0];
        d.secondary = sources[i][1];
        mixers.push_back(d);
    }
    return mixers;
}

std::vector<MaterialProperties> sceneMaterialProperties(size_t materialCount) {
    std::vector<MaterialProperties> materials(materialCount, MaterialProperties{});

    // Table-driven overrides: every entry shares the same mapping/triplanar/
    // invert defaults and only the listed values differ. Kept index-identical
    // to MyApp::setupTextures (this is the same table the GPU upload uses).
    struct MaterialInit {
        uint32_t index;
        float tessLevel;
        float tessMinLevel;
        float tessMaxLevel;
        float tessHeightScale;
        float triplanarScaleU;
        float triplanarScaleV;
        float reflectionStrength;
    };
    static constexpr MaterialInit kMaterialInits[] = {
        {  0u, 5.0f, 2.0f, 16.0f,  8.0f, 0.002f, 0.002f, 0.0f },
        {  5u, 5.0f, 2.0f, 16.0f,  2.0f, 0.002f, 0.002f, 0.8f },
        {  6u, 1.0f, 1.0f,  1.0f,  0.0f, 0.002f, 0.002f, 1.0f },
        {  7u, 2.0f, 2.0f, 16.0f,  8.0f, 0.01f,  0.01f,  0.2f },
        { 12u, 6.0f, 2.0f, 16.0f, 32.0f, 0.002f, 0.002f, 0.3f },
        { 14u, 5.0f, 2.0f, 16.0f,  8.0f, 0.002f, 0.002f, 0.0f },
        { 15u, 5.0f, 2.0f, 16.0f,  8.0f, 0.002f, 0.002f, 0.0f },
        { 16u, 5.0f, 2.0f, 16.0f, 32.0f, 0.002f, 0.002f, 0.0f },
        { 18u, 5.0f, 2.0f, 16.0f, 32.0f, 0.002f, 0.002f, 0.0f },
        { 20u, 5.0f, 2.0f, 16.0f, 32.0f, 0.002f, 0.002f, 0.0f },
    };
    for (const MaterialInit& mi : kMaterialInits) {
        if (mi.index >= materials.size()) continue;
        MaterialProperties& mat = materials[mi.index];
        mat.mappingMode = true;
        mat.tessLevel = mi.tessLevel;
        mat.tessMinLevel = mi.tessMinLevel;
        mat.tessMaxLevel = mi.tessMaxLevel;
        mat.tessHeightScale = mi.tessHeightScale;
        mat.triplanar = true;
        mat.triplanarScaleU = mi.triplanarScaleU;
        mat.triplanarScaleV = mi.triplanarScaleV;
        mat.invertHeight = true;
        mat.reflectionStrength = mi.reflectionStrength;
    }
    return materials;
}
