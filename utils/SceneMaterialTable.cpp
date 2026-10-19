#include "SceneMaterialTable.hpp"

std::vector<SceneTextureTriple> sceneTextureTriples() {
    return {
        { "textures/Wall_Stone_010/COLOR.jpg", "textures/Wall_Stone_010/NORMAL.jpg", "textures/Wall_Stone_010/HEIGHT.jpg", "textures/Wall_Stone_010/ROUGH.jpg", "textures/Wall_Stone_010/OCC.jpg" },
        { "textures/Ground_Dirt_007/COLOR.jpg", "textures/Ground_Dirt_007/NORMAL.jpg", "textures/Ground_Dirt_007/HEIGHT.jpg", "textures/Ground_Dirt_007/ROUGH.jpg", "textures/Ground_Dirt_007/OCC.jpg" },
        { "textures/Dead_leaves_001/COLOR.jpg", "textures/Dead_leaves_001/NORMAL.jpg", "textures/Dead_leaves_001/HEIGHT.jpg", "textures/Dead_leaves_001/ROUGH.jpg", "textures/Dead_leaves_001/OCC.jpg" },
        { "textures/Grass_001/COLOR.jpg", "textures/Grass_001/NORMAL.jpg", "textures/Grass_001/HEIGHT.jpg", "textures/Grass_001/ROUGH.jpg", "textures/Grass_001/OCC.jpg" },
        { "textures/Lava_005/COLOR.jpg", "textures/Lava_005/NORMAL.jpg", "textures/Lava_005/HEIGHT.jpg", "textures/Lava_005/ROUGH.jpg", "textures/Lava_005/OCC.jpg" },
        { "textures/Metal_Pattern_008/COLOR.jpg", "textures/Metal_Pattern_008/NORMAL.jpg", "textures/Metal_Pattern_008/HEIGHT.jpg", "textures/Metal_Pattern_008/ROUGH.jpg", "textures/Metal_Pattern_008/OCC.jpg" },
        { "textures/Linoleum_Floor_001/COLOR.jpg", "textures/Linoleum_Floor_001/NORMAL.jpg", "textures/Linoleum_Floor_001/HEIGHT.jpg", "textures/Linoleum_Floor_001/ROUGH.jpg", "textures/Linoleum_Floor_001/OCC.jpg" },
        { "textures/Rough_rock_021/COLOR.jpg", "textures/Rough_rock_021/NORMAL.jpg", "textures/Rough_rock_021/HEIGHT.jpg", "textures/Rough_rock_021/ROUGH.jpg", "textures/Rough_rock_021/OCC.jpg" },
        { "textures/Sand_007/COLOR.jpg", "textures/Sand_007/NORMAL.jpg", "textures/Sand_007/HEIGHT.jpg", "textures/Sand_007/ROUGH.jpg", "textures/Sand_007/OCC.jpg" },
        { "textures/Snow_001/COLOR.jpg", "textures/Snow_001/NORMAL.jpg", "textures/Snow_001/HEIGHT.jpg", "textures/Snow_001/ROUGH.jpg", "textures/Snow_001/OCC.jpg" },
        { "textures/Sand_002/COLOR.jpg", "textures/Sand_002/NORMAL.jpg", "textures/Sand_002/HEIGHT.jpg", "textures/Sand_002/ROUGH.jpg", "textures/Sand_002/OCC.jpg" },
        { "textures/Bark_001/COLOR.jpg", "textures/Bark_001/NORMAL.jpg", "textures/Bark_001/HEIGHT.jpg", "textures/Bark_001/ROUGH.jpg", "textures/Bark_001/OCC.jpg" },
        { "textures/Concrete_Blocks_013/COLOR.jpg", "textures/Concrete_Blocks_013/NORMAL.jpg", "textures/Concrete_Blocks_013/HEIGHT.jpg", "textures/Concrete_Blocks_013/ROUGH.jpg", "textures/Concrete_Blocks_013/OCC.jpg" },
        { "textures/Asphalt_001/COLOR.jpg", "textures/Asphalt_001/NORMAL.jpg", "textures/Asphalt_001/HEIGHT.jpg", "textures/Asphalt_001/ROUGH.jpg", "textures/Asphalt_001/OCC.jpg" },
        { "textures/Stone_Floor_002/COLOR.jpg", "textures/Stone_Floor_002/NORMAL.jpg", "textures/Stone_Floor_002/HEIGHT.jpg", "textures/Stone_Floor_002/ROUGH.jpg", "textures/Stone_Floor_002/OCC.jpg" },
        { "textures/Canyon_Rock_001/COLOR.jpg", "textures/Canyon_Rock_001/NORMAL.jpg", "textures/Canyon_Rock_001/HEIGHT.jpg", "textures/Canyon_Rock_001/ROUGH.jpg", "textures/Canyon_Rock_001/OCC.jpg" },
        { "textures/Sapphire_001/COLOR.jpg", "textures/Sapphire_001/NORMAL.jpg", "textures/Sapphire_001/HEIGHT.jpg", "textures/Sapphire_001/ROUGH.jpg", "textures/Sapphire_001/OCC.jpg" },
        { "textures/Rough_rock_006/COLOR.jpg", "textures/Rough_rock_006/NORMAL.jpg", "textures/Rough_rock_006/HEIGHT.jpg", "textures/Rough_rock_006/ROUGH.jpg", "textures/Rough_rock_006/OCC.jpg" },
        { "textures/Crystal_Metal_001/COLOR.jpg", "textures/Crystal_Metal_001/NORMAL.jpg", "textures/Crystal_Metal_001/HEIGHT.jpg", "textures/Crystal_Metal_001/ROUGH.jpg", "textures/Crystal_Metal_001/OCC.jpg" },
        { "textures/Sci-fi_Armor_001/COLOR.jpg", "textures/Sci-fi_Armor_001/NORMAL.jpg", "textures/Sci-fi_Armor_001/HEIGHT.jpg", "textures/Sci-fi_Armor_001/ROUGH.jpg", "textures/Sci-fi_Armor_001/OCC.jpg" },
        { "textures/Greeble_Techno_002/COLOR.jpg", "textures/Greeble_Techno_002/NORMAL.jpg", "textures/Greeble_Techno_002/HEIGHT.jpg", "textures/Greeble_Techno_002/ROUGH.jpg", "textures/Greeble_Techno_002/OCC.jpg" },
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
