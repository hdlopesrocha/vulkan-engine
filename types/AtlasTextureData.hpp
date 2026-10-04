#pragma once

struct AtlasTextureData {
    unsigned char* albedoData = nullptr;
    unsigned char* normalData = nullptr;
    unsigned char* opacityData = nullptr;
    int width = 0;
    int height = 0;
};
