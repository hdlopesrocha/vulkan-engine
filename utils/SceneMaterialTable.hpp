#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../types/MaterialProperties.hpp"
#include "SceneTextureTriple.hpp"
#include "SceneMixerDesc.hpp"

// The engine's built-in material table, shared by the Vulkan setup
// (MyApp::setupTextures) and offline exporters so both map a vertex brush /
// texture index to the same files and shading parameters.
//
// Vulkan-free on purpose: the Blender exporter (utils/BlenderExport) and the
// headless server use it without linking the renderer.
//
// Index conventions match the texture array layers exactly:
//   [0, triples.size())       source triples, one image set per entry
//   [triples.size(), +5)      generated mixer layers (Perlin blends)
//   next two                  editable paint placeholder + spare

constexpr uint32_t kSceneMixerCount = 5;
constexpr uint32_t kScenePaintMargin = 2;  // editable placeholder + spare

// The 21 hard-coded texture triples, in layer order (index = vector position).
std::vector<SceneTextureTriple> sceneTextureTriples();

// The five generated blend layers; targets continue after the source triples.
std::vector<SceneMixerDesc> sceneMixerDescs();

// Default MaterialProperties for `materialCount` layers, with the same
// per-layer overrides MyApp::setupTextures installs.
std::vector<MaterialProperties> sceneMaterialProperties(size_t materialCount);
