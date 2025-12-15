// Implementation for VegetationAtlasEditor
#include "VegetationAtlasEditor.hpp"

VegetationAtlasEditor::VegetationAtlasEditor(TextureManager* vegTextureManager, AtlasManager* atlasManager)
    : Widget("Vegetation Atlas Editor"), vegetationTextureManager(vegTextureManager), atlasManager(atlasManager) {
    isOpen = false; // Start closed to avoid crashes on startup
}

VegetationAtlasEditor::~VegetationAtlasEditor() {
    // Cleanup handled elsewhere
}

void VegetationAtlasEditor::render() {
    if (!isOpen) return;
    if (!vegetationTextureManager || !atlasManager) return; // Safety check

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    // Texture selection
    ImGui::Text("Select Vegetation Texture:");
    const char* textureNames[] = { "Foliage", "Grass", "Wild" };
    if (ImGui::Combo("Texture", &currentTextureIndex, textureNames, 3)) {
        // Texture changed - reset selected tile
        selectedTileIndex = -1;
    }

    ImGui::Separator();

    // Left panel - Tile list and controls
    ImGui::BeginChild("TileList", ImVec2(250, 0), true);

    ImGui::Text("Atlas Tiles");
    ImGui::Separator();

    // Add new tile button
    if (ImGui::Button("Add New Tile", ImVec2(-1, 0))) {
        AtlasTile newTile;
        newTile.name = "Tile " + std::to_string(atlasManager->getTileCount(currentTextureIndex) + 1);
        newTile.offsetX = 0.0f;
        newTile.offsetY = 0.0f;
        newTile.scaleX = 0.25f;  // Default to 1/4 of texture
        newTile.scaleY = 0.25f;
        atlasManager->addTile(currentTextureIndex, newTile);
        selectedTileIndex = atlasManager->getTileCount(currentTextureIndex) - 1;
    }

    ImGui::Spacing();

    // List of tiles
    for (size_t i = 0; i < atlasManager->getTileCount(currentTextureIndex); ++i) {
        ImGui::PushID(i);

        const AtlasTile* tile = atlasManager->getTile(currentTextureIndex, i);
        if (!tile) {
            ImGui::PopID();
            continue;
        }

        bool isSelected = (selectedTileIndex == static_cast<int>(i));
        if (ImGui::Selectable(tile->name.c_str(), isSelected)) {
            selectedTileIndex = i;
        }

        // Right-click context menu
        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Delete")) {
                atlasManager->removeTile(currentTextureIndex, i);
                if (selectedTileIndex >= static_cast<int>(atlasManager->getTileCount(currentTextureIndex))) {
                    selectedTileIndex = atlasManager->getTileCount(currentTextureIndex) - 1;
                }
                ImGui::EndPopup();
                ImGui::PopID();
                break;
            }
            ImGui::EndPopup();
        }

        ImGui::PopID();
    }

    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel - Tile editor and preview
    ImGui::BeginChild("TileEditor", ImVec2(0, 0), true);

    if (selectedTileIndex >= 0 && selectedTileIndex < static_cast<int>(atlasManager->getTileCount(currentTextureIndex))) {
        AtlasTile* tile = atlasManager->getTile(currentTextureIndex, selectedTileIndex);
        if (!tile) {
            ImGui::Text("Error: Invalid tile");
            ImGui::EndChild();
            ImGui::End();
            return;
        }

        ImGui::Text("Edit Tile: %s", tile->name.c_str());
        ImGui::SameLine();

        // Delete button (red color)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
        if (ImGui::Button("Delete Tile")) {
            atlasManager->removeTile(currentTextureIndex, selectedTileIndex);
            selectedTileIndex = -1; // Deselect after deletion
            ImGui::PopStyleColor(3);
            ImGui::EndChild();
            ImGui::End();
            return;
        }
        ImGui::PopStyleColor(3);

        ImGui::Separator();

        // Tile name
        char nameBuffer[128];
        strncpy(nameBuffer, tile->name.c_str(), sizeof(nameBuffer) - 1);
        nameBuffer[sizeof(nameBuffer) - 1] = '\0';
        if (ImGui::InputText("Name", nameBuffer, sizeof(nameBuffer))) {
            tile->name = nameBuffer;
        }

        ImGui::Spacing();

        // UV offset controls
        ImGui::Text("Position (UV Offset):");
        ImGui::SliderFloat("Offset X", &tile->offsetX, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Offset Y", &tile->offsetY, 0.0f, 1.0f, "%.3f");

        ImGui::Spacing();

        // UV scale controls
        ImGui::Text("Size (UV Scale):");
        ImGui::SliderFloat("Scale X", &tile->scaleX, 0.01f, 1.0f, "%.3f");
        ImGui::SliderFloat("Scale Y", &tile->scaleY, 0.01f, 1.0f, "%.3f");

        ImGui::Spacing();
        ImGui::Separator();

        // Quick presets
        ImGui::Text("Quick Presets:");
        if (ImGui::Button("1/4 Size (2x2 grid)")) {
            tile->scaleX = 0.5f;
            tile->scaleY = 0.5f;
        }
        ImGui::SameLine();
        if (ImGui::Button("1/9 Size (3x3 grid)")) {
            tile->scaleX = 0.333f;
            tile->scaleY = 0.333f;
        }
        ImGui::SameLine();
        if (ImGui::Button("1/16 Size (4x4 grid)")) {
            tile->scaleX = 0.25f;
            tile->scaleY = 0.25f;
        }

        if (ImGui::Button("Full Texture")) {
            tile->offsetX = 0.0f;
            tile->offsetY = 0.0f;
            tile->scaleX = 1.0f;
            tile->scaleY = 1.0f;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Texture view tabs
        ImGui::Text("Texture View:");
        if (ImGui::BeginTabBar("TextureTabs")) {
            if (ImGui::BeginTabItem("Albedo")) {
                currentTextureView = 0;
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Normal")) {
                currentTextureView = 1;
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Opacity")) {
                currentTextureView = 2;
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        ImGui::Spacing();

        // Preview area with actual texture
        if (vegetationTextureManager && currentTextureIndex < (int)vegetationTextureManager->count()) {
            ImTextureID texID = nullptr;
            try {
                texID = vegetationTextureManager->getImTexture((size_t)currentTextureIndex, currentTextureView);
            } catch (...) {
                // Silently catch any exceptions during texture ID retrieval
            }

            if (texID) {
                float previewSize = 256.0f;
                ImVec2 cursorPos = ImGui::GetCursorScreenPos();

                // Draw the texture using the safe ImTextureID from TextureManager
                ImGui::Image(texID, ImVec2(previewSize, previewSize));

                // Draw overlay showing selected tile region
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                ImVec2 rectMin(
                    cursorPos.x + tile->offsetX * previewSize,
                    cursorPos.y + tile->offsetY * previewSize
                );
                ImVec2 rectMax(
                    cursorPos.x + (tile->offsetX + tile->scaleX) * previewSize,
                    cursorPos.y + (tile->offsetY + tile->scaleY) * previewSize
                );

                // Draw yellow border around selected tile
                drawList->AddRect(rectMin, rectMax, IM_COL32(255, 255, 0, 255), 0.0f, 0, 3.0f);

                // Draw semi-transparent overlay on non-selected areas
                drawList->AddRectFilled(
                    ImVec2(cursorPos.x, cursorPos.y),
                    ImVec2(cursorPos.x + previewSize, rectMin.y),
                    IM_COL32(0, 0, 0, 128)
                );
                drawList->AddRectFilled(
                    ImVec2(cursorPos.x, rectMax.y),
                    ImVec2(cursorPos.x + previewSize, cursorPos.y + previewSize),
                    IM_COL32(0, 0, 0, 128)
                );
                drawList->AddRectFilled(
                    ImVec2(cursorPos.x, rectMin.y),
                    ImVec2(rectMin.x, rectMax.y),
                    IM_COL32(0, 0, 0, 128)
                );
                drawList->AddRectFilled(
                    ImVec2(rectMax.x, rectMin.y),
                    ImVec2(cursorPos.x + previewSize, rectMax.y),
                    IM_COL32(0, 0, 0, 128)
                );

                ImGui::Spacing();
                ImGui::Text("UV Rect: (%.3f, %.3f) to (%.3f, %.3f)", 
                    tile->offsetX, tile->offsetY, 
                    tile->offsetX + tile->scaleX, tile->offsetY + tile->scaleY);
            }
        }

        ImGui::Spacing();

    } else {
        ImGui::Text("Select or create a tile to edit");
    }

    ImGui::EndChild();

    ImGui::End();
}
#include "VegetationAtlasEditor.hpp"

// Auto-generated implementation stub for VegetationAtlasEditor

// Minimal no-op implementation to keep translation unit for build.
