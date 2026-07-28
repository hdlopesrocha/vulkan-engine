#pragma once
#include <imgui.h>
#include <string>
#include <vector>

struct SubPage
{
    std::string label;
};

struct Page
{
    std::string label;
    std::vector<SubPage> subPages;
};

class RadialMenu
{
public:
    RadialMenu() = default;

    void SetVisible(bool visible);
    bool IsVisible() const;

    void SetCenter(ImVec2 center);
    void SetInputVector(const ImVec2& vector);
    void SetPages(const std::vector<Page>& pages);

    void Update();
    void Draw();

    int GetHoveredPage() const;
    int GetHoveredSubPage() const;

    void SetDeadZoneRadius(float radius);
    void SetInnerRadius(float radius);
    void SetOuterRadius(float radius);
    void SetRingSpacing(float spacing);
    void SetTextSize(float size);

    void SetPageHoverColor(ImU32 color);
    void SetSubpageHoverColor(ImU32 color);
    void SetBackgroundColor(ImU32 color);
    void SetOutlineColor(ImU32 color);

    // Texture ring (paginated: 14 texture slots + 2 nav buttons = 16 equal sectors)
    void SetTextureRingActive(bool active);
    bool GetTextureRingActive() const;
    void SetTextures(const std::vector<ImTextureID>& textures);
    int GetHoveredTexture() const;
    bool GetHoveredNavPrev() const;
    bool GetHoveredNavNext() const;
    void SetTextureRingRadius(float radius);
    void SetTextureSectorHoverColor(ImU32 color);
    void SetNavSectorHoverColor(ImU32 color);
    void ResetTexturePage();
    int GetTexturePage() const;
    void SetTexturePage(int page);

    // Label ring (generic text ring, e.g. control modes)
    void SetLabelRingActive(bool active);
    bool GetLabelRingActive() const;
    void SetLabelItems(const std::vector<std::string>& items);
    int GetHoveredLabel() const;
    void SetLabelRingRadius(float radius);
    void SetLabelSectorHoverColor(ImU32 color);

private:
    bool visible = false;

    ImVec2 center = ImVec2(0, 0);
    ImVec2 inputVector = ImVec2(0, 0);

    std::vector<Page> pages;

    float deadZoneRadius = 40.0f;
    float innerRadius = 60.0f;
    float outerRadius = 120.0f;
    float ringSpacing = 4.0f;
    float textSize = 14.0f;

    ImU32 pageHoverColor = IM_COL32(100, 150, 255, 200);
    ImU32 subpageHoverColor = IM_COL32(150, 200, 255, 220);
    ImU32 backgroundColor = IM_COL32(30, 30, 40, 180);
    ImU32 outlineColor = IM_COL32(200, 200, 210, 255);

    float currentAngle = 0.0f;
    float currentRadius = 0.0f;
    int hoveredPage = -1;
    int hoveredSubPage = -1;

    // Texture ring state
    bool textureRingActive = false;
    std::vector<ImTextureID> textures;
    float textureRingRadius = 170.0f;
    ImU32 textureSectorHoverColor = IM_COL32(100, 200, 100, 220);
    int hoveredTexture = -1;
    int hoveredNavPrev = false;
    int hoveredNavNext = false;
    int texturePage = 0;
    static constexpr int kTexturesPerPage = 14;
    static constexpr int kNavSectors = 2;
    static constexpr int kTotalTexSectors = kTexturesPerPage + kNavSectors;
    ImU32 navSectorHoverColor = IM_COL32(180, 180, 60, 220);

    // Label ring state
    bool labelRingActive = false;
    std::vector<std::string> labelItems;
    float labelRingRadius = 170.0f;
    ImU32 labelSectorHoverColor = IM_COL32(100, 200, 100, 220);
    int hoveredLabel = -1;

    void DrawSector(ImDrawList* drawList, float startAngle, float endAngle,
                    float innerR, float outerR, ImU32 fillColor, ImU32 outlineCol);
    void DrawLabel(ImDrawList* drawList, const std::string& label,
                   float angleStart, float angleEnd,
                   float innerR, float outerR, ImU32 color);
    void DrawArrow(ImDrawList* drawList, float angleStart, float angleEnd,
                   float innerR, float outerR, bool right, ImU32 color);
    void DrawTextureSector(ImDrawList* drawList, float startAngle, float endAngle,
                           float innerR, float outerR, ImTextureID tex,
                           bool hovered, ImU32 outlineCol);
};
