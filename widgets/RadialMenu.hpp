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

    // Page/subpage hover
    int GetHoveredPage() const;
    int GetHoveredSubPage() const;

    // Push a subpage ring for the given page index
    void PushSubpageRing(int pageIndex);
    // Push an extra ring on top of the subpage ring
    void PushTextureRing(const std::vector<ImTextureID>& textures);
    void PushLabelRing(const std::vector<std::string>& items);
    void PushHSVSliderRing(const std::string& name, float value, float minVal, float maxVal);
    // Set which item is selected in the current top ring (for ghost highlighting)
    void SetSelectedIndex(int index);
    // Pop the last ring (back navigation)
    void PopRing();
    // Clear all rings
    void ClearRings();

    int GetStackDepth() const;

    // Query the active (top) ring
    enum class RingType { NONE, SUBPAGE, TEXTURE, LABEL, HSV_SLIDER };
    RingType GetActiveRingType() const;

    // Get the page index from the first SUBPAGE ring in the stack (-1 if none)
    int GetStackPageIndex() const;

    // Texture ring queries (active ring must be TEXTURE)
    void SetTextures(const std::vector<ImTextureID>& textures);
    int GetHoveredTexture() const;
    bool GetHoveredNavPrev() const;
    bool GetHoveredNavNext() const;
    void SetTexturePage(int page);
    int GetTexturePage() const;
    void ResetTexturePage();

    // Label ring queries (active ring must be LABEL)
    int GetHoveredLabel() const;
    void SetCurrentItem(int index);

    // HSV slider queries (active ring must be HSV_SLIDER)
    void SetHSVSliderValue(float value);
    float GetHSVSliderValue() const;
    std::string GetHSVSliderName() const;

    // Layout setters
    void SetDeadZoneRadius(float radius);
    void SetInnerRadius(float radius);
    void SetOuterRadius(float radius);
    void SetRingSpacing(float spacing);
    void SetTextSize(float size);

    // Color setters
    void SetPageHoverColor(ImU32 color);
    void SetSubpageHoverColor(ImU32 color);
    void SetSubpageSelectedColor(ImU32 color);
    void SetBackgroundColor(ImU32 color);
    void SetOutlineColor(ImU32 color);
    void SetTextureSectorHoverColor(ImU32 color);
    void SetNavSectorHoverColor(ImU32 color);
    void SetLabelSectorHoverColor(ImU32 color);
    void SetSliderFillColor(ImU32 color);
    void SetSliderTrackColor(ImU32 color);

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
    ImU32 subpageSelectedColor = IM_COL32(80, 200, 120, 220);
    ImU32 backgroundColor = IM_COL32(30, 30, 40, 180);
    ImU32 outlineColor = IM_COL32(200, 200, 210, 255);

    float currentAngle = 0.0f;
    float currentRadius = 0.0f;
    int hoveredPage = -1;
    int hoveredSubPage = -1;

    // --- Navigation stack ---
    struct RingEntry {
        RingType type = RingType::NONE;
        int pageIndex = -1;        // SUBPAGE: which page
        int selectedIndex = -1;    // selected item in this ring

        // TEXTURE
        std::vector<ImTextureID> textures;
        int texturePage = 0;
        int hoveredTexture = -1;
        bool hoveredNavPrev = false;
        bool hoveredNavNext = false;

        // LABEL
        std::vector<std::string> items;
        int hoveredLabel = -1;
        int currentItem = -1;     // index of the currently active option

        // HSV_SLIDER
        std::string sliderName;
        float sliderValue = 0.0f;
        float sliderMin = 0.0f;
        float sliderMax = 1.0f;
    };

    std::vector<RingEntry> stack;

    // Textures to set on next texture ring push
    std::vector<ImTextureID> pendingTextures;

    // Ring colors
    ImU32 textureSectorHoverColor = IM_COL32(100, 200, 100, 220);
    ImU32 navSectorHoverColor = IM_COL32(180, 180, 60, 220);
    ImU32 labelSectorHoverColor = IM_COL32(100, 200, 100, 220);
    ImU32 labelCurrentItemColor = IM_COL32(80, 120, 200, 200);
    ImU32 sliderFillColor = IM_COL32(220, 180, 50, 220);
    ImU32 sliderTrackColor = IM_COL32(60, 60, 70, 180);

    static constexpr int kTexturesPerPage = 14;
    static constexpr int kNavSectors = 2;
    static constexpr int kTotalTexSectors = kTexturesPerPage + kNavSectors;

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
    void DrawSliderRing(ImDrawList* drawList, float innerR, float outerR,
                        float value, float minVal, float maxVal,
                        const std::string& label, ImU32 trackCol, ImU32 fillCol,
                        ImU32 textCol);
};
