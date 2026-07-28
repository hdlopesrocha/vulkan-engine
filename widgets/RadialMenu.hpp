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

    void DrawSector(ImDrawList* drawList, float startAngle, float endAngle,
                    float innerR, float outerR, ImU32 fillColor, ImU32 outlineCol);
    void DrawLabel(ImDrawList* drawList, const std::string& label,
                   float angleStart, float angleEnd,
                   float innerR, float outerR, ImU32 color);
};
