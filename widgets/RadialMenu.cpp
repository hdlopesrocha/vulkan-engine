#include "RadialMenu.hpp"
#include <cmath>

static constexpr float TWO_PI = 6.28318530718f;

void RadialMenu::SetVisible(bool visible_)
{
    visible = visible_;
}

bool RadialMenu::IsVisible() const
{
    return visible;
}

void RadialMenu::SetCenter(ImVec2 center_)
{
    center = center_;
}

void RadialMenu::SetInputVector(const ImVec2& vector)
{
    inputVector = vector;
}

void RadialMenu::SetPages(const std::vector<Page>& pages_)
{
    pages = pages_;
}

void RadialMenu::SetDeadZoneRadius(float radius)
{
    deadZoneRadius = radius;
}

void RadialMenu::SetInnerRadius(float radius)
{
    innerRadius = radius;
}

void RadialMenu::SetOuterRadius(float radius)
{
    outerRadius = radius;
}

void RadialMenu::SetRingSpacing(float spacing)
{
    ringSpacing = spacing;
}

void RadialMenu::SetTextSize(float size)
{
    textSize = size;
}

void RadialMenu::SetPageHoverColor(ImU32 color)
{
    pageHoverColor = color;
}

void RadialMenu::SetSubpageHoverColor(ImU32 color)
{
    subpageHoverColor = color;
}

void RadialMenu::SetBackgroundColor(ImU32 color)
{
    backgroundColor = color;
}

void RadialMenu::SetOutlineColor(ImU32 color)
{
    outlineColor = color;
}

int RadialMenu::GetHoveredPage() const
{
    return hoveredPage;
}

int RadialMenu::GetHoveredSubPage() const
{
    return hoveredSubPage;
}

void RadialMenu::SetTextureRingActive(bool active)
{
    textureRingActive = active;
}

bool RadialMenu::GetTextureRingActive() const
{
    return textureRingActive;
}

void RadialMenu::SetTextures(const std::vector<ImTextureID>& textures_)
{
    textures = textures_;
}

int RadialMenu::GetHoveredTexture() const
{
    return hoveredTexture;
}

void RadialMenu::SetTextureRingRadius(float radius)
{
    textureRingRadius = radius;
}

void RadialMenu::SetTextureSectorHoverColor(ImU32 color)
{
    textureSectorHoverColor = color;
}

void RadialMenu::Update()
{
    hoveredPage = -1;
    hoveredSubPage = -1;
    hoveredTexture = -1;

    if (!visible || pages.empty())
        return;

    currentRadius = std::sqrt(inputVector.x * inputVector.x + inputVector.y * inputVector.y);
    currentAngle = std::atan2(inputVector.y, inputVector.x);
    if (currentAngle < 0.0f)
        currentAngle += TWO_PI;

    if (currentRadius < deadZoneRadius)
        return;

    // Texture ring hover detection (outermost ring)
    if (textureRingActive && !textures.empty())
    {
        if (currentRadius > outerRadius + ringSpacing && currentRadius <= textureRingRadius)
        {
            int texCount = static_cast<int>(textures.size());
            float texAngle = TWO_PI / static_cast<float>(texCount);
            int texIdx = static_cast<int>(currentAngle / texAngle);
            if (texIdx >= texCount)
                texIdx = texCount - 1;
            hoveredTexture = texIdx;
        }
        // When texture ring is active, don't process inner rings
        return;
    }

    if (currentRadius > outerRadius)
        return;

    int pageCount = static_cast<int>(pages.size());
    float pageAngle = TWO_PI / static_cast<float>(pageCount);

    int pageIdx = static_cast<int>(currentAngle / pageAngle);
    if (pageIdx >= pageCount)
        pageIdx = pageCount - 1;

    hoveredPage = pageIdx;

    if (currentRadius < innerRadius)
        return;

    const Page& page = pages[pageIdx];
    if (page.subPages.empty())
        return;

    int subCount = static_cast<int>(page.subPages.size());
    float subAngle = pageAngle / static_cast<float>(subCount);

    float pageStart = static_cast<float>(pageIdx) * pageAngle;
    float localAngle = currentAngle - pageStart;

    int subIdx = static_cast<int>(localAngle / subAngle);
    if (subIdx >= subCount)
        subIdx = subCount - 1;

    hoveredSubPage = subIdx;
}

void RadialMenu::Draw()
{
    if (!visible || pages.empty())
        return;

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    int pageCount = static_cast<int>(pages.size());
    float pageAngle = TWO_PI / static_cast<float>(pageCount);

    // Inner ring: pages (deadZone → innerRadius)
    for (int i = 0; i < pageCount; ++i)
    {
        float startAngle = static_cast<float>(i) * pageAngle;
        float endAngle = startAngle + pageAngle;

        ImU32 fillColor = (i == hoveredPage) ? pageHoverColor : backgroundColor;

        DrawSector(drawList, startAngle, endAngle, deadZoneRadius, innerRadius, fillColor, outlineColor);

        DrawLabel(drawList, pages[i].label, startAngle, endAngle, deadZoneRadius, innerRadius, IM_COL32(255, 255, 255, 255));
    }

    // Outer ring: subpages of hovered page (innerRadius → outerRadius)
    if (hoveredPage >= 0 && hoveredPage < pageCount)
    {
        const Page& page = pages[hoveredPage];
        if (!page.subPages.empty())
        {
            int subCount = static_cast<int>(page.subPages.size());
            float subAngle = pageAngle / static_cast<float>(subCount);
            float pageStart = static_cast<float>(hoveredPage) * pageAngle;

            for (int j = 0; j < subCount; ++j)
            {
                float startAngle = pageStart + static_cast<float>(j) * subAngle;
                float endAngle = startAngle + subAngle;

                ImU32 fillColor = (j == hoveredSubPage) ? subpageHoverColor : backgroundColor;

                DrawSector(drawList, startAngle, endAngle, innerRadius + ringSpacing, outerRadius, fillColor, outlineColor);

                DrawLabel(drawList, page.subPages[j].label, startAngle, endAngle, innerRadius + ringSpacing, outerRadius, IM_COL32(255, 255, 255, 255));
            }
        }
    }

    // Texture ring: spans 360° (outerRadius + ringSpacing → textureRingRadius)
    if (textureRingActive && !textures.empty())
    {
        int texCount = static_cast<int>(textures.size());
        float texAngle = TWO_PI / static_cast<float>(texCount);

        for (int t = 0; t < texCount; ++t)
        {
            float startAngle = static_cast<float>(t) * texAngle;
            float endAngle = startAngle + texAngle;

            bool hovered = (t == hoveredTexture);
            DrawTextureSector(drawList, startAngle, endAngle,
                              outerRadius + ringSpacing, textureRingRadius,
                              textures[t], hovered, outlineColor);
        }
    }

    drawList->AddCircleFilled(center, deadZoneRadius, IM_COL32(20, 20, 30, 200), 64);
    drawList->AddCircle(center, deadZoneRadius, outlineColor, 64, 1.0f);
}

void RadialMenu::DrawSector(ImDrawList* drawList, float startAngle, float endAngle,
                             float innerR, float outerR, ImU32 fillColor, ImU32 outlineCol)
{
    const int segments = 32;
    ImVector<ImVec2> points;

    for (int i = 0; i <= segments; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(segments);
        float angle = startAngle + t * (endAngle - startAngle);
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        points.push_back(ImVec2(center.x + cosA * outerR, center.y + sinA * outerR));
    }

    for (int i = segments; i >= 0; --i)
    {
        float t = static_cast<float>(i) / static_cast<float>(segments);
        float angle = startAngle + t * (endAngle - startAngle);
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        points.push_back(ImVec2(center.x + cosA * innerR, center.y + sinA * innerR));
    }

    drawList->AddConvexPolyFilled(points.Data, points.Size, fillColor);

    for (int i = 0; i < points.Size; ++i)
    {
        drawList->AddLine(points[i], points[(i + 1) % points.Size], outlineCol, 1.0f);
    }
}

void RadialMenu::DrawLabel(ImDrawList* drawList, const std::string& label,
                            float angleStart, float angleEnd,
                            float innerR, float outerR, ImU32 color)
{
    float midAngle = (angleStart + angleEnd) * 0.5f;
    float midRadius = (innerR + outerR) * 0.5f;

    ImVec2 textSize_ = ImGui::CalcTextSize(label.c_str());
    ImVec2 textPos = ImVec2(
        center.x + std::cos(midAngle) * midRadius - textSize_.x * 0.5f,
        center.y + std::sin(midAngle) * midRadius - textSize_.y * 0.5f
    );

    drawList->AddText(textPos, color, label.c_str());
}

void RadialMenu::DrawTextureSector(ImDrawList* drawList, float startAngle, float endAngle,
                                    float innerR, float outerR, ImTextureID tex,
                                    bool hovered, ImU32 outlineCol)
{
    if (!tex)
    {
        DrawSector(drawList, startAngle, endAngle, innerR, outerR, backgroundColor, outlineCol);
        return;
    }

    // Approximate the curved sector with UV-mapped quads
    const int segments = 16;
    float span = endAngle - startAngle;

    for (int i = 0; i < segments; ++i)
    {
        float a0 = startAngle + span * static_cast<float>(i) / static_cast<float>(segments);
        float a1 = startAngle + span * static_cast<float>(i + 1) / static_cast<float>(segments);
        float c0 = std::cos(a0), s0 = std::sin(a0);
        float c1 = std::cos(a1), s1 = std::sin(a1);

        ImVec2 p0(center.x + c0 * innerR, center.y + s0 * innerR);
        ImVec2 p1(center.x + c1 * innerR, center.y + s1 * innerR);
        ImVec2 p2(center.x + c1 * outerR, center.y + s1 * outerR);
        ImVec2 p3(center.x + c0 * outerR, center.y + s0 * outerR);

        float u0 = static_cast<float>(i) / static_cast<float>(segments);
        float u1 = static_cast<float>(i + 1) / static_cast<float>(segments);

        if (hovered)
        {
            drawList->AddImageQuad(tex, p0, p1, p2, p3,
                                   ImVec2(u0, 0), ImVec2(u1, 0), ImVec2(u1, 1), ImVec2(u0, 1),
                                   textureSectorHoverColor);
        }
        else
        {
            drawList->AddImageQuad(tex, p0, p1, p2, p3,
                                   ImVec2(u0, 0), ImVec2(u1, 0), ImVec2(u1, 1), ImVec2(u0, 1),
                                   IM_COL32_WHITE);
        }
    }

    // Outline
    const int outlineSegments = 32;
    for (int i = 0; i <= outlineSegments; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(outlineSegments);
        float a = startAngle + t * span;
        float ca = std::cos(a), sa = std::sin(a);

        ImVec2 inner(center.x + ca * innerR, center.y + sa * innerR);
        ImVec2 outer(center.x + ca * outerR, center.y + sa * outerR);

        if (i > 0)
        {
            float prevA = startAngle + static_cast<float>(i - 1) / static_cast<float>(outlineSegments) * span;
            float cpa = std::cos(prevA), spa = std::sin(prevA);
            drawList->AddLine(ImVec2(center.x + cpa * innerR, center.y + spa * innerR), inner, outlineCol, 1.0f);
            drawList->AddLine(ImVec2(center.x + cpa * outerR, center.y + spa * outerR), outer, outlineCol, 1.0f);
        }
        if (i == 0 || i == outlineSegments)
        {
            drawList->AddLine(inner, outer, outlineCol, 1.0f);
        }
    }

    // Hover highlight ring
    if (hovered)
    {
        drawList->AddCircle(center, innerR, outlineCol, 64, 2.0f);
        drawList->AddCircle(center, outerR, outlineCol, 64, 2.0f);
    }
}
