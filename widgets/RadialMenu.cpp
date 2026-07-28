#include "RadialMenu.hpp"
#include <cmath>

static constexpr float TWO_PI = 6.28318530718f;
static constexpr float PI = 3.14159265358f;

// ---- Visibility ----

void RadialMenu::SetVisible(bool visible_)
{
    visible = visible_;
    if (!visible_)
        stack.clear();
}

bool RadialMenu::IsVisible() const { return visible; }

// ---- Basic setters ----

void RadialMenu::SetCenter(ImVec2 center_) { center = center_; }
void RadialMenu::SetInputVector(const ImVec2& vector) { inputVector = vector; }
void RadialMenu::SetPages(const std::vector<Page>& pages_) { pages = pages_; }

void RadialMenu::SetDeadZoneRadius(float r) { deadZoneRadius = r; }
void RadialMenu::SetInnerRadius(float r) { innerRadius = r; }
void RadialMenu::SetOuterRadius(float r) { outerRadius = r; }
void RadialMenu::SetRingSpacing(float s) { ringSpacing = s; }
void RadialMenu::SetTextSize(float s) { textSize = s; }

void RadialMenu::SetPageHoverColor(ImU32 c) { pageHoverColor = c; }
void RadialMenu::SetSubpageHoverColor(ImU32 c) { subpageHoverColor = c; }
void RadialMenu::SetSubpageSelectedColor(ImU32 c) { subpageSelectedColor = c; }
void RadialMenu::SetBackgroundColor(ImU32 c) { backgroundColor = c; }
void RadialMenu::SetOutlineColor(ImU32 c) { outlineColor = c; }
void RadialMenu::SetTextureSectorHoverColor(ImU32 c) { textureSectorHoverColor = c; }
void RadialMenu::SetNavSectorHoverColor(ImU32 c) { navSectorHoverColor = c; }
void RadialMenu::SetLabelSectorHoverColor(ImU32 c) { labelSectorHoverColor = c; }
void RadialMenu::SetSliderFillColor(ImU32 c) { sliderFillColor = c; }
void RadialMenu::SetSliderTrackColor(ImU32 c) { sliderTrackColor = c; }

// ---- Hover queries ----

int RadialMenu::GetHoveredPage() const { return hoveredPage; }
int RadialMenu::GetHoveredSubPage() const { return hoveredSubPage; }

// ---- Stack navigation ----

void RadialMenu::PushSubpageRing(int pageIndex)
{
    RingEntry e;
    e.type = RingType::SUBPAGE;
    e.pageIndex = pageIndex;
    e.selectedIndex = -1;
    stack.push_back(e);
}

void RadialMenu::PushTextureRing(const std::vector<ImTextureID>& textures)
{
    RingEntry e;
    e.type = RingType::TEXTURE;
    e.textures = textures;
    e.texturePage = 0;
    e.selectedIndex = -1;
    stack.push_back(e);
}

void RadialMenu::PushLabelRing(const std::vector<std::string>& items)
{
    RingEntry e;
    e.type = RingType::LABEL;
    e.items = items;
    e.selectedIndex = -1;
    stack.push_back(e);
}

void RadialMenu::PushHSVSliderRing(const std::string& name, float value, float minVal, float maxVal)
{
    RingEntry e;
    e.type = RingType::HSV_SLIDER;
    e.sliderName = name;
    e.sliderValue = value;
    e.sliderMin = minVal;
    e.sliderMax = maxVal;
    e.prevSliderAngle = -1.0f;
    e.selectedIndex = -1;
    stack.push_back(e);
}

void RadialMenu::PopRing()
{
    if (!stack.empty())
        stack.pop_back();
}

void RadialMenu::SetSelectedIndex(int index)
{
    if (!stack.empty())
        stack.back().selectedIndex = index;
}

void RadialMenu::ClearRings() { stack.clear(); }

int RadialMenu::GetStackDepth() const { return static_cast<int>(stack.size()); }

RadialMenu::RingType RadialMenu::GetActiveRingType() const
{
    return stack.empty() ? RingType::NONE : stack.back().type;
}

int RadialMenu::GetStackPageIndex() const
{
    for (int i = 0; i < static_cast<int>(stack.size()); ++i)
    {
        if (stack[i].type == RingType::SUBPAGE)
            return stack[i].pageIndex;
    }
    return -1;
}

// ---- Texture ring queries ----

void RadialMenu::SetTextures(const std::vector<ImTextureID>& textures)
{
    pendingTextures = textures;
}

int RadialMenu::GetHoveredTexture() const
{
    if (stack.empty() || stack.back().type != RingType::TEXTURE) return -1;
    return stack.back().hoveredTexture;
}

bool RadialMenu::GetHoveredNavPrev() const
{
    if (stack.empty() || stack.back().type != RingType::TEXTURE) return false;
    return stack.back().hoveredNavPrev;
}

bool RadialMenu::GetHoveredNavNext() const
{
    if (stack.empty() || stack.back().type != RingType::TEXTURE) return false;
    return stack.back().hoveredNavNext;
}

void RadialMenu::SetTexturePage(int page)
{
    if (stack.empty() || stack.back().type != RingType::TEXTURE) return;
    RingEntry& e = stack.back();
    int totalPages = (static_cast<int>(e.textures.size()) + kTexturesPerPage - 1) / kTexturesPerPage;
    if (totalPages <= 0) totalPages = 1;
    e.texturePage = page % totalPages;
    if (e.texturePage < 0) e.texturePage += totalPages;
}

int RadialMenu::GetTexturePage() const
{
    if (stack.empty() || stack.back().type != RingType::TEXTURE) return 0;
    return stack.back().texturePage;
}

void RadialMenu::ResetTexturePage()
{
    if (stack.empty() || stack.back().type != RingType::TEXTURE) return;
    stack.back().texturePage = 0;
}

// ---- Label ring queries ----

int RadialMenu::GetHoveredLabel() const
{
    if (stack.empty() || stack.back().type != RingType::LABEL) return -1;
    return stack.back().hoveredLabel;
}

// ---- HSV slider queries ----

void RadialMenu::SetHSVSliderValue(float value)
{
    if (stack.empty() || stack.back().type != RingType::HSV_SLIDER) return;
    stack.back().sliderValue = value;
}

float RadialMenu::GetHSVSliderValue() const
{
    if (stack.empty() || stack.back().type != RingType::HSV_SLIDER) return 0.0f;
    return stack.back().sliderValue;
}

std::string RadialMenu::GetHSVSliderName() const
{
    if (stack.empty() || stack.back().type != RingType::HSV_SLIDER) return {};
    return stack.back().sliderName;
}

// ---- Update ----

void RadialMenu::Update()
{
    hoveredPage = -1;
    hoveredSubPage = -1;

    if (!visible || pages.empty())
        return;

    // Apply pending textures to the active ring if it's a TEXTURE ring
    if (!stack.empty() && stack.back().type == RingType::TEXTURE && !pendingTextures.empty())
        stack.back().textures = pendingTextures;

    currentRadius = std::sqrt(inputVector.x * inputVector.x + inputVector.y * inputVector.y);
    currentAngle = std::atan2(inputVector.y, inputVector.x);
    if (currentAngle < 0.0f)
        currentAngle += TWO_PI;

    if (currentRadius < deadZoneRadius)
        return;

    float bandWidth = outerRadius - innerRadius;

    // Find the active (topmost) ring's band and handle input
    if (!stack.empty())
    {
        RingEntry& active = stack.back();
        int topIdx = static_cast<int>(stack.size()) - 1;

        // Compute the top ring's band
        float topInnerR = innerRadius + ringSpacing;
        float topOuterR = innerRadius + ringSpacing + bandWidth;
        for (int i = 0; i < topIdx; ++i)
        {
            topInnerR = topOuterR + ringSpacing;
            topOuterR = topInnerR + bandWidth;
        }

        switch (active.type)
        {
        case RingType::HSV_SLIDER:
        {
            if (currentRadius > topInnerR)
            {
                float sliderAngle = currentAngle + PI * 0.5f;
                if (sliderAngle >= TWO_PI) sliderAngle -= TWO_PI;
                if (sliderAngle < 0.0f)    sliderAngle += TWO_PI;

                if (active.prevSliderAngle < 0.0f)
                {
                    active.prevSliderAngle = sliderAngle;
                }
                else
                {
                    float delta = sliderAngle - active.prevSliderAngle;
                    if (delta > PI)  delta -= TWO_PI;
                    if (delta < -PI) delta += TWO_PI;

                    float range = active.sliderMax - active.sliderMin;
                    float newVal = active.sliderValue + (delta / TWO_PI) * range;

                    if (newVal <= active.sliderMin) {
                        newVal = active.sliderMin;
                        active.prevSliderAngle = sliderAngle;
                    } else if (newVal >= active.sliderMax) {
                        newVal = active.sliderMax;
                        active.prevSliderAngle = sliderAngle;
                    }

                    active.sliderValue = newVal;
                }
                active.prevSliderAngle = sliderAngle;
            }
            return;
        }

        case RingType::TEXTURE:
        {
            active.hoveredTexture = -1;
            active.hoveredNavPrev = false;
            active.hoveredNavNext = false;
            if (currentRadius > topInnerR && currentRadius <= topOuterR)
            {
                float sectorAngle = TWO_PI / static_cast<float>(kTotalTexSectors);
                int idx = static_cast<int>(currentAngle / sectorAngle);
                if (idx >= kTotalTexSectors)
                    idx = kTotalTexSectors - 1;

                active.hoveredNavPrev = (idx == kTexturesPerPage);
                active.hoveredNavNext = (idx == kTexturesPerPage + 1);

                if (!active.hoveredNavPrev && !active.hoveredNavNext)
                {
                    int pageOffset = active.texturePage * kTexturesPerPage;
                    int absIdx = pageOffset + idx;
                    if (absIdx < static_cast<int>(active.textures.size()))
                        active.hoveredTexture = absIdx;
                }
            }
            return;
        }

        case RingType::LABEL:
        {
            active.hoveredLabel = -1;
            if (currentRadius > topInnerR && currentRadius <= topOuterR)
            {
                int count = static_cast<int>(active.items.size());
                float itemAngle = TWO_PI / static_cast<float>(count);
                int idx = static_cast<int>(currentAngle / itemAngle);
                if (idx >= count)
                    idx = count - 1;
                active.hoveredLabel = idx;
            }
            return;
        }

        case RingType::SUBPAGE:
        {
            // Subpage is the active ring — detect subpage hover in its band
            if (active.pageIndex >= 0 && active.pageIndex < static_cast<int>(pages.size()))
            {
                const Page& page = pages[active.pageIndex];
                if (!page.subPages.empty())
                {
                    float subAngle = TWO_PI / static_cast<float>(page.subPages.size());
                    if (currentRadius > topInnerR && currentRadius <= topOuterR)
                    {
                        int subIdx = static_cast<int>(currentAngle / subAngle);
                        if (subIdx >= static_cast<int>(page.subPages.size()))
                            subIdx = static_cast<int>(page.subPages.size()) - 1;
                        hoveredSubPage = subIdx;
                    }
                }
            }
            return;
        }

        case RingType::NONE:
            break;
        }
    }

    // If a SUBPAGE ring is in the stack (but not the top), detect subpage hover in its band
    for (int si = 0; si < static_cast<int>(stack.size()); ++si)
    {
        if (stack[si].type == RingType::SUBPAGE)
        {
            // Compute this layer's band
            float layerInnerR = innerRadius + ringSpacing;
            float layerOuterR = innerRadius + ringSpacing + bandWidth;
            for (int i = 0; i < si; ++i)
            {
                layerInnerR = layerOuterR + ringSpacing;
                layerOuterR = layerInnerR + bandWidth;
            }

            if (stack[si].pageIndex >= 0 && stack[si].pageIndex < static_cast<int>(pages.size()))
            {
                const Page& page = pages[stack[si].pageIndex];
                if (!page.subPages.empty())
                {
                    float subAngle = TWO_PI / static_cast<float>(page.subPages.size());
                    if (currentRadius > layerInnerR && currentRadius <= layerOuterR)
                    {
                        int subIdx = static_cast<int>(currentAngle / subAngle);
                        if (subIdx >= static_cast<int>(page.subPages.size()))
                            subIdx = static_cast<int>(page.subPages.size()) - 1;
                        hoveredSubPage = subIdx;
                    }
                }
            }
            break;
        }
    }

    // Detect page hover only when no SUBPAGE ring is in the stack
    bool hasSubpage = false;
    for (int i = 0; i < static_cast<int>(stack.size()); ++i)
    {
        if (stack[i].type == RingType::SUBPAGE) { hasSubpage = true; break; }
    }

    if (!hasSubpage && currentRadius <= outerRadius)
    {
        int pageCount = static_cast<int>(pages.size());
        float pageAngle = TWO_PI / static_cast<float>(pageCount);
        int pageIdx = static_cast<int>(currentAngle / pageAngle);
        if (pageIdx >= pageCount)
            pageIdx = pageCount - 1;
        hoveredPage = pageIdx;

        if (currentRadius < innerRadius)
            return;
    }
}

// ---- Draw ----

void RadialMenu::Draw()
{
    if (!visible || pages.empty())
        return;

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    int pageCount = static_cast<int>(pages.size());
    float pageAngle = TWO_PI / static_cast<float>(pageCount);

    // 1. Inner ring: pages (deadZone → innerRadius)
    int selectedPageIndex = GetStackPageIndex();

    for (int i = 0; i < pageCount; ++i)
    {
        float startAngle = static_cast<float>(i) * pageAngle;
        float endAngle = startAngle + pageAngle;

        ImU32 fillColor = backgroundColor;
        if (i == selectedPageIndex)
            fillColor = subpageSelectedColor;
        else if (i == hoveredPage)
            fillColor = pageHoverColor;

        DrawSector(drawList, startAngle, endAngle, deadZoneRadius, innerRadius, fillColor, outlineColor);
        DrawLabel(drawList, pages[i].label, startAngle, endAngle, deadZoneRadius, innerRadius, IM_COL32(255, 255, 255, 255));
    }

    // 2. Draw all rings from the stack (bottom to top), each at its own band
    float bandInner = innerRadius + ringSpacing;
    float bandWidth = outerRadius - innerRadius;

    for (int si = 0; si < static_cast<int>(stack.size()); ++si)
    {
        const RingEntry& entry = stack[si];
        bool isTop = (si == static_cast<int>(stack.size()) - 1);

        float layerInnerR = bandInner;
        float layerOuterR = bandInner + bandWidth;

        switch (entry.type)
        {
        case RingType::SUBPAGE:
        {
            if (entry.pageIndex < 0 || entry.pageIndex >= pageCount)
                break;
            const Page& page = pages[entry.pageIndex];
            if (page.subPages.empty())
                break;

            int subCount = static_cast<int>(page.subPages.size());
            float subAngle = TWO_PI / static_cast<float>(subCount);

            bool hasActiveRing = !isTop;

            for (int j = 0; j < subCount; ++j)
            {
                float startAngle = static_cast<float>(j) * subAngle;
                float endAngle = startAngle + subAngle;

                ImU32 fillColor = backgroundColor;
                if (hasActiveRing && j == entry.selectedIndex)
                    fillColor = subpageSelectedColor;
                else if (isTop && j == hoveredSubPage)
                    fillColor = subpageHoverColor;

                DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, fillColor, outlineColor);
                DrawLabel(drawList, page.subPages[j].label, startAngle, endAngle, layerInnerR, layerOuterR, IM_COL32(255, 255, 255, 255));
            }
            break;
        }

        case RingType::TEXTURE:
        {
            if (entry.textures.empty()) break;
            int totalTex = static_cast<int>(entry.textures.size());
            float sectorAngle = TWO_PI / static_cast<float>(kTotalTexSectors);

            for (int s = 0; s < kTotalTexSectors; ++s)
            {
                float startAngle = static_cast<float>(s) * sectorAngle;
                float endAngle = startAngle + sectorAngle;

                if (s == kTexturesPerPage)
                {
                    bool hovered = isTop && entry.hoveredNavPrev;
                    ImU32 fillColor = hovered ? navSectorHoverColor : backgroundColor;
                    DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, fillColor, outlineColor);
                    DrawArrow(drawList, startAngle, endAngle, layerInnerR, layerOuterR, false, IM_COL32(255, 255, 255, 255));
                }
                else if (s == kTexturesPerPage + 1)
                {
                    bool hovered = isTop && entry.hoveredNavNext;
                    ImU32 fillColor = hovered ? navSectorHoverColor : backgroundColor;
                    DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, fillColor, outlineColor);
                    DrawArrow(drawList, startAngle, endAngle, layerInnerR, layerOuterR, true, IM_COL32(255, 255, 255, 255));
                }
                else
                {
                    int pageOffset = entry.texturePage * kTexturesPerPage;
                    int texIdx = pageOffset + s;
                    bool hovered = isTop && (texIdx == entry.hoveredTexture);
                    bool selected = !isTop && (texIdx == entry.selectedIndex);
                    if (texIdx < totalTex)
                        DrawTextureSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR,
                                          entry.textures[texIdx], hovered || selected, outlineColor);
                    else
                        DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, backgroundColor, outlineColor);
                }
            }
            break;
        }

        case RingType::LABEL:
        {
            if (entry.items.empty()) break;
            int count = static_cast<int>(entry.items.size());
            float itemAngle = TWO_PI / static_cast<float>(count);

            for (int i = 0; i < count; ++i)
            {
                float startAngle = static_cast<float>(i) * itemAngle;
                float endAngle = startAngle + itemAngle;

                ImU32 fillColor = backgroundColor;
                if (isTop && i == entry.hoveredLabel)
                    fillColor = labelSectorHoverColor;
                else if (!isTop && i == entry.selectedIndex)
                    fillColor = subpageSelectedColor;

                DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, fillColor, outlineColor);
                DrawLabel(drawList, entry.items[i], startAngle, endAngle, layerInnerR, layerOuterR, IM_COL32(255, 255, 255, 255));
            }
            break;
        }

        case RingType::HSV_SLIDER:
        {
            DrawSliderRing(drawList, layerInnerR, layerOuterR,
                           entry.sliderValue, entry.sliderMin, entry.sliderMax, entry.sliderName,
                           sliderTrackColor, sliderFillColor, IM_COL32(255, 255, 255, 255));
            break;
        }

        case RingType::NONE:
            break;
        }

        bandInner = layerOuterR + ringSpacing;
    }

    // Center circle
    drawList->AddCircleFilled(center, deadZoneRadius, IM_COL32(20, 20, 30, 200), 64);
    drawList->AddCircle(center, deadZoneRadius, outlineColor, 64, 1.0f);
}

// ---- Drawing primitives ----

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

    drawList->AddConcavePolyFilled(points.Data, points.Size, fillColor);

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

void RadialMenu::DrawArrow(ImDrawList* drawList, float angleStart, float angleEnd,
                            float innerR, float outerR, bool right, ImU32 color)
{
    float midAngle = (angleStart + angleEnd) * 0.5f;
    float midRadius = (innerR + outerR) * 0.5f;

    const char* str = right ? ">" : "<";
    ImVec2 textSize_ = ImGui::CalcTextSize(str);
    ImVec2 textPos = ImVec2(
        center.x + std::cos(midAngle) * midRadius - textSize_.x * 0.5f,
        center.y + std::sin(midAngle) * midRadius - textSize_.y * 0.5f
    );

    drawList->AddText(textPos, color, str);
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
            drawList->AddImageQuad(tex, p0, p1, p2, p3,
                                   ImVec2(u0, 0), ImVec2(u1, 0), ImVec2(u1, 1), ImVec2(u0, 1),
                                   textureSectorHoverColor);
        else
            drawList->AddImageQuad(tex, p0, p1, p2, p3,
                                   ImVec2(u0, 0), ImVec2(u1, 0), ImVec2(u1, 1), ImVec2(u0, 1),
                                   IM_COL32_WHITE);
    }

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

    if (hovered)
    {
        drawList->AddCircle(center, innerR, outlineCol, 64, 2.0f);
        drawList->AddCircle(center, outerR, outlineCol, 64, 2.0f);
    }
}

void RadialMenu::DrawSliderRing(ImDrawList* drawList, float innerR, float outerR,
                                float value, float minVal, float maxVal,
                                const std::string& label, ImU32 trackCol, ImU32 fillCol,
                                ImU32 textCol)
{
    float range = maxVal - minVal;
    if (range <= 0.0f) range = 1.0f;
    float t = (value - minVal) / range;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    float startAngle = -PI * 0.5f;
    float fillAngle = startAngle + t * TWO_PI;

    DrawSector(drawList, startAngle, startAngle + TWO_PI, innerR, outerR, trackCol, outlineColor);

    if (t > 0.01f)
    {
        DrawSector(drawList, startAngle, fillAngle, innerR, outerR, fillCol, outlineColor);
    }

    char buf[64];
    if (label == "Hue" || label == "Azimuth" || label == "Elevation")
        snprintf(buf, sizeof(buf), "%s: %.0f", label.c_str(), value);
    else
        snprintf(buf, sizeof(buf), "%s: %.1f%%", label.c_str(), value);

    ImVec2 textSize_ = ImGui::CalcTextSize(buf);
    drawList->AddText(
        ImVec2(center.x - textSize_.x * 0.5f, center.y - textSize_.y * 0.5f),
        textCol, buf);

    for (int i = 0; i <= 4; ++i)
    {
        float angle = startAngle + static_cast<float>(i) / 4.0f * TWO_PI;
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        drawList->AddLine(
            ImVec2(center.x + cosA * innerR, center.y + sinA * innerR),
            ImVec2(center.x + cosA * (innerR + 4.0f), center.y + sinA * (innerR + 4.0f)),
            outlineColor, 1.5f);
    }

    {
        float cosA = std::cos(fillAngle);
        float sinA = std::sin(fillAngle);
        float dotR = (innerR + outerR) * 0.5f;
        drawList->AddCircleFilled(
            ImVec2(center.x + cosA * dotR, center.y + sinA * dotR),
            5.0f, IM_COL32(255, 255, 255, 255), 16);
    }
}
