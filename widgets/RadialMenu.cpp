#include "RadialMenu.hpp"
#include "RadialMenuIcons.hpp"
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

void RadialMenu::SetHoverColor(ImU32 c) { hoverColor = c; }
void RadialMenu::SetSelectedColor(ImU32 c) { selectedColor = c; }
void RadialMenu::SetBackgroundColor(ImU32 c) { backgroundColor = c; }
void RadialMenu::SetOutlineColor(ImU32 c) { outlineColor = c; }
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

void RadialMenu::PushLabelRing(const std::vector<std::string>& items, const std::vector<std::string>& textItems)
{
    RingEntry e;
    e.type = RingType::LABEL;
    e.items = items;
    e.textItems = textItems;
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

std::string RadialMenu::GetHoveredLabelItem() const
{
    if (stack.empty() || stack.back().type != RingType::LABEL) return {};
    const RingEntry& e = stack.back();
    if (e.hoveredLabel >= 0 && e.hoveredLabel < static_cast<int>(e.items.size()))
        return e.items[e.hoveredLabel];
    return {};
}

std::string RadialMenu::GetHoveredLabelTextItem() const
{
    if (stack.empty() || stack.back().type != RingType::LABEL) return {};
    const RingEntry& e = stack.back();
    if (e.hoveredLabel >= 0 && e.hoveredLabel < static_cast<int>(e.textItems.size()))
        return e.textItems[e.hoveredLabel];
    return {};
}

void RadialMenu::SetCurrentItem(int index)
{
    if (!stack.empty() && stack.back().type == RingType::LABEL)
        stack.back().currentItem = index;
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

// ---- Center circle labels ----

void RadialMenu::SetCenterLabels(const std::vector<std::string>& labels)
{
    centerLabels = labels;
}

// ---- Center labels from full stack ----

std::vector<std::string> RadialMenu::BuildCenterLabels() const
{
    std::vector<std::string> out;

    // First line: page text
    // If stack has a SUBPAGE ring, use its page; otherwise use hoveredPage
    int pageIdx = GetStackPageIndex();
    if (pageIdx >= 0 && pageIdx < static_cast<int>(pages.size())) {
        out.push_back(pages[pageIdx].textLabel);
    } else if (hoveredPage >= 0 && hoveredPage < static_cast<int>(pages.size())) {
        out.push_back(pages[hoveredPage].textLabel);
    }

    for (size_t i = 0; i < stack.size(); ++i)
    {
        const RingEntry& e = stack[i];

        switch (e.type)
        {
        case RingType::SUBPAGE:
        {
            int hs = (i == stack.size() - 1) ? hoveredSubPage : e.selectedIndex;
            if (e.pageIndex >= 0 && e.pageIndex < static_cast<int>(pages.size())) {
                const auto& subs = pages[e.pageIndex].subPages;
                if (hs >= 0 && hs < static_cast<int>(subs.size()))
                    out.push_back(subs[hs].textLabel);
            }
            break;
        }
        case RingType::TEXTURE:
        {
            int ht = (i == stack.size() - 1) ? e.hoveredTexture : e.selectedIndex;
            if (ht >= 0)
                out.push_back(std::to_string(ht));
            else if (i == stack.size() - 1 && e.hoveredNavPrev)
                out.push_back(ICON_FA_ANGLE_LEFT);
            else if (i == stack.size() - 1 && e.hoveredNavNext)
                out.push_back(ICON_FA_ANGLE_RIGHT);
            break;
        }
        case RingType::LABEL:
        {
            int li = (i == stack.size() - 1) ? e.hoveredLabel : e.selectedIndex;
            if (li < 0) li = e.currentItem;
            if (li >= 0 && li < static_cast<int>(e.textItems.size()))
                out.push_back(e.textItems[li]);
            break;
        }
        case RingType::HSV_SLIDER:
        {
            char buf[64];
            if (e.sliderName == "Hue" || e.sliderName == "Azimuth" || e.sliderName == "Elevation")
                snprintf(buf, sizeof(buf), "%.0f", e.sliderValue);
            else
                snprintf(buf, sizeof(buf), "%.1f%%",  e.sliderValue);
            out.push_back(buf);
            break;
        }
        case RingType::NONE:
            break;
        }
    }

    return out;
}

// ---- Helpers ----

int RadialMenu::GetSectorIndex(float angle, int count) const
{
    float sectorAngle = TWO_PI / static_cast<float>(count);
    int idx = static_cast<int>(std::fmod(angle + sectorAngle * 0.5f, TWO_PI) / sectorAngle);
    if (idx >= count)
        idx = count - 1;
    return idx;
}

void RadialMenu::DetectSubpageHover(int pageIndex)
{
    if (pageIndex < 0 || pageIndex >= static_cast<int>(pages.size()))
        return;
    const Page& page = pages[pageIndex];
    if (page.subPages.empty())
        return;
    hoveredSubPage = GetSectorIndex(currentAngle, static_cast<int>(page.subPages.size()));
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
    currentAngle = std::fmod(std::atan2(inputVector.y, inputVector.x) + PI * 0.5f + TWO_PI, TWO_PI);

    // Detect hover for the active ring (works even inside dead zone)
    if (!stack.empty())
    {
        RingEntry& active = stack.back();

        switch (active.type)
        {
        case RingType::HSV_SLIDER:
        {
            if (currentRadius >= deadZoneRadius) {
                float range = active.sliderMax - active.sliderMin;
                float rawValue = active.sliderMin + (currentAngle / TWO_PI) * range;

                float halfRange = range * 0.5f;
                if (std::abs(rawValue - active.sliderValue) > halfRange)
                    rawValue = (rawValue > active.sliderValue) ? active.sliderMin : active.sliderMax;

                active.sliderValue = rawValue;
            }
            centerLabels = BuildCenterLabels();
            return;
        }

        case RingType::TEXTURE:
        {
            active.hoveredTexture = -1;
            active.hoveredNavPrev = false;
            active.hoveredNavNext = false;
            if (currentRadius > deadZoneRadius)
            {
                int idx = GetSectorIndex(currentAngle, kTotalTexSectors);

                active.hoveredNavPrev = (idx == kTexturesPerPage);
                active.hoveredNavNext = (idx == kTexturesPerPage + 1);

                if (!active.hoveredNavPrev && !active.hoveredNavNext)
                {
                    int absIdx = active.texturePage * kTexturesPerPage + idx;
                    if (absIdx < static_cast<int>(active.textures.size()))
                        active.hoveredTexture = absIdx;
                }
            }
            centerLabels = BuildCenterLabels();
            return;
        }

        case RingType::LABEL:
        {
            active.hoveredLabel = GetSectorIndex(currentAngle, static_cast<int>(active.items.size()));
            centerLabels = BuildCenterLabels();
            return;
        }

        case RingType::SUBPAGE:
        {
            DetectSubpageHover(active.pageIndex);
            centerLabels = BuildCenterLabels();
            return;
        }

        case RingType::NONE:
            break;
        }
    }

    // Ghost subpage hover: detect hover for a SUBPAGE ring lower in the stack
    for (int si = 0; si < static_cast<int>(stack.size()); ++si)
    {
        if (stack[si].type == RingType::SUBPAGE)
        {
            DetectSubpageHover(stack[si].pageIndex);
            break;
        }
    }

    // Page hover: only when no SUBPAGE ring is in the stack
    bool hasSubpage = false;
    for (int i = 0; i < static_cast<int>(stack.size()); ++i)
    {
        if (stack[i].type == RingType::SUBPAGE) { hasSubpage = true; break; }
    }

    if (!hasSubpage)
    {
        hoveredPage = GetSectorIndex(currentAngle, static_cast<int>(pages.size()));
    }

    centerLabels = BuildCenterLabels();
}

// ---- Draw ----

void RadialMenu::Draw()
{
    if (!visible || pages.empty())
        return;

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    int pageCount = static_cast<int>(pages.size());
    float angleOffset = -PI * 0.5f; // Start sectors at 12 o'clock

    // 1. Inner ring: pages (deadZone → innerRadius)
    int selectedPageIndex = GetStackPageIndex();

    DrawSectorRing(drawList, pageCount, deadZoneRadius, innerRadius,
        [&](int i) -> std::string { return pages[i].label; },
        [&](int i) -> ImU32 {
            if (i == selectedPageIndex) return selectedColor;
            if (i == hoveredPage) return hoverColor;
            return 0;
        });

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

            bool hasActiveRing = !isTop;

            DrawSectorRing(drawList, static_cast<int>(page.subPages.size()), layerInnerR, layerOuterR,
                [&](int j) -> std::string { return page.subPages[j].label; },
                [&](int j) -> ImU32 {
                    if (hasActiveRing && j == entry.selectedIndex) return selectedColor;
                    if (isTop && j == hoveredSubPage) return hoverColor;
                    return 0;
                });
            break;
        }

        case RingType::TEXTURE:
        {
            if (entry.textures.empty()) break;
            int totalTex = static_cast<int>(entry.textures.size());
            float sectorAngle = TWO_PI / static_cast<float>(kTotalTexSectors);

            for (int s = 0; s < kTotalTexSectors; ++s)
            {
                float startAngle = angleOffset - sectorAngle * 0.5f + static_cast<float>(s) * sectorAngle;
                float endAngle = startAngle + sectorAngle;

                if (s == kTexturesPerPage)
                {
                    bool hovered = isTop && entry.hoveredNavPrev;
                    DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, backgroundColor, outlineColor);
                    DrawArrow(drawList, startAngle, endAngle, layerInnerR, layerOuterR, false, IM_COL32(255, 255, 255, 255));
                    if (hovered)
                        DrawInnerBorder(drawList, startAngle, endAngle, layerInnerR, hoverColor, 10.0f);
                }
                else if (s == kTexturesPerPage + 1)
                {
                    bool hovered = isTop && entry.hoveredNavNext;
                    DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, backgroundColor, outlineColor);
                    DrawArrow(drawList, startAngle, endAngle, layerInnerR, layerOuterR, true, IM_COL32(255, 255, 255, 255));
                    if (hovered)
                        DrawInnerBorder(drawList, startAngle, endAngle, layerInnerR, hoverColor, 10.0f);
                }
                else
                {
                    int pageOffset = entry.texturePage * kTexturesPerPage;
                    int texIdx = pageOffset + s;
                    bool hovered = isTop && (texIdx == entry.hoveredTexture);
                    bool selected = (texIdx == entry.selectedIndex);
                    if (texIdx < totalTex)
                        DrawTextureSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR,
                                          entry.textures[texIdx], outlineColor);
                    else
                        DrawSector(drawList, startAngle, endAngle, layerInnerR, layerOuterR, backgroundColor, outlineColor);

                    if (hovered)
                        DrawInnerBorder(drawList, startAngle, endAngle, layerInnerR, hoverColor, 10.0f);
                    else if (selected)
                        DrawInnerBorder(drawList, startAngle, endAngle, layerInnerR, selectedColor, 10.0f);
                }
            }
            break;
        }

        case RingType::LABEL:
        {
            if (entry.items.empty()) break;

            DrawSectorRing(drawList, static_cast<int>(entry.items.size()), layerInnerR, layerOuterR,
                [&](int i) -> std::string { return entry.items[i]; },
                [&](int i) -> ImU32 {
                    if (isTop && i == entry.hoveredLabel) return hoverColor;
                    if (!isTop && i == entry.selectedIndex) return selectedColor;
                    if (i == entry.currentItem) return selectedColor;
                    return 0;
                });
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

    // Draw center labels line by line
    if (!centerLabels.empty())
    {
        float lineH = textSize * 1.3f;
        float totalH = lineH * static_cast<float>(centerLabels.size());
        float startY = center.y - totalH * 0.5f + lineH * 0.5f;

        for (size_t i = 0; i < centerLabels.size(); ++i)
        {
            ImVec2 ts = ImGui::CalcTextSize(centerLabels[i].c_str());
            float y = startY + static_cast<float>(i) * lineH;
            if (y + ts.y * 0.5f > center.y + deadZoneRadius - 4.0f)
                break;
            drawList->AddText(
                ImVec2(center.x - ts.x * 0.5f, y - ts.y * 0.5f),
                IM_COL32(255, 255, 255, 255), centerLabels[i].c_str());
        }
    }
}

// ---- Drawing primitives ----

void RadialMenu::GenerateArc(float startAngle, float endAngle, float innerR, float outerR,
                             ImVector<ImVec2>& points) const
{
    const int segments = 32;
    points.clear();

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
}

void RadialMenu::DrawSector(ImDrawList* drawList, float startAngle, float endAngle,
                             float innerR, float outerR, ImU32 fillColor, ImU32 outlineCol)
{
    ImVector<ImVec2> points;
    GenerateArc(startAngle, endAngle, innerR, outerR, points);
    drawList->AddConcavePolyFilled(points.Data, points.Size, fillColor);

    for (int i = 0; i < points.Size; ++i)
        drawList->AddLine(points[i], points[(i + 1) % points.Size], outlineCol, 1.0f);
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
    DrawLabel(drawList, right ? ICON_FA_ANGLE_RIGHT : ICON_FA_ANGLE_LEFT, angleStart, angleEnd, innerR, outerR, color);
}

void RadialMenu::DrawTextureSector(ImDrawList* drawList, float startAngle, float endAngle,
                                    float innerR, float outerR, ImTextureID tex,
                                    ImU32 outlineCol)
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

        drawList->AddImageQuad(tex, p0, p1, p2, p3,
                               ImVec2(u0, 0), ImVec2(u1, 0), ImVec2(u1, 1), ImVec2(u0, 1),
                               IM_COL32_WHITE);
    }

    // Outline arcs
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
            drawList->AddLine(inner, outer, outlineCol, 1.0f);
    }
}

void RadialMenu::DrawInnerBorder(ImDrawList* drawList, float startAngle, float endAngle,
                                 float innerR, ImU32 color, float width)
{
    ImVector<ImVec2> points;
    GenerateArc(startAngle, endAngle, innerR, innerR + width, points);
    drawList->AddConcavePolyFilled(points.Data, points.Size, color);
}

void RadialMenu::DrawSectorRing(ImDrawList* drawList, int sectorCount, float innerR, float outerR,
                                 const std::function<std::string(int)>& labelForIndex,
                                 const std::function<ImU32(int)>& borderColorForIndex)
{
    float sectorAngle = TWO_PI / static_cast<float>(sectorCount);
    float angleOffset = -PI * 0.5f;

    for (int i = 0; i < sectorCount; ++i)
    {
        float startAngle = angleOffset - sectorAngle * 0.5f + static_cast<float>(i) * sectorAngle;
        float endAngle = startAngle + sectorAngle;

        DrawSector(drawList, startAngle, endAngle, innerR, outerR, backgroundColor, outlineColor);

        std::string label = labelForIndex(i);
        if (!label.empty())
            DrawLabel(drawList, label, startAngle, endAngle, innerR, outerR, IM_COL32(255, 255, 255, 255));

        ImU32 borderColor = borderColorForIndex(i);
        if (borderColor)
            DrawInnerBorder(drawList, startAngle, endAngle, innerR, borderColor, 10.0f);
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
