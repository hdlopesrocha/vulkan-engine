#include "QueueTimelineWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cmath>

static ImVec4 hsv(float h) {
    // Simple HSV->RGB with fixed saturation/value for distinct row colors.
    float s = 0.65f, v = 0.95f;
    float r, g, b;
    int i = static_cast<int>(h * 6.0f) % 6;
    float f = h * 6.0f - std::floor(h * 6.0f);
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    return ImVec4(r, g, b, 1.0f);
}

uint64_t QueueTimelineWidget::nowNs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

QueueTimelineWidget::QueueTimelineWidget(VulkanApp*) : Widget("Queue Timeline", "⏱") {
    rowsBuilt_ = false;
}

void QueueTimelineWidget::updateWithApp(VulkanApp* app) {
    if (!app) return;

    // Build the queue row list once (handles are stable for the app lifetime).
    if (!rowsBuilt_) {
        struct Cand { VkQueue h; const char* n; };
        std::vector<Cand> cands = {
            { app->getGraphicsQueue(),   "Graphics" },
            { app->getPresentQueue(),    "Present" },
            { app->getSolidQueue(),      "Solid" },
            { app->getWaterQueue(),      "Water" },
            { app->getVegetationQueue(), "Vegetation" },
            { app->getSdfQueue(),        "SDF" },
            { app->getBoundingBoxQueue(),"BoundingBox" },
            { app->geometryTransferQueue(), "Geometry" },
            { app->getTransferQueue(),   "Transfer" },
        };
        for (size_t i = 0; i < app->parallelGraphicsQueues.size(); ++i) {
            cands.push_back({ app->parallelGraphicsQueues[i],
                              ("Gfx" + std::to_string(i)).c_str() });
        }
        // List every logical queue as its own row (do NOT de-duplicate by
        // handle: on GPUs that expose a single graphics queue, Solid/Water/
        // Vegetation/SDF/BBox/Geometry all alias the same VkQueue, and we want
        // each to appear so the user can see the full logical-queue inventory).
        // Aliased rows naturally light up together since they share a handle.
        float hue = 0.0f;
        for (auto& c : cands) {
            if (c.h == VK_NULL_HANDLE) continue;
            rows_.push_back({ c.h, c.n, hsv(hue) });
            hue += 0.13f;
            if (hue >= 1.0f) hue -= 1.0f;
        }
        rowsBuilt_ = true;
    }

    app->getQueueTimeline(segments_);
}

void QueueTimelineWidget::render() {
    if (!ImGui::Begin(title.c_str())) {
        ImGui::End();
        return;
    }

    if (rows_.empty()) {
        ImGui::Text("No queue activity captured yet.");
        ImGui::End();
        return;
    }
    if (segments_.empty()) {
        ImGui::Text("No submissions recorded yet.");
        ImGui::End();
        return;
    }

    // Map handle -> all row indices (rows may alias the same VkQueue handle).
    std::unordered_map<VkQueue, std::vector<int>> rowsForHandle;
    for (size_t i = 0; i < rows_.size(); ++i)
        rowsForHandle[rows_[i].handle].push_back(static_cast<int>(i));

    ImGui::Checkbox("Per-frame slots", &perFrameMode_);
    if (!perFrameMode_) {
        ImGui::SameLine();
        ImGui::SliderFloat("Window (ms)", &windowMs_, 10.0f, 250.0f);
        ImGui::SameLine();
        ImGui::Text("rolling %.0f ms window", windowMs_);
    }

    const float labelW = 92.0f;
    const float cellH = 15.0f;
    const float plotW = std::max(200.0f, ImGui::GetContentRegionAvail().x - labelW);

    // Draw one queue row as PRECISE rectangles (no slot quantization): each
    // submission maps to an exact x-interval [startNs,endNs] on the time axis.
    // Completed submissions are drawn solid; still-in-flight submissions
    // (endNs == 0) are drawn dimmed, extending to wEnd ("now"), so they are
    // visible but clearly distinguishable from resolved work.
    // frameFilter == UINT64_MAX means "show all frames" (rolling mode).
    auto drawRow = [&](int r, uint64_t wStart, uint64_t wEnd, uint64_t span, uint64_t frameFilter) {
        ImGui::AlignTextToFramePadding();
        ImGui::Text("%s", rows_[r].name.c_str());
        ImGui::SameLine(labelW);
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1 = ImVec2(p0.x + plotW, p0.y + cellH);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p0, p1, IM_COL32(28, 28, 30, 255));
        // Faint time gridlines for reference (not quantization — purely visual).
        const int grid = 12;
        for (int g = 1; g < grid; ++g) {
            float gx = p0.x + (float)g / (float)grid * plotW;
            dl->AddLine(ImVec2(gx, p0.y), ImVec2(gx, p1.y), IM_COL32(58, 58, 64, 255));
        }
        const int cr = (int)(rows_[r].color.x * 255);
        const int cg = (int)(rows_[r].color.y * 255);
        const int cb = (int)(rows_[r].color.z * 255);
        for (auto& s : segments_) {
            if (frameFilter != UINT64_MAX && s.frame != frameFilter) continue;
            if (s.queue != rows_[r].handle) continue;
            bool inflight = (s.endNs == 0);
            uint64_t e = inflight ? wEnd : s.endNs;
            if (e < wStart || s.startNs > wEnd) continue;
            float x0 = p0.x + (float)((double)(s.startNs - wStart) / (double)span) * plotW;
            float x1 = p0.x + (float)((double)(e - wStart) / (double)span) * plotW;
            x0 = std::clamp(x0, p0.x, p1.x);
            x1 = std::clamp(x1, p0.x, p1.x);
            if (x1 - x0 < 1.0f) x1 = x0 + 1.0f;
            ImU32 col = inflight ? IM_COL32(cr, cg, cb, 90) : IM_COL32(cr, cg, cb, 255);
            dl->AddRectFilled(ImVec2(x0, p0.y + 1.0f), ImVec2(x1, p1.y - 1.0f), col);
        }
        ImGui::Dummy(ImVec2(plotW, cellH));
    };

    if (perFrameMode_) {
        // ---- Per-frame precise timeline (one row per queue) -----------
        // Show the LATEST frame. Completed submissions are solid; still-in-flight
        // ones (fence not signaled yet) are drawn dimmed up to "now" so the view
        // stays populated and honest instead of looking 100% busy or empty.
        uint64_t latest = 0;
        for (auto& s : segments_) latest = std::max(latest, s.frame);
        latestFrame_ = latest;

        uint64_t now = nowNs();
        uint64_t fStart = UINT64_MAX, fEnd = 0;
        for (auto& s : segments_) {
            if (s.frame != latest) continue;
            if (s.startNs < fStart) fStart = s.startNs;
            uint64_t e = s.endNs ? s.endNs : now;
            if (e > fEnd) fEnd = e;
        }
        if (fEnd <= fStart || (fEnd - fStart) < 2000) {
            ImGui::Text("Frame %llu has no submissions yet.", (unsigned long long)latest);
            ImGui::End();
            return;
        }
        const uint64_t span = fEnd - fStart;
        ImGui::Text("Frame %llu   span %.3f ms   (solid = done, dim = in-flight)",
                    (unsigned long long)latest, span / 1e6);
        for (size_t r = 0; r < rows_.size(); ++r)
            drawRow((int)r, fStart, fEnd, span, latest);

        // Parallelism strip: overlap count per slot (in-flight counted up to now).
        ImGui::Separator();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Parallel");
        const int slots = 72;
        std::vector<int> parallel(slots, 0);
        for (auto& s : segments_) {
            if (s.frame != latest) continue;
            auto it = rowsForHandle.find(s.queue);
            if (it == rowsForHandle.end()) continue;
            uint64_t e = s.endNs ? s.endNs : now;
            long long a = (long long)(s.startNs - fStart) * slots / (long long)span;
            long long b = (long long)(e - fStart) * slots / (long long)span;
            int s0 = (int)std::clamp<long long>(a, 0, slots - 1);
            int s1 = (int)std::clamp<long long>(b, 0, slots - 1);
            if (s1 < s0) s1 = s0;
            for (int k = s0; k <= s1; ++k) parallel[k]++;
        }
        int maxP = 1;
        for (int k = 0; k < slots; ++k) maxP = std::max(maxP, parallel[k]);
        ImGui::SameLine(labelW);
        ImVec2 pp0 = ImGui::GetCursorScreenPos();
        ImVec2 pp1 = ImVec2(pp0.x + plotW, pp0.y + 10.0f);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(pp0, pp1, IM_COL32(28, 28, 30, 255));
        for (int k = 0; k < slots; ++k) {
            float fx0 = pp0.x + (float)k / (float)slots * plotW;
            float fx1 = pp0.x + (float)(k + 1) / (float)slots * plotW;
            float f = (float)parallel[k] / (float)maxP;
            ImVec4 c(f, 0.9f * (1.0f - f) + 0.1f, 0.3f, 1.0f);
            dl->AddRectFilled(ImVec2(fx0, pp0.y), ImVec2(fx1, pp1.y),
                              IM_COL32((int)(c.x * 255), (int)(c.y * 255), (int)(c.z * 255), 255));
        }
        ImGui::Dummy(ImVec2(plotW, 10.0f));
        ImGui::SameLine();
        ImGui::Text("queues busy concurrently / time");
    } else {
        // ---- Rolling precise Gantt over a fixed time window ----------
        uint64_t now = nowNs();
        uint64_t wEnd = now;
        uint64_t wStart = now - (uint64_t)(windowMs_ * 1e6);
        if (wEnd <= wStart) wEnd = wStart + 1;
        const uint64_t span = wEnd - wStart;
        ImGui::Text("Rolling window %.1f ms   (rows = queues, x = time)", windowMs_);
        for (size_t r = 0; r < rows_.size(); ++r)
            drawRow((int)r, wStart, wEnd, span, UINT64_MAX);
    }

    ImGui::End();
}
