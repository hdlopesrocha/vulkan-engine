#include "VideoTextureWidget.hpp"

#include "../vulkan/resources/TextureArrayManager.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace {
constexpr float kPreviewSize = 256.0f;

std::string formatSeconds(double seconds) {
    const int total = static_cast<int>(std::max(0.0, seconds));
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%02d:%02d.%02d", total / 60, total % 60,
                  static_cast<int>((seconds - static_cast<double>(total)) * 100.0));
    return buffer;
}
}  // namespace

VideoTextureWidget::VideoTextureWidget(std::shared_ptr<VideoTextureService> videos_,
                                       const char* title_)
    : Widget(title_, u8"\uf03d"), videos(std::move(videos_)) {}

void VideoTextureWidget::render() {
    if (!videos) return;
    if (!scanned) {
        available = VideoTextureService::listAvailableVideos();
        scanned = true;
    }

    const size_t count = videos->count();
    const uint32_t capacity = videos->slotCapacity();
    ImGui::Text("%zu clip(s) loaded into %u video slot(s)", count, capacity);
    ImGui::SameLine();
    if (ImGui::Button("Rescan clips")) {
        available = VideoTextureService::listAvailableVideos();
    }
    ImGui::Separator();

    // ── Load more clips ─────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Load clips", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (available.empty()) {
            ImGui::TextDisabled("No clips found. Drop files in videos/base or videos/extra and run "
                                "`make videos` / `make videos-extra`.");
        } else if (ImGui::BeginTable("video-available", 4,
                                     ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV |
                                         ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("clip", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("library", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("state", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("action", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableHeadersRow();
            for (const auto& clip : available) {
                bool loaded = false;
                for (size_t i = 0; i < count; ++i) {
                    const auto state = videos->state(i);
                    if (state.path == clip.path || state.name == clip.name) { loaded = true; break; }
                }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(clip.name.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(clip.category.c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(loaded ? "loaded" : "-");
                ImGui::TableSetColumnIndex(3);
                ImGui::PushID(clip.path.c_str());
                ImGui::BeginDisabled(loaded || count >= capacity);
                if (ImGui::SmallButton("Load")) {
                    const int slot = videos->loadVideo(clip.path, clip.name);
                    if (slot >= 0) {
                        // A clip loaded interactively starts playing (and loops)
                        // immediately; the row is selected so the preview shows it.
                        const auto loaded = videos->states();
                        for (size_t i = 0; i < loaded.size(); ++i) {
                            if (loaded[i].slot == static_cast<uint32_t>(slot)) {
                                videos->setLooping(i, true);
                                videos->setPlaying(i, true);
                                selected = static_cast<int>(i);
                                break;
                            }
                        }
                        std::printf("[VideoTextureWidget] loaded %s into slot %d\n", clip.name.c_str(), slot);
                    }
                }
                ImGui::EndDisabled();
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        if (count >= capacity) {
            ImGui::TextDisabled("Video array full: unload a clip or raise VULKAN_VIDEO_SLOTS.");
        }
    }

    if (count == 0) {
        ImGui::Separator();
        ImGui::TextUnformatted("No video textures loaded.");
        return;
    }

    ImGui::Separator();

    // ── Global transport: a shared clock keeps several clips synchronised ──
    if (ImGui::Button("Play all")) videos->setPlayingAll(true);
    ImGui::SameLine();
    if (ImGui::Button("Pause all")) videos->setPlayingAll(false);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    ImGui::DragFloat("rate##all", &rate, 0.01f, -4.0f, 4.0f, "%.2fx");
    ImGui::SameLine();
    if (ImGui::Button("Apply rate")) {
        for (size_t i = 0; i < count; ++i) videos->setRate(i, rate);
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("loop all", &loopAll)) {
        for (size_t i = 0; i < count; ++i) videos->setLooping(i, loopAll);
    }
    ImGui::SetNextItemWidth(220.0f);
    ImGui::DragFloat("shared clock", &seekSeconds, 0.05f, 0.0f, 3600.0f, "%.2f s");
    ImGui::SameLine();
    if (ImGui::Button("Seek all")) videos->seekAll(static_cast<double>(seekSeconds));
    ImGui::SameLine();
    if (ImGui::Button("Copy current")) {
        const auto states = videos->states();
        if (!states.empty()) {
            seekSeconds = static_cast<float>(states[0].clock);
            videos->seekAll(states[0].clock);
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Log diagnostics")) videos->logDiagnostics();

    if (ImGui::BeginTable("video-list", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("list", ImGuiTableColumnFlags_WidthFixed, 180.0f);
        ImGui::TableSetupColumn("details", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        for (size_t i = 0; i < count; ++i) {
            const auto state = videos->state(i);
            char label[160];
            std::snprintf(label, sizeof(label), "%s##%zu", state.name.c_str(), i);
            if (ImGui::Selectable(label, selected == static_cast<int>(i))) selected = static_cast<int>(i);
        }
        ImGui::TableSetColumnIndex(1);

        const size_t index = static_cast<size_t>(std::clamp(selected, 0, static_cast<int>(count) - 1));
        const auto s = videos->state(index);
        ImGui::Text("%s", s.name.c_str());
        ImGui::Text("slot %u  %s  %dx%d  %.1f fps  %s", s.slot, s.codec.c_str(), s.width, s.height, s.fps,
                    s.backend.c_str());
        ImGui::Text("clock %s / %s   frame %s%s", formatSeconds(s.clock).c_str(),
                    formatSeconds(s.duration).c_str(), formatSeconds(s.frameTime).c_str(),
                    s.decodePending ? "  (decoding)" : "");
        ImGui::Text("decoded %llu  uploaded %llu  seeks %llu  cache hits %llu",
                    static_cast<unsigned long long>(s.framesDecoded),
                    static_cast<unsigned long long>(s.framesUploaded),
                    static_cast<unsigned long long>(s.seeks),
                    static_cast<unsigned long long>(s.cacheHits));
        ImGui::Text("avg decode %.2f ms  avg seek %.2f ms  last upload %.2f ms", s.avgDecodeMs, s.avgSeekMs,
                    s.lastUploadMs);
        ImGui::Text("yuv %.1f MiB  staging %.1f MiB  source %.1f MiB  naive RGBA %.1f MiB", s.gpuYuvBytes / 1048576.0,
                    s.stagingBytes / 1048576.0, static_cast<double>(s.compressedBytes) / 1048576.0,
                    static_cast<double>(s.frameCount) * 1024.0 * 1024.0 * 4.0 / 1048576.0);

        // Live preview of the slot's current frame (the same texture the video
        // materials sample; updates while playing or after a seek).
        const ImTextureID preview = videos->previewTexture(s.slot);
        if (preview) {
            ImGui::Spacing();
            ImGui::Image(preview, ImVec2(kPreviewSize, kPreviewSize));
            ImGui::TextDisabled("slot %u live preview (frame %s)", s.slot,
                                formatSeconds(s.frameTime).c_str());
        } else {
            ImGui::TextDisabled("slot %u has no preview yet", s.slot);
        }

        bool playing = s.playing;
        if (ImGui::Checkbox("playing", &playing)) videos->setPlaying(index, playing);
        ImGui::SameLine();
        bool looping = s.looping;
        if (ImGui::Checkbox("loop", &looping)) videos->setLooping(index, looping);
        float itemRate = s.rate;
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::SliderFloat("rate", &itemRate, -4.0f, 4.0f, "%.2fx")) videos->setRate(index, itemRate);
        float position = static_cast<float>(s.clock);
        ImGui::SetNextItemWidth(240.0f);
        if (ImGui::SliderFloat("position", &position, 0.0f, static_cast<float>(std::max(0.001, s.duration)), "%.2f s")) {
            videos->seek(index, static_cast<double>(position));
        }
        ImGui::TextDisabled("material: kind Video, textureIndex %u, blend mode from the material", s.slot);
        if (ImGui::Button("Unload clip")) {
            videos->unloadVideo(s.slot);
            selected = 0;
        }
        ImGui::EndTable();
    }
}
