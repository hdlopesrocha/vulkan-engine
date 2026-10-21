#include "VideoTextureWidget.hpp"

#include "../vulkan/resources/TextureArrayManager.hpp"
#include "components/ColumnLayout.hpp"
#include "components/ImGuiHelpers.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {
constexpr float kPreviewSize = 256.0f;
constexpr float kPreviewColumnWidth = 300.0f;

std::string formatSeconds(double seconds) {
    const int total = static_cast<int>(std::max(0.0, seconds));
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%02d:%02d.%02d", total / 60, total % 60,
                  static_cast<int>((seconds - static_cast<double>(total)) * 100.0));
    return buffer;
}
}  // namespace

VideoTextureWidget::VideoTextureWidget(std::shared_ptr<VideoTextureService> videos_)
    : Widget("Video Textures", u8"\uf03d"), videos(std::move(videos_)) {
    // The panel is the primary video inspector: open it on first run.
    isOpen = true;
}

void VideoTextureWidget::render() {
    if (!videos) return;
    if (!scanned) {
        available = VideoTextureService::listAvailableVideos();
        scanned = true;
    }
    // Opening the inspector is an explicit request to see the clips. Start
    // them (and loop) once even when VULKAN_VIDEO_AUTOPLAY disabled autoplay.
    if (!playbackPrimed) {
        videos->setPlayingAll(true);
        for (size_t i = 0; i < videos->count(); ++i) videos->setLooping(i, true);
        playbackPrimed = true;
    }

    ImGui::SetNextWindowPos(ImVec2(0, 24), ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(1280, 680), ImGuiCond_Appearing);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    const size_t count = videos->count();
    if (count == 0) {
        ImGui::TextUnformatted("No video textures loaded.");
        ImGui::Separator();
        if (ImGui::Button("Rescan clips")) available = VideoTextureService::listAvailableVideos();
        return;
    }
    if (selected < 0 || static_cast<size_t>(selected) >= count) selected = 0;

    const uint32_t capacity = videos->slotCapacity();
    const auto current = videos->state(static_cast<size_t>(selected));

    std::vector<ImGuiComponents::ColumnSection> sections;
    sections.reserve(5);

    // ── Live preview + selected-clip controls ─────────────────────────────
    sections.push_back({[&]() {
        ImGui::Text("Preview");
        ImGuiComponents::ColSeparator(kPreviewColumnWidth);
        ImGui::Text("%s  (slot %u)", current.name.c_str(), current.slot);
        const ImTextureID preview = videos->previewTexture(current.slot);
        ImGuiHelpers::ImageOrUnavailable(preview, ImVec2(kPreviewSize, kPreviewSize),
                                         "No preview yet (slot empty or upload pending)");
        ImGui::TextDisabled("live slot %u  frame %s", current.slot,
                            formatSeconds(current.frameTime).c_str());

        bool playing = current.playing;
        if (ImGuiComponents::CheckboxField("playing", &playing, "Start/stop this clip.")) {
            videos->setPlaying(static_cast<size_t>(selected), playing);
        }
        ImGui::SameLine();
        bool looping = current.looping;
        if (ImGuiComponents::CheckboxField("loop", &looping, "Restart at the end of the clip.")) {
            videos->setLooping(static_cast<size_t>(selected), looping);
        }
        if (!current.playing) {
            ImGui::SameLine();
            if (ImGui::Button("Play")) videos->setPlaying(static_cast<size_t>(selected), true);
            ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.15f, 1.0f),
                               "Clip is paused: the preview stays on its current frame.");
        }
        float itemRate = current.rate;
        ImGui::SetNextItemWidth(kPreviewColumnWidth);
        if (ImGui::SliderFloat("##clip_rate", &itemRate, -4.0f, 4.0f, "rate %.2fx")) {
            videos->setRate(static_cast<size_t>(selected), itemRate);
        }
        float position = static_cast<float>(current.clock);
        ImGui::SetNextItemWidth(kPreviewColumnWidth);
        if (ImGui::SliderFloat("##clip_pos", &position, 0.0f,
                               static_cast<float>(std::max(0.001, current.duration)), "position %.2f s")) {
            videos->seek(static_cast<size_t>(selected), static_cast<double>(position));
        }
        ImGui::TextDisabled("material: kind Video, textureIndex %u", current.slot);
        if (ImGui::Button("Unload clip")) {
            videos->unloadVideo(current.slot);
            selected = 0;
        }
    }, kPreviewColumnWidth});

    // ── Transport ─────────────────────────────────────────────────────────
    sections.push_back({[&]() {
        ImGui::Text("Transport");
        ImGuiComponents::ColSeparator();
        ImGui::Text("%zu clip(s) loaded into %u slot(s)", count, capacity);
        if (ImGui::Button("Play all")) videos->setPlayingAll(true);
        ImGui::SameLine();
        if (ImGui::Button("Pause all")) videos->setPlayingAll(false);
        if (ImGuiComponents::CheckboxField("Loop all", &loopAll,
                "Enable looping on every loaded clip.")) {
            for (size_t i = 0; i < count; ++i) videos->setLooping(i, loopAll);
        }
        if (ImGuiComponents::SliderFloatField("Rate", &rate, -4.0f, 4.0f, "%.2fx",
                "Playback rate applied to every clip.")) {
            for (size_t i = 0; i < count; ++i) videos->setRate(i, rate);
        }
        ImGuiComponents::FieldLabel("Shared clock", "Seek every clip to this timestamp.");
        ImGui::SetNextItemWidth(ImGuiComponents::kColumnWidth);
        ImGui::DragFloat("##shared_clock", &seekSeconds, 0.05f, 0.0f, 3600.0f, "%.2f s");
        if (ImGui::Button("Seek all")) videos->seekAll(static_cast<double>(seekSeconds));
        ImGui::SameLine();
        if (ImGui::Button("Copy current")) {
            seekSeconds = static_cast<float>(current.clock);
            videos->seekAll(current.clock);
        }
        if (ImGui::Button("Log diagnostics")) videos->logDiagnostics();
    }});

    // ── Loaded clips ──────────────────────────────────────────────────────
    sections.push_back({[&]() {
        ImGui::Text("Loaded clips");
        ImGuiComponents::ColSeparator();
        for (size_t i = 0; i < count; ++i) {
            const auto s = videos->state(i);
            ImGui::PushID(static_cast<int>(i));
            char label[192];
            std::snprintf(label, sizeof(label), "%s  [slot %u]%s##row", s.name.c_str(), s.slot,
                          s.playing ? "" : "  (paused)");
            if (ImGui::Selectable(label, selected == static_cast<int>(i))) selected = static_cast<int>(i);
            if (!s.playing) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.15f, 1.0f), "paused");
            }
            ImGui::PopID();
        }
        ImGui::TextDisabled("Select a clip to drive the preview and controls.");
    }});

    // ── Decoder details ───────────────────────────────────────────────────
    sections.push_back({[&]() {
        ImGui::Text("Details");
        ImGuiComponents::ColSeparator();
        ImGui::Text("%s", current.name.c_str());
        ImGui::Text("slot %u  %s  %dx%d  %.1f fps  %s", current.slot, current.codec.c_str(),
                    current.width, current.height, current.fps, current.backend.c_str());
        ImGui::Text("clock %s / %s   frame %s%s",
                    formatSeconds(current.clock).c_str(), formatSeconds(current.duration).c_str(),
                    formatSeconds(current.frameTime).c_str(),
                    current.decodePending ? "  (decoding)" : "");
        ImGui::Text("decoded %llu  uploaded %llu  seeks %llu  cache hits %llu",
                    static_cast<unsigned long long>(current.framesDecoded),
                    static_cast<unsigned long long>(current.framesUploaded),
                    static_cast<unsigned long long>(current.seeks),
                    static_cast<unsigned long long>(current.cacheHits));
        ImGui::Text("avg decode %.2f ms  avg seek %.2f ms  last upload %.2f ms",
                    current.avgDecodeMs, current.avgSeekMs, current.lastUploadMs);
        ImGui::Text("yuv %.1f MiB  staging %.1f MiB  source %.1f MiB  naive RGBA %.1f MiB",
                    current.gpuYuvBytes / 1048576.0, current.stagingBytes / 1048576.0,
                    static_cast<double>(current.compressedBytes) / 1048576.0,
                    static_cast<double>(current.frameCount) * 1024.0 * 1024.0 * 4.0 / 1048576.0);
    }});

    // ── Clip library ──────────────────────────────────────────────────────
    sections.push_back({[&]() {
        ImGui::Text("Clip library");
        ImGuiComponents::ColSeparator();
        if (ImGui::Button("Rescan clips")) available = VideoTextureService::listAvailableVideos();
        if (available.empty()) {
            ImGui::TextDisabled("No clips found. Add converted clips to videos/base or videos/extra.");
        } else if (ImGui::BeginTable("video-library", 2,
                                     ImGuiTableFlags_SizingStretchProp |
                                         ImGuiTableFlags_BordersInnerV |
                                         ImGuiTableFlags_RowBg,
                                     ImVec2(ImGuiComponents::kColumnWidth, 0.0f))) {
            ImGui::TableSetupColumn("clip", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("##load", ImGuiTableColumnFlags_WidthFixed, 52.0f);
            ImGui::TableHeadersRow();
            for (const auto& clip : available) {
                bool loaded = false;
                for (size_t i = 0; i < count; ++i) {
                    const auto s = videos->state(i);
                    if (s.path == clip.path || s.name == clip.name) { loaded = true; break; }
                }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(clip.name.c_str());
                ImGui::TextDisabled("%s", clip.category.c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::PushID(clip.path.c_str());
                ImGui::BeginDisabled(loaded || count >= capacity);
                if (ImGui::SmallButton("Load")) {
                    const int slot = videos->loadVideo(clip.path, clip.name);
                    if (slot >= 0) {
                        const auto states = videos->states();
                        for (size_t i = 0; i < states.size(); ++i) {
                            if (states[i].slot == static_cast<uint32_t>(slot)) {
                                videos->setLooping(i, true);
                                videos->setPlaying(i, true);
                                selected = static_cast<int>(i);
                                break;
                            }
                        }
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
    }});

    static std::vector<float> cachedH;
    const std::vector<float> estimates = {
        430.0f,                                   // Preview (image + controls)
        ImGuiComponents::EstimateSectionHeight(8), // Transport
        ImGuiComponents::EstimateSectionHeight(10),// Loaded clips
        ImGuiComponents::EstimateSectionHeight(9), // Details
        ImGuiComponents::EstimateSectionHeight(12),// Library
    };
    ImGuiComponents::LayoutSections(sections, cachedH, estimates);
}
