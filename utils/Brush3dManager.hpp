#pragma once

#include "Brush3dEntry.hpp"
#include <vector>
#include <algorithm>

// Active brush control mode. The radial menu and Brush3dWidget both use this
// to keep the selected mode in sync.
enum class BrushControlMode {
    TRANSLATE,
    AIM,
    SCALE,
    TEXTURE,
    ATTRIBUTE,
    COLOR,
    MODE_COUNT
};

// Active brush paint mode (Add / Remove / Paint).
enum class BrushPaintMode {
    ADD,
    REMOVE,
    PAINT,
    MODE_COUNT
};

// Brush interaction mode (Drag vs Click).
enum class BrushDragMode {
    DRAG,
    CLICK,
    MODE_COUNT
};

class Brush3dManager {
public:
    Brush3dManager() = default;

    // Entries owned by the manager
    std::vector<BrushEntry> entries;

    // Index of currently-selected entry (0-based). If entries is empty this is 0.
    int selectedIndex = 0;

    // Active brush control mode (shared across all controllers)
    BrushControlMode controlMode = BrushControlMode::TRANSLATE;

    // Active brush paint mode (shared across all controllers)
    BrushPaintMode paintMode = BrushPaintMode::ADD;

    // Brush interaction mode (shared across all controllers)
    BrushDragMode dragMode = BrushDragMode::DRAG;

    void addEntry() {
        entries.emplace_back();
        selectedIndex = static_cast<int>(entries.size()) - 1;
    }

    void removeLast() {
        if (entries.empty()) return;
        entries.pop_back();
        if (entries.empty()) selectedIndex = 0;
        else selectedIndex = std::min(selectedIndex, static_cast<int>(entries.size()) - 1);
    }

    void removeAt(int idx) {
        if (idx < 0 || idx >= static_cast<int>(entries.size())) return;
        entries.erase(entries.begin() + idx);
        if (entries.empty()) selectedIndex = 0;
        else selectedIndex = std::min(selectedIndex, static_cast<int>(entries.size()) - 1);
    }

    std::vector<BrushEntry>& getEntries() { return entries; }
    const std::vector<BrushEntry>& getEntries() const { return entries; }

    int getSelectedIndex() const { return selectedIndex; }
    void setSelectedIndex(int idx) {
        if (entries.empty()) { selectedIndex = 0; return; }
        if (idx < 0) idx = 0;
        if (idx >= static_cast<int>(entries.size())) idx = static_cast<int>(entries.size()) - 1;
        selectedIndex = idx;
    }

    BrushEntry* getSelectedEntry() {
        if (entries.empty()) return nullptr;
        return &entries[selectedIndex];
    }
    const BrushEntry* getSelectedEntry() const {
        if (entries.empty()) return nullptr;
        return &entries[selectedIndex];
    }

    // HSV helpers (hsv.x=Hue 0..360, hsv.y=Saturation 0..1, hsv.z=Value 0..1)
    float getHue() const {
        const BrushEntry* e = getSelectedEntry();
        return e ? e->hsv.x : 0.0f;
    }
    float getSaturation() const {
        const BrushEntry* e = getSelectedEntry();
        return e ? e->hsv.y * 100.0f : 0.0f;
    }
    float getValue() const {
        const BrushEntry* e = getSelectedEntry();
        return e ? e->hsv.z * 100.0f : 0.0f;
    }

    void next() { if (!entries.empty() && selectedIndex + 1 < static_cast<int>(entries.size())) ++selectedIndex; }
    void prev() { if (selectedIndex > 0) --selectedIndex; }
};
