#pragma once

#include "utils/Brush3dEntry.hpp"
#include <vector>
#include <algorithm>

class Brush3dManager {
public:
    Brush3dManager() = default;

    // Entries owned by the manager
    std::vector<BrushEntry> entries;

    // Index of currently-selected entry (0-based). If entries is empty this is 0.
    int selectedIndex = 0;

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

    void next() { if (!entries.empty() && selectedIndex + 1 < static_cast<int>(entries.size())) ++selectedIndex; }
    void prev() { if (selectedIndex > 0) --selectedIndex; }
};
