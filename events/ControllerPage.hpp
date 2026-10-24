#pragma once

#include <memory>
#include <string>
#include <vector>

#include "PageCategory.hpp"
#include "PageControl.hpp"

// A node in the controller page tree. The tree is two levels deep:
//   root -> [category pages] -> [control subpages]
// Each controller owns its own tree so every controller can be on a different
// page / subpage independently.
struct ControllerPage {
    std::string name;
    PageCategory category = PageCategory::CAMERA;
    PageControl control = PageControl::TRANSLATE;
    // When false the publisher must NOT emit application events for this page.
    bool propagate = true;
    std::vector<std::shared_ptr<ControllerPage>> children;

    std::shared_ptr<ControllerPage> addChild(const std::string &name_, PageControl ctrl, bool propagate_ = true) {
        auto c = std::make_shared<ControllerPage>();
        c->name = name_;
        c->control = ctrl;
        c->category = category; // subpage inherits the parent category
        c->propagate = propagate_;
        children.push_back(c);
        return c;
    }
    int childCount() const { return static_cast<int>(children.size()); }
    const ControllerPage *child(int i) const {
        if (i < 0 || i >= static_cast<int>(children.size())) return nullptr;
        return children[i].get();
    }
};
