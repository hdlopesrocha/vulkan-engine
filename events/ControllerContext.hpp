#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// Identifies which physical controller a context belongs to.
#include "ControllerId.hpp"
#include "PageCategory.hpp"
#include "PageControl.hpp"
#include "ControllerPage.hpp"

// Holds the page tree for a single controller and tracks the active path
// (active top-level page + active subpage). The page tree and the navigation
// logic live entirely here so every controller publisher can reuse them; only
// the physical input mapping differs between controllers.
class ControllerContext {
public:
    explicit ControllerContext(ControllerId id) : id_(id) { buildDefaultTree(); }

    // Build the default page tree (Camera + Brush categories with subpages).
    void buildDefaultTree() {
        root_ = std::make_shared<ControllerPage>();
        root_->name = "Root";

        // Camera category
        auto camera = std::make_shared<ControllerPage>();
        camera->name = "Camera";
        camera->category = PageCategory::CAMERA;
        camera->control = PageControl::TRANSLATE;
        // A single Transform subpage combines translate + rotate.
        camera->addChild("Translate", PageControl::TRANSLATE);
        // Non-propagating page: raw mouse / input is handed to ImGui / UI.
        camera->addChild("UI", PageControl::UI, /*propagate=*/false);
        root_->children.push_back(camera);

        // Brush category (SDF brush). Translate, rotate and scale are combined
        // into a single Transform subpage; texture/attribute are separate.
        auto brush = std::make_shared<ControllerPage>();
        brush->name = "Brush";
        brush->category = PageCategory::BRUSH;
        brush->control = PageControl::TRANSLATE;
        brush->addChild("Control",    PageControl::CONTROL);
        brush->addChild("Texture",    PageControl::TEXTURE);
        brush->addChild("Attributes", PageControl::ATTRIBUTE);
        brush->addChild("Color",      PageControl::COLOR);
        root_->children.push_back(brush);

        // Light category
        auto light = std::make_shared<ControllerPage>();
        light->name = "Light";
        light->category = PageCategory::LIGHT;
        light->control = PageControl::AZIMUTH;
        light->addChild("Azimuth",   PageControl::AZIMUTH);
        light->addChild("Elevation", PageControl::ELEVATION);
        root_->children.push_back(light);

        pageIndex_ = 0;
        subpageIndex_ = 0;
    }

    // Navigation.
    void nextPage() {
        int n = static_cast<int>(root_->children.size());
        if (n == 0) return;
        pageIndex_ = (pageIndex_ + 1) % n;
        clampSubpage();
    }
    void prevPage() {
        int n = static_cast<int>(root_->children.size());
        if (n == 0) return;
        pageIndex_ = (pageIndex_ - 1 + n) % n;
        clampSubpage();
    }
    void nextSubpage() {
        const ControllerPage *p = activePage();
        if (!p || p->childCount() == 0) return;
        subpageIndex_ = (subpageIndex_ + 1) % p->childCount();
    }
    void prevSubpage() {
        const ControllerPage *p = activePage();
        if (!p || p->childCount() == 0) return;
        subpageIndex_ = (subpageIndex_ - 1 + p->childCount()) % p->childCount();
    }
    // Jump directly to the subpage exposing the given control (used to set
    // controller defaults, e.g. mouse -> UI so ImGui works out of the box).
    void selectControl(PageControl ctrl) {
        const ControllerPage *p = activePage();
        if (!p) return;
        for (int i = 0; i < p->childCount(); ++i) {
            if (p->child(i) && p->child(i)->control == ctrl) { subpageIndex_ = i; return; }
        }
    }
    // Switch to the page with the given category, then select the subpage.
    void selectPage(PageCategory cat, PageControl ctrl = PageControl::TRANSLATE) {
        for (int i = 0; i < static_cast<int>(root_->children.size()); ++i) {
            if (root_->children[i]->category == cat) {
                pageIndex_ = i;
                clampSubpage();
                selectControl(ctrl);
                return;
            }
        }
    }

    // Active path queries.
    const ControllerPage *activePage() const {
        if (pageIndex_ < 0 || pageIndex_ >= static_cast<int>(root_->children.size())) return nullptr;
        return root_->children[pageIndex_].get();
    }
    const ControllerPage *activeSubpage() const {
        const ControllerPage *p = activePage();
        if (!p) return nullptr;
        if (p->childCount() == 0) return p;
        int i = subpageIndex_;
        if (i < 0 || i >= p->childCount()) i = 0;
        return p->children[i].get();
    }
    PageCategory activeCategory() const {
        const ControllerPage *p = activePage();
        return p ? p->category : PageCategory::CAMERA;
    }
    PageControl activeControl() const {
        const ControllerPage *s = activeSubpage();
        return s ? s->control : PageControl::TRANSLATE;
    }
    // True when the active subpage must NOT emit application events.
    bool isNoPropagate() const {
        const ControllerPage *s = activeSubpage();
        return s ? !s->propagate : false;
    }

    std::string activePageName() const {
        const ControllerPage *p = activePage();
        return p ? p->name : "";
    }
    std::string activeSubpageName() const {
        const ControllerPage *s = activeSubpage();
        return s ? s->name : "";
    }

private:
    void clampSubpage() {
        const ControllerPage *p = activePage();
        if (!p) { subpageIndex_ = 0; return; }
        if (subpageIndex_ >= p->childCount()) subpageIndex_ = 0;
        if (subpageIndex_ < 0) subpageIndex_ = 0;
    }

    ControllerId id_;
    std::shared_ptr<ControllerPage> root_;
    int pageIndex_ = 0;
    int subpageIndex_ = 0;
};
