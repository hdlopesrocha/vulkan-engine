#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "Event.hpp"
#include "IEventHandler.hpp"

// Identifies which physical controller a context / navigation event belongs to.
enum class ControllerId { KEYBOARD, MOUSE, GAMEPAD, WIIMOTE };

// Top-level category of control. The active category decides whether an input
// acts on the camera or on the selected SDF brush.
enum class PageCategory { CAMERA, BRUSH };

// Sub-control selected inside a category. This is what a given input axis /
// button maps to for the active category. UI pages are non-propagating: the
// publisher must not emit application events while on them so that ImGui /
// other UI can consume the raw input without interference.
enum class PageControl {
    TRANSLATE,   // move position (camera or brush)
    ROTATE,      // change orientation (yaw/pitch/roll)
    SCALE,       // change size
    TEXTURE,     // change material / texture
    ATTRIBUTE,   // change SDF-specific attributes
    UI           // non-propagating passthrough (ImGui / UI)
};

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

    std::shared_ptr<ControllerPage> addChild(const std::string &name, PageControl ctrl, bool propagate = true) {
        auto c = std::make_shared<ControllerPage>();
        c->name = name;
        c->control = ctrl;
        c->category = category; // subpage inherits the parent category
        c->propagate = propagate;
        children.push_back(c);
        return c;
    }
    int childCount() const { return static_cast<int>(children.size()); }
    const ControllerPage *child(int i) const {
        if (i < 0 || i >= static_cast<int>(children.size())) return nullptr;
        return children[i].get();
    }
};

// Event used to switch pages. Publishing it lets ANY controller (e.g. the
// keyboard) switch the pages of ANY other controller (e.g. the mouse). Target
// contexts react in their IEventHandler::onEvent.
class PageNavigationEvent : public Event {
public:
    enum class Action { NEXT_PAGE, PREV_PAGE, NEXT_SUBPAGE, PREV_SUBPAGE };

    PageNavigationEvent(ControllerId target_, Action action_)
        : target(target_), action(action_) {}

    std::string name() const override { return "PageNavigationEvent"; }

    ControllerId target;
    Action action;
};

// Holds the page tree for a single controller and tracks the active path
// (active top-level page + active subpage). The page tree and the navigation
// logic live entirely here so every controller publisher can reuse them; only
// the physical input mapping differs between controllers.
class ControllerContext : public IEventHandler {
public:
    explicit ControllerContext(ControllerId id) : id_(id) { buildDefaultTree(); }

    ControllerId id() const { return id_; }

    // Build the default page tree (Camera + Brush categories with subpages).
    void buildDefaultTree() {
        root_ = std::make_shared<ControllerPage>();
        root_->name = "Root";

        // Camera category
        auto camera = std::make_shared<ControllerPage>();
        camera->name = "Camera";
        camera->category = PageCategory::CAMERA;
        camera->control = PageControl::TRANSLATE;
        camera->addChild("Translate", PageControl::TRANSLATE);
        camera->addChild("Rotate", PageControl::ROTATE);
        // Non-propagating page: raw mouse / input is handed to ImGui / UI.
        camera->addChild("UI", PageControl::UI, /*propagate=*/false);
        root_->children.push_back(camera);

        // Brush category (SDF brush). Mouse manipulation is intentionally not
        // implemented yet, but the pages exist so the tree is reusable.
        auto brush = std::make_shared<ControllerPage>();
        brush->name = "Brush";
        brush->category = PageCategory::BRUSH;
        brush->control = PageControl::TRANSLATE;
        brush->addChild("Translate", PageControl::TRANSLATE);
        brush->addChild("Rotate", PageControl::ROTATE);
        brush->addChild("Scale", PageControl::SCALE);
        brush->addChild("Texture", PageControl::TEXTURE);
        brush->addChild("Attributes", PageControl::ATTRIBUTE);
        root_->children.push_back(brush);

        pageIndex_ = 0;
        subpageIndex_ = 0;
    }

    // Navigation (also invoked from PageNavigationEvent).
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
    void applyAction(PageNavigationEvent::Action a) {
        switch (a) {
            case PageNavigationEvent::Action::NEXT_PAGE: nextPage(); break;
            case PageNavigationEvent::Action::PREV_PAGE: prevPage(); break;
            case PageNavigationEvent::Action::NEXT_SUBPAGE: nextSubpage(); break;
            case PageNavigationEvent::Action::PREV_SUBPAGE: prevSubpage(); break;
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
    int activePageIndex() const { return pageIndex_; }
    int activeSubpageIndex() const { return subpageIndex_; }

    // IEventHandler: respond to PageNavigationEvent targeting this context.
    void onEvent(const EventPtr &event) override {
        if (!event) return;
        auto *nav = dynamic_cast<PageNavigationEvent *>(event.get());
        if (!nav) return;
        if (nav->target != id_) return;
        applyAction(nav->action);
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
