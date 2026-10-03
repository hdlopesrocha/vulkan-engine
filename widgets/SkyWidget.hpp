#pragma once

#include "Widget.hpp"
#include "SkySettings.hpp"

class SkyWidget : public Widget {
private:
    // Reference to centralized settings owned externally (e.g., MyApp)
    SkySettings &settings;

public:
    explicit SkyWidget(SkySettings &settings);

    void render() override;
};
