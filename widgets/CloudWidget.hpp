#pragma once

#include "Widget.hpp"
#include "CloudSettings.hpp"

class CloudWidget : public Widget {
public:
    explicit CloudWidget(CloudSettings& settings);
    void render() override;

private:
    CloudSettings& settings;
};
