#pragma once

#include "ControllerParameters.hpp"

class ControllerManager {
public:
    ControllerManager() = default;

    // Owned ControllerParameters instance
    ControllerParameters parameters;

    ControllerParameters* getParameters() { return &parameters; }
    const ControllerParameters* getParameters() const { return &parameters; }

    ControllerParameters::pageType getCurrentPage() const { return parameters.currentPage; }
    void setCurrentPage(ControllerParameters::pageType p) { parameters.currentPage = p; }
};
