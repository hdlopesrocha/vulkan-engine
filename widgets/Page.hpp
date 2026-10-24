#pragma once

#include <string>
#include <vector>

#include "SubPage.hpp"

struct Page
{
    std::string label;      // icon symbol (displayed in sector)
    std::string textLabel;  // text (displayed in center circle)
    std::vector<SubPage> subPages;
};
