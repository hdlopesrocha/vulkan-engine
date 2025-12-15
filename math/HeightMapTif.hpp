#ifndef HEIGHT_MAP_TIF_HPP
#define HEIGHT_MAP_TIF_HPP

#include "HeightFunction.hpp"
#include "BoundingBox.hpp"
#include <string>
#include <vector>

class HeightMapTif : public HeightFunction {
public:
    std::vector<std::vector<float>> data;
    std::vector<int16_t> data1;
    BoundingBox box;
    int width;
    int height;
    int sizePerTile;
    HeightMapTif(const std::string & filename, BoundingBox box, int sizePerTile, float verticalScale, float verticalShift);
    float getHeightAt(float x, float z) const override;
};

#endif
