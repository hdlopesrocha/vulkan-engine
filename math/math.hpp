#ifndef MATH_HPP
#define MATH_HPP

#include <stb/stb_perlin.h>
#include <bitset>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/integer.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/matrix.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <stack>
#include <map>
#include <tsl/robin_map.h>
#include <filesystem>
#include <algorithm>
#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>

#define INFO_TYPE_FILE 99
#define INFO_TYPE_REMOVE 0
#define DISCARD_BRUSH_INDEX -1

#include "Common.hpp"
#include "AbstractBoundingBox.hpp"
#include "BoundingCube.hpp"
#include "BoundingBox.hpp"
#include "BoundingSphere.hpp"
#include "Plane.hpp"
#include "HeightFunction.hpp"
#include "CachedHeightMapSurface.hpp"
#include "PerlinSurface.hpp"
#include "FractalPerlinSurface.hpp"
#include "GradientPerlinSurface.hpp"
#include "HeightMap.hpp"
#include "HeightMapTif.hpp"
#include "Geometry.hpp"
#include "SphereGeometry.hpp"
#include "BoxGeometry.hpp"
#include "BoxLineGeometry.hpp"
#include "Frustum.hpp"
#include "TexturePainter.hpp"
#include "Camera.hpp"
#include "Tile.hpp"
#include "TileDraw.hpp"
#include "Brush3d.hpp"
#include "Transformation.hpp"
#include "Math.hpp"

enum BrushMode { ADD, REMOVE, REPLACE, BrushMode_COUNT };
const char* toString(BrushMode v);

#endif // MATH_HPP
