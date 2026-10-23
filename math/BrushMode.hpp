#pragma once
#define DISCARD_BRUSH_INDEX -1
// Sentinel returned by LandBrush: the terrain material is classified on the GPU
// (OctreeTriangles.comp, shaders/includes/LandBrush.glsl) from the vertex's
// world position and normal, so the octree build never has to compute the
// surface position (SDF::getPosition).
#define LAND_BRUSH_INDEX 255

enum BrushMode { ADD, REMOVE, REPLACE, PAINT, BrushMode_COUNT };

 
