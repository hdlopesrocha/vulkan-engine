#pragma once
#define DISCARD_BRUSH_INDEX -1
// Sentinel returned by LandBrush: the terrain material is classified on the GPU
// (OctreeTriangles.comp, shaders/includes/LandBrush.glsl) from the vertex's
// world position and normal, so the octree build never has to compute the
// surface position (SDF::getPosition).
#define LAND_BRUSH_INDEX 255
// Sentinel returned by NormalBrush. The GPU keeps only nodes whose solved
// normal faces the painter's normal (+Y) and resolves them to the brush value
// (the water material); every other node is DISCARD_BRUSH_INDEX. Layer
// independent: any dynamic layer using NormalBrush gets the same behavior.
#define NORMAL_BRUSH_INDEX 254

enum BrushMode { ADD, REMOVE, REPLACE, PAINT, BrushMode_COUNT };

 
