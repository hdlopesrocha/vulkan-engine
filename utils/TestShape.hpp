#pragma once

// Catalogue of procedural test objects applied by MainSceneLoader.
// applyTestObjects walks every entry so adding a new shape only needs a new
// enum value + switch case in MainSceneLoader::applyTestObject.
enum class TestShape {
    Octrahedron = 0,
    Pyramid,
    Torus,
    Cone,
    Cylinder,
    TaperedCylinder,
    TaperedCapsule,
    Mirror,
    PerlinDistort,
    PerlinCarve,
    SineDistort,
    VoronoiDistort,
    VoronoiDistortInverse,
    SvgExtrusion,
    SdfGivenCube,
    VideoCube,
    SweptOctrahedron,
    CubeProjectionX,
    CubeProjectionY,
    CubeProjectionZ,
    CubeAngular,
    Count
};
