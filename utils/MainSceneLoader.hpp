#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "Scene.hpp"
#include "TestShape.hpp"
#include "../space/Octree.hpp"

// math types
#include "../math/BoundingBox.hpp"
#include "../math/BoundingSphere.hpp"
#include "../math/Transformation.hpp"
#include "../math/CachedHeightMapSurface.hpp"
#include "../math/HeightMap.hpp"
#include "../math/GradientPerlinSurface.hpp"

// SDF primitives
#include "../sdf/HeightMapDistanceFunction.hpp"
#include "../sdf/BoxDistanceFunction.hpp"
#include "../sdf/SphereDistanceFunction.hpp"
#include "../sdf/CapsuleDistanceFunction.hpp"
#include "../sdf/OctahedronDistanceFunction.hpp"
#include "../sdf/PyramidDistanceFunction.hpp"
#include "../sdf/TorusDistanceFunction.hpp"
#include "../sdf/ConeDistanceFunction.hpp"
#include "../sdf/CylinderDistanceFunction.hpp"
#include "../sdf/TaperedCylinderDistanceFunction.hpp"
#include "../sdf/TaperedCapsuleDistanceFunction.hpp"
#include "../sdf/OctreeDifferenceFunction.hpp"

// SDF sweep
#include "../sdf/SweepSignedDistanceFunction.hpp"

// SDF operations
#include "../sdf/AddSignedDistanceOperation.hpp"
#include "../sdf/DeleteSignedDistanceOperation.hpp"
#include "../sdf/PaintSignedDistanceOperation.hpp"

// SDF effects
#include "../sdf/PerlinDistortDistanceEffect.hpp"
#include "../sdf/PerlinCarveDistanceEffect.hpp"
#include "../sdf/SineDistortDistanceEffect.hpp"
#include "../sdf/VoronoiCarveDistanceEffect.hpp"
#include "../sdf/RoadSpline.hpp"
#include "../sdf/RoadDistanceFunction.hpp"
#include "../sdf/TriangleStripDistanceFunction.hpp"
#include "../sdf/SvgShape.hpp"
#include "../sdf/SvgPolygonDistanceFunction.hpp"


#include "LandBrush.hpp"
#include "SimpleBrush.hpp"
#include "SvgBrush.hpp"
#include "WaterBrush.hpp"
#include "NormalBrush.hpp"


class MainSceneLoader : public SceneLoaderCallback {
public:


    // angle=0.95 (cos≈18°): normals within 18° → flat surface → full distance tolerance.
    // distance=0.2: flat patches may have up to 20% cube-size SDF error (curved gets 10%).
    Simplifier simplifier = Simplifier(0.95f, 0.2f, true);
    // Brush index of the first MaterialKind::Video material (set by MyApp after
    // the material palette is built). Negative disables the demo video cube.
    int videoBrushIndex = -1;
    void setVideoBrushIndex(int index) { videoBrushIndex = index; }
    // Brush index of the solid-layer cube demonstrating SdfGiven coordinates.
    int sdfGivenBrushIndex = -1;
    void setSdfGivenBrushIndex(int index) { sdfGivenBrushIndex = index; }
    MainSceneLoader() {};
    ~MainSceneLoader() = default;


    // Grid placement of the catalogue entries: 4 per row along X, rows along
    // Z, and a new Y layer every 16 entries. `size` is the full object extent;
    // `margin` is the gap between grid cells.
    float shapeIndex(TestShape shape) {
        return static_cast<float>(static_cast<int>(shape));
    }

    glm::vec3 getCenter(TestShape shape, const glm::vec3& testObjectOrigin, const glm::vec3& size, const glm::vec3& margin) {
        int elementsPerRow = 4;
        int shIndex = shapeIndex(shape);
        
        int row = std::floor((float)shIndex / (float)elementsPerRow);
        int col = std::fmod((float)shIndex, (float)elementsPerRow);
        int layer = std::floor((float)shIndex / (float)(elementsPerRow * elementsPerRow));
        glm::vec3 indices = glm::vec3(col, layer, row);
        glm::vec3 shift = (margin + size) * indices;


        return testObjectOrigin + shift;
    }


    // Apply one catalogue entry. All test objects share the same grid layout
    // (getCenter()); the material and SDF primitive differ per shape.
    void applyTestObject(
        TestShape shape,
        Octree& opaqueLayer,
        Octree::OctreeNodeDataHandler& opaqueUpdateHandler,
        Octree::OctreeNodeDataHandler& opaqueDeleteHandler,
        Octree::OctreeNodeDataHandler& transparentUpdateHandler,
        Octree::OctreeNodeDataHandler& transparentDeleteHandler,
        Octree& videoLayer,
        Octree::OctreeNodeDataHandler& videoUpdateHandler,
        Octree::OctreeNodeDataHandler& videoDeleteHandler
    ) {
        (void)transparentUpdateHandler;
        (void)transparentDeleteHandler;

        const int cellSize = 8;
        const glm::vec3 testObjectSize(128.0f, 128.0f, 128.0f);
        const glm::vec3 testObjectMargin(64.0f, 64.0f, 64.0f);
        const glm::vec3 testObjectOrigin(0.0f, 512.0f, 0.0f);


        switch (shape) {
            case TestShape::Octrahedron: {
                std::cout << "\topaqueLayer.add(octahedron)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0, 0, 0);
                OctahedronDistanceFunction function = OctahedronDistanceFunction(model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::Pyramid: {
                std::cout << "\topaqueLayer.add(pyramid)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                PyramidDistanceFunction function = PyramidDistanceFunction(model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::Torus: {
                std::cout << "\topaqueLayer.add(torus)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                TorusDistanceFunction function = TorusDistanceFunction(glm::vec2(0.5, 0.25), model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(3), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::Cone: {
                std::cout << "\topaqueLayer.add(cone)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                ConeDistanceFunction function = ConeDistanceFunction(model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::Cylinder: {
                std::cout << "\topaqueLayer.add(cylinder)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                CylinderDistanceFunction function = CylinderDistanceFunction(model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(4), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::TaperedCylinder: {
                std::cout << "\topaqueLayer.add(taperedCylinder)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                TaperedCylinderDistanceFunction function(0.25f, 0.5f, model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(11), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::TaperedCapsule: {
                std::cout << "\topaqueLayer.add(taperedCapsule)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                TaperedCapsuleDistanceFunction function(
                    glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 0.5f, 0.25f,
                    model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(11), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::Mirror: {
                std::cout << "\topaqueLayer.add(mirror)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                glm::vec3 len = glm::vec3(testObjectSize.x*0.125f, testObjectSize.y, testObjectSize.z);
                BoundingBox box = BoundingBox(center - len * 0.5f, center + len * 0.5f);
                Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
                BoxDistanceFunction function = BoxDistanceFunction(model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(6), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::PerlinDistort: {
                std::cout << "\topaqueLayer.add(perlinDistort)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                SphereDistanceFunction function = SphereDistanceFunction(model, cellSize);
                PerlinDistortDistanceEffect distortedFunction = PerlinDistortDistanceEffect(function, cellSize*2.0f, 0.1f/cellSize, glm::vec3(0), 0.0f, 1.0f, model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), distortedFunction, model, SimpleBrush(15), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::PerlinCarve: {
                std::cout << "\topaqueLayer.add(perlinCarve)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                SphereDistanceFunction function = SphereDistanceFunction(model, cellSize);
                PerlinCarveDistanceEffect carvedFunction = PerlinCarveDistanceEffect(function, cellSize*2.0f, 0.1f/cellSize, 0.1f, glm::vec3(0), 0.0f, 1.0f, model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), carvedFunction, model, SimpleBrush(15), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::SineDistort: {
                std::cout << "\topaqueLayer.add(sineDistort)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                SphereDistanceFunction function = SphereDistanceFunction(model, cellSize);
                SineDistortDistanceEffect carvedFunction = SineDistortDistanceEffect(function, cellSize*2.0f, 0.1f/cellSize, glm::vec3(0), model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), carvedFunction, model, SimpleBrush(15), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::VoronoiDistort: {
                std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                SphereDistanceFunction function = SphereDistanceFunction(model, cellSize);
                VoronoiCarveDistanceEffect distortFunction = VoronoiCarveDistanceEffect(function, cellSize*2.0f, 0.1f/cellSize, glm::vec3(0), 0.0f, 1.0f, model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), distortFunction, model, SimpleBrush(15), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::VoronoiDistortInverse: {
                std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                Transformation model = Transformation(testObjectSize*0.5f, center, 0,0,0);
                SphereDistanceFunction function = SphereDistanceFunction(model, cellSize);
                VoronoiCarveDistanceEffect distortFunction = VoronoiCarveDistanceEffect(function, cellSize*2.0f, 0.1f/cellSize, glm::vec3(0), 0.0f, -1.0f, model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), distortFunction, model, SimpleBrush(15), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            case TestShape::SvgExtrusion: {
                std::cout << "\topaqueLayer.add(svg extrusion \u2229 box)" << std::endl;
                SvgShape svgShape;
                std::string svgError;
                std::string svgResolved;
                if (SvgShape::loadFromFileResolved("svg/test_logo.svg", svgShape, svgError, &svgResolved)) {
                    auto svgShapePtr = std::make_shared<const SvgShape>(std::move(svgShape));
                    std::cout << "\t\tsvg '" << svgResolved << "': " << svgShapePtr->contours.size()
                              << " contours" << std::endl;
                    glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                    Transformation model(glm::vec3(testObjectSize*0.5f), center, 0.0f, 0.0f, 0.0f);
                    SvgPolygonDistanceFunction function(svgShapePtr, model, cellSize);
                    SvgBrush brush(svgShapePtr, model, /*materialIndex=*/15, glm::vec3(0.0f, 0.5f, 0.5f),
                                   /*tintFromFill=*/true);
                    opaqueLayer.apply(AddSignedDistanceOperation(), function, model, brush, TriplanarMappingPainter(),
                                      cellSize * 0.5f, simplifier, opaqueUpdateHandler,
                                      opaqueDeleteHandler);
                } else {
                    std::cerr << "\tsvg brush skipped: " << svgError << std::endl;
                }
                break;
            }
            case TestShape::SdfGivenCube: {
            if (sdfGivenBrushIndex >= 0) {
                std::cout << "\topaqueLayer.add(sdfGiven cube)" << std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                glm::vec3 len = glm::vec3(testObjectSize);
                BoundingBox box = BoundingBox(center - len * 0.5f, center + len * 0.5f);
                Transformation model = Transformation(box.getLength() * 0.5f, box.getCenter(), 0, 0, 0);
                BoxDistanceFunction function = BoxDistanceFunction(model, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(sdfGivenBrushIndex), TriplanarMappingPainter(),
                                  cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
            }
                break;
            }

            case TestShape::VideoCube: {
                std::cout << "\tvideoLayer.add(cube)" << std::endl;
                glm::vec3 center = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                glm::vec3 len = glm::vec3(testObjectSize);
                BoundingBox box = BoundingBox(center - len * 0.5f, center + len * 0.5f);
                Transformation model = Transformation(box.getLength() * 0.5f, box.getCenter(), 0, 0, 0);
                BoxDistanceFunction function = BoxDistanceFunction(model, cellSize);
                videoLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(videoBrushIndex), TriplanarMappingPainter(),
                                 cellSize, simplifier, videoUpdateHandler, videoDeleteHandler);
            
                break;
            }
            case TestShape::SweptOctrahedron: {
                std::cout << "\topaqueLayer.add(swept octahedron)"<< std::endl;
                glm::vec3 centerA = getCenter(shape, testObjectOrigin, testObjectSize, testObjectMargin);
                glm::vec3 centerB = centerA + glm::vec3(0.0f, testObjectSize.y, 0.0f);
                Transformation modelA = Transformation(testObjectSize* 0.5f, centerA, 0, 0, 0);
                Transformation modelB = Transformation(testObjectSize * 0.125f, centerB, 45, 0, 0);
                Transformation sweepModel = Transformation(testObjectSize*0.5f, (centerA + centerB) * 0.5f, 0, 0, 0);
                OctahedronDistanceFunction fn1(modelA, cellSize);
                OctahedronDistanceFunction fn2(modelB, cellSize);
                SweepSignedDistanceFunction<OctahedronDistanceFunction> sweepFn(fn1, fn2, sweepModel, cellSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), sweepFn, sweepModel, SimpleBrush(16), TriplanarMappingPainter(), cellSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                break;
            }
            default:
                std::cerr << "\tunknown test shape " << static_cast<int>(shape) << std::endl;
                break;
        }
    }

    void applyTestObjects(
        Octree& opaqueLayer,
        Octree::OctreeNodeDataHandler& opaqueUpdateHandler,
        Octree::OctreeNodeDataHandler& opaqueDeleteHandler,
        Octree::OctreeNodeDataHandler& transparentUpdateHandler,
        Octree::OctreeNodeDataHandler& transparentDeleteHandler,
        Octree& videoLayer,
        Octree::OctreeNodeDataHandler& videoUpdateHandler,
        Octree::OctreeNodeDataHandler& videoDeleteHandler
    ) {
        // Call every catalogue entry; the switch in applyTestObject owns the
        // per-shape work (including the conditional SdfGiven/video demos).
        for (int i = 0; i < static_cast<int>(TestShape::Count); ++i) {
            applyTestObject(static_cast<TestShape>(i), opaqueLayer, opaqueUpdateHandler,
                            opaqueDeleteHandler, transparentUpdateHandler, transparentDeleteHandler,
                            videoLayer, videoUpdateHandler, videoDeleteHandler);
        }
    }

    void loadScene(
        std::vector<Octree*>& octrees,
        std::vector<Octree::OctreeNodeDataHandler>& updateHandlers,
        std::vector<Octree::OctreeNodeDataHandler>& deleteHandlers
        ) {
        // Dynamic layers: layer 0 = opaque terrain, layer 1 = transparent
        // water when present. Extra layers start empty (brushes populate them).
        if (octrees.empty() || !octrees[0]) return;
        Octree& opaqueLayer = *octrees[0];
        Octree::OctreeNodeDataHandler opaqueUpdateHandler;
        Octree::OctreeNodeDataHandler opaqueDeleteHandler;
        if (!updateHandlers.empty()) opaqueUpdateHandler = updateHandlers[0];
        if (!deleteHandlers.empty()) opaqueDeleteHandler = deleteHandlers[0];
        Octree* transparentPtr = octrees.size() > 1 ? octrees[1] : nullptr;
        
        Octree::OctreeNodeDataHandler transparentUpdateHandler;
        Octree::OctreeNodeDataHandler transparentDeleteHandler;
        if (updateHandlers.size() > 1) transparentUpdateHandler = updateHandlers[1];
        if (deleteHandlers.size() > 1) transparentDeleteHandler = deleteHandlers[1];
        const bool hasTransparent = (transparentPtr != nullptr);
        Octree& transparentLayer = hasTransparent ? *transparentPtr : opaqueLayer;

          Octree& videoLayer = *octrees[2];
            Octree::OctreeNodeDataHandler videoUpdateHandler;
            Octree::OctreeNodeDataHandler videoDeleteHandler;
            if (updateHandlers.size() > 2) videoUpdateHandler = updateHandlers[2];
            if (deleteHandlers.size() > 2) videoDeleteHandler = deleteHandlers[2];

        int sizePerTile = 32;
        int tiles= 256;
        int height = 2048;
        glm::vec4 translate(0.0f);
        glm::vec4 scale(1.0f);

        BoundingBox mapBox = BoundingBox(glm::vec3(-sizePerTile*tiles*0.5,-height*0.5,-sizePerTile*tiles*0.5), glm::vec3(sizePerTile*tiles*0.5,height*0.5,sizePerTile*tiles*0.5));

        {
            Transformation model = Transformation();
            std::cout << "\tGradientPerlinSurface"<< std::endl;
            GradientPerlinSurface heightFunction = GradientPerlinSurface(height, 1.0/(256.0f*sizePerTile), -64);
            std::cout << "\tCachedHeightMapSurface"<< std::endl;
            CachedHeightMapSurface cache = CachedHeightMapSurface(heightFunction, mapBox, sizePerTile);
            std::cout << "\tHeightMap"<< std::endl;
            HeightMap heightMap = HeightMap(cache, mapBox, sizePerTile);
            std::cout << "\tHeightMapDistanceFunction"<< std::endl;
            HeightMapDistanceFunction function = HeightMapDistanceFunction(&heightMap, sizePerTile);
            
            std::cout << "\topaqueLayer.add(heightmap)"<< std::endl;
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, LandBrush(), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(0), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 256);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(5), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 128);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(DeleteSignedDistanceOperation(), function, model, SimpleBrush(20), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 128);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(DeleteSignedDistanceOperation(), function, model, SimpleBrush(4), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        
        {
            Transformation model = Transformation();
            std::cout << "\topaqueLayer.del(capsule)"<< std::endl;
            glm::vec3 a = glm::vec3(0,0, -3000);
            glm::vec3 b = glm::vec3(0,500,0);
            float r = 256.0f;
            CapsuleDistanceFunction function(a, b, r, model, sizePerTile);
            PerlinDistortDistanceEffect distortedFunction = PerlinDistortDistanceEffect(function, 64.0f, 0.1f/32.0f, glm::vec3(0), 0.0f, 1.0f, model, sizePerTile);
            opaqueLayer.apply(DeleteSignedDistanceOperation(), distortedFunction, model, SimpleBrush(7), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        if (hasTransparent) {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            glm::vec3 center = glm::vec3(1500+512.0f,512.0f,500+512.0f);
            BoundingSphere sphere = BoundingSphere(center, 64);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, sizePerTile);
            transparentLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(1), TriplanarMappingPainter(), sizePerTile, simplifier, transparentUpdateHandler, transparentDeleteHandler);
        }

        if (hasTransparent) {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            glm::vec3 center = glm::vec3(1500+256.0f,256.0f,500+512.0f + 256.0f);
            BoundingSphere sphere = BoundingSphere(center, 128);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, sizePerTile);
            transparentLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(2), TriplanarMappingPainter(), sizePerTile, simplifier, transparentUpdateHandler, transparentDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(0), TriplanarMappingPainter(), sizePerTile*4, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(2500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(0), TriplanarMappingPainter(), sizePerTile*0.25, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        if (hasTransparent) {
            Transformation model = Transformation();
            std::cout << "\ttransparentLayer.add(water)"<< std::endl;
            BoundingBox waterBox = mapBox;
            float bias = sizePerTile*2.0;
            waterBox.setMax(mapBox.getMax() - glm::vec3(bias));
            waterBox.setMin(mapBox.getMin() + glm::vec3(bias));
            waterBox.setMaxY(0);
            waterBox.setMinY(mapBox.getMinY()*0.5f);
            NormalBrush waterBrush = NormalBrush(
                0, 
                glm::vec3(0.5f, 0.5f, 0.5f), 
                glm::vec3(0.0f, 1.0f, 0.0f)
            );
            OctreeDifferenceFunction function(&opaqueLayer, waterBox, bias);
            transparentLayer.apply(AddSignedDistanceOperation(), function, model, waterBrush, TriplanarMappingPainter(), sizePerTile, simplifier, transparentUpdateHandler, transparentDeleteHandler);
        }
    


        glm::vec3 fireSpotPosition(0.0f, 1200.0f, 0.0f);
        float fireSpotHeight = 64.0f;
        float fireSpotRadius = 128.0f;

        {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            BoundingSphere sphere = BoundingSphere(fireSpotPosition, fireSpotRadius);
            Transformation model = Transformation(glm::vec3(sphere.radius,fireSpotHeight,sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, sizePerTile);
            transparentLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(4), TriplanarMappingPainter(), sizePerTile, simplifier, transparentUpdateHandler, transparentDeleteHandler);
        }


        {
            std::cout << "\topaqueLayer.add(road)"<< std::endl;
            std::vector<RoadSpline::ControlPoint> ctrlPts;
            glm::vec3 up(0.0f, 1.0f, 0.0f);
            int numPts = 32;
            for (int i = 0; i <= numPts; ++i) {
                float t = (float)i / (float)numPts;
                float angle = t * 2.0f * glm::pi<float>();
                float x = fireSpotRadius * glm::cos(angle);
                float y = fireSpotHeight;
                float z = fireSpotRadius * glm::sin(angle);
                ctrlPts.emplace_back(fireSpotPosition + glm::vec3(x, y, z), up);
            }

            RoadSpline roadSpline(ctrlPts);
            Transformation roadModel = Transformation();
            int numSegs = 24;
            float overlap = 0.05f;
            float halfDiag = glm::length(glm::vec2(fireSpotRadius, fireSpotRadius)) * 0.5f;
            for (int i = 0; i < numSegs; ++i) {
                float t0 = (float)i / (float)numSegs;
                float t1 = (float)(i + 1) / (float)numSegs;
                float tMin = std::max(0.0f, t0 - overlap / (float)numSegs);
                float tMax = std::min(1.0f, t1 + overlap / (float)numSegs);
                bool startCap = false;
                bool endCap   = false;
                BoundingSphere segSphere = roadSpline.boundingSphereInRange(tMin, tMax, halfDiag);
                RoadDistanceFunction roadFunc(&roadSpline, 32.0f, 3*32.0f,
                                              tMin, tMax, startCap, endCap,
                                              segSphere.center, segSphere.radius, Transformation(), sizePerTile);
                opaqueLayer.apply(AddSignedDistanceOperation(), roadFunc, roadModel,
                                  SimpleBrush(14), TriplanarMappingPainter(),
                                  sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
            }
        }
        glm::vec3 ringPosition(0.0f, 800.0f, 0.0f);
        glm::vec3 ringSize(512.0f, 32.0f, 512.0f);
        if (hasTransparent) {
            int numSegs = 256;
            float angleStep = 2.0f * glm::pi<float>() / numSegs;
            float overlap = 0.05f;
            float unitInner = 1.00f - 512.0f / 1024.0f;
            float unitOuter = 1.00f;
            float unitHalfThick = 64.0f / 1024.0f;
            float worldScale = 1024.0f;
            for (int ringIdx = 0; ringIdx < 1; ++ringIdx) {
                std::cout << "\ttransparentLayer.add(triangleStrip " << ringIdx << ")" << std::endl;
                Transformation tsModel = Transformation(glm::vec3(worldScale),
                    ringPosition, 0, 0, 0);
                for (int i = 0; i < numSegs; ++i) {
                    float a0 = i * angleStep;
                    float a1 = (i + 1) * angleStep;

                    glm::vec3 v0(unitInner * glm::cos(a0), 0.0f, unitInner * glm::sin(a0));
                    glm::vec3 v1(unitOuter * glm::cos(a0), 0.0f, unitOuter * glm::sin(a0));
                    glm::vec3 v2(unitInner * glm::cos(a1), 0.0f, unitInner * glm::sin(a1));
                    glm::vec3 v3(unitOuter * glm::cos(a1), 0.0f, unitOuter * glm::sin(a1));

                    float aMin = a0 - overlap * angleStep;
                    float aMax = a1 + overlap * angleStep;
                    float hh = unitHalfThick * worldScale;
                    float y = ringPosition.y;
                    BoundingBox aabb;
                    aabb.setMin(glm::vec3(std::numeric_limits<float>::max()));
                    aabb.setMax(glm::vec3(std::numeric_limits<float>::lowest()));
                    int samples = 4;
                    for (int j = 0; j <= samples; ++j) {
                        float a = aMin + (aMax - aMin) * ((float)j / (float)samples);
                        float rInner = unitInner * worldScale;
                        float rOuter = unitOuter * worldScale;
                        glm::vec3 inner(rInner * glm::cos(a), y - hh, rInner * glm::sin(a));
                        glm::vec3 outer(rOuter * glm::cos(a), y + hh, rOuter * glm::sin(a));
                        aabb.setMin(glm::min(aabb.getMin(), inner));
                        aabb.setMin(glm::min(aabb.getMin(), outer));
                        aabb.setMax(glm::max(aabb.getMax(), inner));
                        aabb.setMax(glm::max(aabb.getMax(), outer));
                    }
                    glm::vec3 segCenter = aabb.getCenter();
                    float segRadius = glm::length(aabb.getLength()) * 0.5f;

                    TriangleStripDistanceFunction tsFunc(v0, v1, v2, v3, unitHalfThick,
                                                         segCenter, segRadius, tsModel, sizePerTile);
                    transparentLayer.apply(AddSignedDistanceOperation(), tsFunc, tsModel,
                                      SimpleBrush(0), TriplanarMappingPainter(),
                                      sizePerTile*0.25f, simplifier, transparentUpdateHandler, transparentDeleteHandler);
                }
            }
        }


        {
            std::cout << "\topaqueLayer.paint(box)"<< std::endl;
            glm::vec3 min = glm::vec3(2500+128,256-128,-1000+128);
            glm::vec3 len = glm::vec3(256.0f, 512.0f, 512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(PaintSignedDistanceOperation(), function, model, SimpleBrush(1), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        
        {
            std::cout << "\topaqueLayer.paint(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500+128,256-128,-1000+128);
            glm::vec3 len = glm::vec3(256.0f, 512.0f, 512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, sizePerTile);
            opaqueLayer.apply(PaintSignedDistanceOperation(), function, model, SimpleBrush(1), TriplanarMappingPainter(), sizePerTile, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }


        applyTestObjects(opaqueLayer, opaqueUpdateHandler, opaqueDeleteHandler, 
                            transparentUpdateHandler, transparentDeleteHandler,
                         videoLayer, videoUpdateHandler, videoDeleteHandler);

    }


};
