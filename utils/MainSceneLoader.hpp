#pragma once

#include <iostream>
#include <chrono>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "Scene.hpp"
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


#include "LandBrush.hpp"
#include "SimpleBrush.hpp"
#include "WaterBrush.hpp"


class MainSceneLoader : public SceneLoaderCallback {
public:


    // angle=0.95 (cos≈18°): normals within 18° → flat surface → full distance tolerance.
    // distance=0.2: flat patches may have up to 20% cube-size SDF error (curved gets 10%).
    Simplifier simplifier = Simplifier(0.95f, 0.2f, true);
    MainSceneLoader() {};
    ~MainSceneLoader() = default;
    void action(
        Octree &opaqueLayer, 
        const Octree::OctreeNodeDataHandler& opaqueUpdateHandler, 
        const Octree::OctreeNodeDataHandler& opaqueDeleteHandler, 
        Octree &transparentLayer, 
        const Octree::OctreeNodeDataHandler& transparentUpdateHandler, 
        const Octree::OctreeNodeDataHandler& transparentDeleteHandler
    ) {

    }

    void loadScene(
        Octree &opaqueLayer, 
        Octree::OctreeNodeDataHandler &opaqueUpdateHandler,
        Octree::OctreeNodeDataHandler &opaqueDeleteHandler,
        Octree &transparentLayer,
        Octree::OctreeNodeDataHandler &transparentUpdateHandler,
        Octree::OctreeNodeDataHandler &transparentDeleteHandler
        ) {

        int sizePerTile = 30;
        int tiles= 256;
        int height = 2048;
        float minSize = 30;
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
            HeightMapDistanceFunction function = HeightMapDistanceFunction(&heightMap, minSize);
            
            std::cout << "\topaqueLayer.add(heightmap)"<< std::endl;
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, LandBrush(), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(0), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 256);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(5), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 128);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            opaqueLayer.apply(DeleteSignedDistanceOperation(), function, model, SimpleBrush(20), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 128);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            opaqueLayer.apply(DeleteSignedDistanceOperation(), function, model, SimpleBrush(4), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        
        {
            Transformation model = Transformation();
            std::cout << "\topaqueLayer.del(capsule)"<< std::endl;
            glm::vec3 a = glm::vec3(0,0, -3000);
            glm::vec3 b = glm::vec3(0,500,0);
            float r = 256.0f;
            CapsuleDistanceFunction function(a, b, r, model, minSize);
            PerlinDistortDistanceEffect distortedFunction = PerlinDistortDistanceEffect(function, 64.0f, 0.1f/32.0f, glm::vec3(0), 0.0f, 1.0f, model, minSize);
            opaqueLayer.apply(DeleteSignedDistanceOperation(), distortedFunction, model, SimpleBrush(7), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 64);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            transparentLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(1), minSize, simplifier, transparentUpdateHandler, transparentDeleteHandler);
        }

        {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,1500);
            glm::vec3 len = glm::vec3(1024.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 256);
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            transparentLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(2), minSize, simplifier, transparentUpdateHandler, transparentDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(octahedron)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*0);
            float radius = 256.0f;
            Transformation model = Transformation(glm::vec3(radius), center, 0, 0, 0);
            OctahedronDistanceFunction function = OctahedronDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(swept octahedron)"<< std::endl;
            float radius = 256.0f;
            glm::vec3 centerA = glm::vec3(512,512, 512*5);
            glm::vec3 centerB = glm::vec3(512,512*2, 512*5);
            Transformation modelA = Transformation(glm::vec3(radius), centerA, 0, 0, 0);
            Transformation modelB = Transformation(glm::vec3(radius*0.5f), centerB, 45, 0, 0);
            Transformation sweepModel = Transformation(glm::vec3(radius), (centerA + centerB) * 0.5f, 0, 0, 0);
            OctahedronDistanceFunction fn1(modelA, minSize);
            OctahedronDistanceFunction fn2(modelB, minSize);
            SweepSignedDistanceFunction<OctahedronDistanceFunction> sweepFn(fn1, fn2, sweepModel, minSize*0.25f);
            opaqueLayer.apply(AddSignedDistanceOperation(), sweepFn, sweepModel, SimpleBrush(16), minSize*0.25f, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(pyramid)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*1);
            float radius = 256.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            PyramidDistanceFunction function = PyramidDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(torus)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*2);
            float radius = 256.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            TorusDistanceFunction function = TorusDistanceFunction(glm::vec2(0.5, 0.25), model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(cone)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*3);
            float radius = 256.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            ConeDistanceFunction function = ConeDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(cylinder)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*4);
            float radius = 256.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            CylinderDistanceFunction function = CylinderDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(7), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(taperedCylinder)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*5);
            float radius = 256.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            TaperedCylinderDistanceFunction function(0.25f, 0.5f, model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(11), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(taperedCapsule)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*6);
            float radius = 256.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            TaperedCapsuleDistanceFunction function(
                glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 0.5f, 0.25f,
                model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(11), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

        {
            std::cout << "\topaqueLayer.add(mirror)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*7);
            glm::vec3 len = glm::vec3(32.0f, 256.0f, 256.0f);
            BoundingBox box = BoundingBox(center - len * 0.5f, center + len * 0.5f);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(6), minSize*0.5f, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }

    

        {
            std::cout << "\topaqueLayer.add(perlinDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*0);
            float radius = 200.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            PerlinDistortDistanceEffect distortedFunction = PerlinDistortDistanceEffect(function, 48.0f, 0.1f/32.0f, glm::vec3(0), 0.0f, 1.0f, model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), distortedFunction, model, SimpleBrush(15), minSize*0.25f, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(perlinCarve)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*1);
            float radius = 200.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            PerlinCarveDistanceEffect carvedFunction = PerlinCarveDistanceEffect(function, 64.0f, 0.1f/32.0f, 0.1f, glm::vec3(0), 0.0f, 1.0f, model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), carvedFunction, model, SimpleBrush(15), minSize*0.25f, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(sineDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*2);
            float radius = 200.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            SineDistortDistanceEffect carvedFunction = SineDistortDistanceEffect(function, 32.0f, 0.1f/2.0f, glm::vec3(0), model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), carvedFunction, model, SimpleBrush(15), minSize*0.25f, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*3);
            float radius = 200.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            VoronoiCarveDistanceEffect distortFunction = VoronoiCarveDistanceEffect(function, 64.0f, 64.0f, glm::vec3(0), 0.0f, 1.0f, model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), distortFunction, model, SimpleBrush(15), minSize*0.25f, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*4);
            float radius = 200.0f;
            Transformation model(glm::vec3(radius), center, 0,0,0);
            SphereDistanceFunction function = SphereDistanceFunction(model, minSize);
            VoronoiCarveDistanceEffect distortFunction = VoronoiCarveDistanceEffect(function, 64.0f, 64.0f, glm::vec3(0), 0.0f, -1.0f, model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), distortFunction, model, SimpleBrush(15), minSize*0.25f, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
    
        {
            Transformation model = Transformation();
            std::cout << "\ttransparentLayer.add(water)"<< std::endl;
            BoundingBox waterBox = mapBox;
            float bias = minSize*2.0;
            waterBox.setMax(mapBox.getMax() - glm::vec3(bias));
            waterBox.setMin(mapBox.getMin() + glm::vec3(bias));
            waterBox.setMaxY(0);
            waterBox.setMinY(mapBox.getMinY()*0.5f);
            OctreeDifferenceFunction function(&opaqueLayer, waterBox, bias);
            transparentLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(0), minSize, simplifier, transparentUpdateHandler, transparentDeleteHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(0), minSize*4, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(2500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, minSize);
            opaqueLayer.apply(AddSignedDistanceOperation(), function, model, SimpleBrush(0), minSize*0.25, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        
        {
            std::cout << "\topaqueLayer.add(road)"<< std::endl;
            std::vector<RoadSpline::ControlPoint> ctrlPts;
            glm::vec3 up(0.0f, 1.0f, 0.0f);

            int numPts = 32;
            float radius = 1500.0f;
            for (int i = 0; i <= numPts; ++i) {
                float t = (float)i / (float)numPts;
                float angle = t * 2.0f * glm::pi<float>();
                float x = radius * glm::cos(angle);
                float z = radius * glm::sin(angle);
                float y = 256.0f;
                ctrlPts.emplace_back(glm::vec3(x, y, z), up);
            }

            RoadSpline roadSpline(ctrlPts);
            Transformation roadModel = Transformation();
            int numSegs = 24;
            float overlap = 0.05f;
            float halfDiag = glm::length(glm::vec2(256.0f, 256.0f)) * 0.5f;
            for (int i = 0; i < numSegs; ++i) {
                float t0 = (float)i / (float)numSegs;
                float t1 = (float)(i + 1) / (float)numSegs;
                float tMin = std::max(0.0f, t0 - overlap / (float)numSegs);
                float tMax = std::min(1.0f, t1 + overlap / (float)numSegs);
                bool startCap = false;
                bool endCap   = false;
                BoundingSphere segSphere = roadSpline.boundingSphereInRange(tMin, tMax, halfDiag);
                RoadDistanceFunction roadFunc(&roadSpline, 512.0f, 64.0f,
                                              tMin, tMax, startCap, endCap,
                                              segSphere.center, segSphere.radius, Transformation(), minSize);
                opaqueLayer.apply(AddSignedDistanceOperation(), roadFunc, roadModel,
                                  SimpleBrush(12),
                                  minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
            }
        }

        {
            int numSegs = 96;
            float angleStep = 2.0f * glm::pi<float>() / numSegs;
            float overlap = 0.05f;
            float unitInner = 1.00f - 512.0f / 1500.0f;
            float unitOuter = 1.00f;
            float unitHalfThick = 32.0f / 1500.0f;
            float worldScale = 1500.0f;
            float heights[1] = { 800.0f };
            for (int ringIdx = 0; ringIdx < 1; ++ringIdx) {
                std::cout << "\topaqueLayer.add(triangleStrip " << ringIdx << ")" << std::endl;
                Transformation tsModel = Transformation(glm::vec3(worldScale),
                    glm::vec3(0.0f, heights[ringIdx], 0.0f), 0, 0, 0);
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
                    float y = heights[ringIdx];
                    glm::vec3 aabbMin(1e30f), aabbMax(-1e30f);
                    int samples = 4;
                    for (int j = 0; j <= samples; ++j) {
                        float a = aMin + (aMax - aMin) * ((float)j / (float)samples);
                        float rInner = unitInner * worldScale;
                        float rOuter = unitOuter * worldScale;
                        glm::vec3 inner(rInner * glm::cos(a), y - hh, rInner * glm::sin(a));
                        glm::vec3 outer(rOuter * glm::cos(a), y + hh, rOuter * glm::sin(a));
                        aabbMin = glm::min(aabbMin, inner);
                        aabbMin = glm::min(aabbMin, outer);
                        aabbMax = glm::max(aabbMax, inner);
                        aabbMax = glm::max(aabbMax, outer);
                    }
                    glm::vec3 segCenter = (aabbMin + aabbMax) * 0.5f;
                    float segRadius = glm::distance(segCenter, aabbMax);
                    TriangleStripDistanceFunction tsFunc(v0, v1, v2, v3, unitHalfThick,
                                                         segCenter, segRadius, tsModel, minSize);
                    opaqueLayer.apply(AddSignedDistanceOperation(), tsFunc, tsModel,
                                      SimpleBrush(14),
                                      minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
                }
            }
        }


        {
            std::cout << "\topaqueLayer.paint(box)"<< std::endl;
            glm::vec3 min = glm::vec3(2500+128,256-128,-1000+128);
            glm::vec3 len = glm::vec3(256.0f, 512.0f, 512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, minSize);
            opaqueLayer.apply(PaintSignedDistanceOperation(), function, model, SimpleBrush(1), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }
        
        {
            std::cout << "\topaqueLayer.paint(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500+128,256-128,-1000+128);
            glm::vec3 len = glm::vec3(256.0f, 512.0f, 512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            BoxDistanceFunction function = BoxDistanceFunction(model, minSize);
            opaqueLayer.apply(PaintSignedDistanceOperation(), function, model, SimpleBrush(1), minSize, simplifier, opaqueUpdateHandler, opaqueDeleteHandler);
        }


    }


};
