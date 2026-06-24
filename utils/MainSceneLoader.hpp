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

// SDF primitives & wrappers
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

#include "../sdf/WrappedHeightMap.hpp"
#include "../sdf/WrappedBox.hpp"
#include "../sdf/WrappedSphere.hpp"
#include "../sdf/WrappedCapsule.hpp"
#include "../sdf/WrappedOctahedron.hpp"
#include "../sdf/WrappedPyramid.hpp"
#include "../sdf/WrappedTorus.hpp"
#include "../sdf/WrappedCone.hpp"
#include "../sdf/WrappedCylinder.hpp"
#include "../sdf/WrappedTaperedCylinder.hpp"
#include "../sdf/WrappedTaperedCapsule.hpp"
#include "../sdf/WrappedPerlinDistortDistanceEffect.hpp"
#include "../sdf/WrappedPerlinCarveDistanceEffect.hpp"
#include "../sdf/WrappedSineDistortDistanceEffect.hpp"
#include "../sdf/WrappedVoronoiCarveDistanceEffect.hpp"
#include "../sdf/WrappedOctreeDifference.hpp"
#include "../sdf/RoadSpline.hpp"
#include "../sdf/RoadDistanceFunction.hpp"
#include "../sdf/WrappedRoad.hpp"
#include "../sdf/TriangleStripDistanceFunction.hpp"
#include "../sdf/WrappedTriangleStrip.hpp"

// tree generation
#include "../tree/TreeHandler.hpp"

// change handlers & brushes
#include "LiquidSpaceChangeHandler.hpp"
#include "SolidSpaceChangeHandler.hpp"
#include "LandBrush.hpp"
#include "SimpleBrush.hpp"
#include "WaterBrush.hpp"


class MainSceneLoader : public SceneLoaderCallback {
public:


    Simplifier simplifier = Simplifier(0.99f, 0.1f, true);
    MainSceneLoader() {};
    ~MainSceneLoader() = default;
    void action(Octree &opaqueLayer, const OctreeChangeHandler& opaqueHandler,Octree &transparentLayer,const OctreeChangeHandler& transparentHandler) {

    }

    void loadScene(Octree &opaqueLayer, const OctreeChangeHandler& opaqueHandler,Octree &transparentLayer,const OctreeChangeHandler& transparentHandler) {

        //WrappedSignedDistanceFunction::resetCalls();
        int sizePerTile = 30;
        int tiles= 256;
        int height = 2048;
        float minSize = 30;
        glm::vec4 translate(0.0f);
        glm::vec4 scale(1.0f);

        BoundingBox mapBox = BoundingBox(glm::vec3(-sizePerTile*tiles*0.5,-height*0.5,-sizePerTile*tiles*0.5), glm::vec3(sizePerTile*tiles*0.5,height*0.5,sizePerTile*tiles*0.5));
        //camera.position.x = mapBox.getCenter().x;
        //camera.position.y = mapBox.getMaxY();
        //camera.position.z = mapBox.getCenter().z;

        {
            Transformation model = Transformation();
            std::cout << "\tGradientPerlinSurface"<< std::endl;
            GradientPerlinSurface heightFunction = GradientPerlinSurface(height, 1.0/(256.0f*sizePerTile), -64);
            std::cout << "\tCachedHeightMapSurface"<< std::endl;
            CachedHeightMapSurface cache = CachedHeightMapSurface(heightFunction, mapBox, sizePerTile);
            std::cout << "\tHeightMap"<< std::endl;
            HeightMap heightMap = HeightMap(cache, mapBox, sizePerTile);
            std::cout << "\tHeightMapDistanceFunction"<< std::endl;
            HeightMapDistanceFunction function = HeightMapDistanceFunction(&heightMap);
            std::cout << "\tWrappedHeightMap"<< std::endl;
            WrappedHeightMap wrappedFunction = WrappedHeightMap(&function);
            //wrappedFunction.cacheEnabled = true;
            
            std::cout << "\topaqueLayer.add(heightmap)"<< std::endl;
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, LandBrush(), minSize, simplifier, opaqueHandler);
        }
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(0), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 256);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(5), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 128);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            opaqueLayer.apply(SDF::opSubtraction, &wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 128);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            opaqueLayer.apply(SDF::opSubtraction, &wrappedFunction, model, translate, scale, SimpleBrush(4), minSize, simplifier, opaqueHandler);
        }
        
        {
            Transformation model = Transformation();
            std::cout << "\topaqueLayer.del(capsule)"<< std::endl;
            glm::vec3 a = glm::vec3(0,0, -3000);
            glm::vec3 b = glm::vec3(0,500,0);
            float r = 256.0f;
            CapsuleDistanceFunction function(a, b, r);
            WrappedCapsule wrappedFunction = WrappedCapsule(&function);
            WrappedPerlinDistortDistanceEffect distortedFunction = WrappedPerlinDistortDistanceEffect(&wrappedFunction, 64.0f, 0.1f/32.0f, glm::vec3(0), 0.0f, 1.0f);
            opaqueLayer.apply(SDF::opSubtraction, &distortedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 64);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            transparentLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(1), minSize, simplifier, transparentHandler);
        }

        {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,1500);
            glm::vec3 len = glm::vec3(1024.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 256);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            transparentLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(2), minSize, simplifier, transparentHandler);
        }

        {
            std::cout << "\topaqueLayer.add(octahedron)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*0);
            float radius = 256.0f;
            OctahedronDistanceFunction function = OctahedronDistanceFunction();
            Transformation model = Transformation(glm::vec3(radius), center, 0, 0, 0);
            WrappedOctahedron wrappedFunction = WrappedOctahedron(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(pyramid)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*1);
            float radius = 256.0f;
            PyramidDistanceFunction function = PyramidDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedPyramid wrappedFunction = WrappedPyramid(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(torus)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*2);
            float radius = 256.0f;
            TorusDistanceFunction function = TorusDistanceFunction(glm::vec2(0.5, 0.25));
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedTorus wrappedFunction = WrappedTorus(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(cone)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*3);
            float radius = 256.0f;
            ConeDistanceFunction function = ConeDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedCone wrappedFunction = WrappedCone(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(cylinder)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*4);
            float radius = 256.0f;
            CylinderDistanceFunction function = CylinderDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedCylinder wrappedFunction = WrappedCylinder(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(taperedCylinder)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*5);
            float radius = 256.0f;
            // Bottom radius 0.25, top radius 0.5 in local space → wider at top
            TaperedCylinderDistanceFunction function(0.25f, 0.5f);
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedTaperedCylinder wrappedFunction = WrappedTaperedCylinder(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(11), minSize, simplifier, opaqueHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(taperedCapsule)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*6);
            float radius = 256.0f;
            // Wider at bottom (r1=0.5), narrower at top (r2=0.25)
            TaperedCapsuleDistanceFunction function(
                glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 0.5f, 0.25f);
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedTaperedCapsule wrappedFunction = WrappedTaperedCapsule(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(11), minSize, simplifier, opaqueHandler);
        }
    
        if(false){
            std::cout << "\topaqueLayer.add(proceduralTree)"<< std::endl;
            tree::TreeHandler treeHandler;
            // Long thin branches: thin root, long segments, wide sparse crown
            treeHandler.setRoot(glm::vec3(0.0f, -1.0f, 0.0f), 0.06f);
            treeHandler.setParams(tree::TreeParams{1.5f, 0.25f, 0.35f, 50});
            treeHandler.populateEllipsoid(glm::vec3(0.0f, 0.5f, 0.0f), 0.6f, 0.15f, 250, 42u);
            treeHandler.generate();

            glm::vec3 treeCenter = glm::vec3(512, 512, 512*6);
            float treeScale = 280.0f;
            Transformation treeModel(glm::vec3(treeScale), treeCenter, 0, 0, 0);
            treeHandler.applyToOctree(opaqueLayer, opaqueHandler, treeModel, translate, scale,
                                      11, minSize*0.25f, simplifier);
        
       }

        {
            std::cout << "\topaqueLayer.add(perlinDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*0);
            float radius = 200.0f;
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            WrappedPerlinDistortDistanceEffect distortedFunction = WrappedPerlinDistortDistanceEffect(&wrappedFunction, 48.0f, 0.1f/32.0f, glm::vec3(0), 0.0f, 1.0f);
            //distortedFunction.cacheEnabled = true;
            opaqueLayer.apply(SDF::opUnion, &distortedFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(perlinCarve)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*1);
            float radius = 200.0f;
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            WrappedPerlinCarveDistanceEffect carvedFunction = WrappedPerlinCarveDistanceEffect(&wrappedFunction, 64.0f, 0.1f/32.0f, 0.1f, glm::vec3(0), 0.0f, 1.0f);
            //carvedFunction.cacheEnabled = true;
            opaqueLayer.apply(SDF::opUnion, &carvedFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(sineDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*2);
            float radius = 200.0f;
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            WrappedSineDistortDistanceEffect carvedFunction = WrappedSineDistortDistanceEffect(&wrappedFunction, 32.0f, 0.1f/2.0f, glm::vec3(0));
            //carvedFunction.cacheEnabled = true;
            opaqueLayer.apply(SDF::opUnion, &carvedFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*3);
            float radius = 200.0f;
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            WrappedVoronoiCarveDistanceEffect distortFunction = WrappedVoronoiCarveDistanceEffect(&wrappedFunction, 64.0f, 64.0f, glm::vec3(0), 0.0f, 1.0f);
            opaqueLayer.apply(SDF::opUnion, &distortFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*4);
            float radius = 200.0f;
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            WrappedVoronoiCarveDistanceEffect distortFunction = WrappedVoronoiCarveDistanceEffect(&wrappedFunction, 64.0f, 64.0f, glm::vec3(0), 0.0f, -1.0f);
            opaqueLayer.apply(SDF::opUnion, &distortFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }
    
        {
            Transformation model = Transformation();
            std::cout << "\ttransparentLayer.add(water)"<< std::endl;
            BoundingBox waterBox = mapBox;
            waterBox.setMax(mapBox.getMax() - glm::vec3(minSize*2.0f));
            waterBox.setMin(mapBox.getMin() + glm::vec3(minSize*2.0f));
            waterBox.setMaxY(0);
            waterBox.setMinY(mapBox.getMinY()*0.5f);
            OctreeDifferenceFunction function(&opaqueLayer, waterBox, minSize*2.0f);
            WrappedOctreeDifference wrappedFunction = WrappedOctreeDifference(&function);
            //wrappedFunction.cacheEnabled = true;
            transparentLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, WaterBrush(0), minSize, simplifier, transparentHandler);
        }
    
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(0), minSize*4, simplifier, opaqueHandler);
        }
        
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(2500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.apply(SDF::opUnion, &wrappedFunction, model, translate, scale, SimpleBrush(0), minSize*0.25, simplifier, opaqueHandler);
        }
        
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(2500+128,256+128,-1000+128);
            glm::vec3 len = glm::vec3(256.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.apply(SDF::opPaint, &wrappedFunction, model, translate, scale, SimpleBrush(1), minSize*4.0, simplifier, opaqueHandler);
        }
        
        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500+128,256+128,-1000+128);
            glm::vec3 len = glm::vec3(256.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.apply(SDF::opPaint, &wrappedFunction, model, translate, scale, SimpleBrush(1), minSize*0.25, simplifier, opaqueHandler);
        }
        //brushContext->model.scale = glm::vec3(256.0f);

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
                RoadDistanceFunction roadFunc(&roadSpline, 512.0f, 64.0f,
                                              tMin, tMax, startCap, endCap);
                BoundingSphere segSphere = roadSpline.boundingSphereInRange(tMin, tMax, halfDiag);
                WrappedRoad wrappedRoad(&roadFunc, segSphere.center, segSphere.radius);
                opaqueLayer.apply(SDF::opUnion, &wrappedRoad, roadModel,
                                  translate, scale, SimpleBrush(13),
                                  minSize, simplifier, opaqueHandler);
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

                    // Triangle strip section: v0,v1,v2 = 1st tri, v1,v2,v3 = 2nd tri
                    // Vertices in local unit space (Y=0 = strip mid-surface)
                    glm::vec3 v0(unitInner * glm::cos(a0), 0.0f, unitInner * glm::sin(a0));
                    glm::vec3 v1(unitOuter * glm::cos(a0), 0.0f, unitOuter * glm::sin(a0));
                    glm::vec3 v2(unitInner * glm::cos(a1), 0.0f, unitInner * glm::sin(a1));
                    glm::vec3 v3(unitOuter * glm::cos(a1), 0.0f, unitOuter * glm::sin(a1));

                    TriangleStripDistanceFunction tsFunc(v0, v1, v2, v3, unitHalfThick);

                    // Bounding sphere in world space
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
                    WrappedTriangleStrip wrappedTs(&tsFunc, segCenter, segRadius);
                    opaqueLayer.apply(SDF::opUnion, &wrappedTs, tsModel,
                                      translate, scale, SimpleBrush(14),
                                      minSize, simplifier, opaqueHandler);
                }
            }
        }

    }


};