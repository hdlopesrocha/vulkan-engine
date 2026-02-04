#pragma once

#include <iostream>
#include <chrono>
#include <glm/glm.hpp>

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
#include "../sdf/WrappedPerlinDistortDistanceEffect.hpp"
#include "../sdf/WrappedPerlinCarveDistanceEffect.hpp"
#include "../sdf/WrappedSineDistortDistanceEffect.hpp"
#include "../sdf/WrappedVoronoiCarveDistanceEffect.hpp"
#include "../sdf/WrappedOctreeDifference.hpp"

// change handlers & brushes
#include "LiquidSpaceChangeHandler.hpp"
#include "SolidSpaceChangeHandler.hpp"
#include "LandBrush.hpp"
#include "SimpleBrush.hpp"
#include "WaterBrush.hpp"


class MainSceneLoader : public SceneLoaderCallback {
public:


    Simplifier simplifier = Simplifier(0.99f, 0.1f, true);
    MainSceneLoader() {

    };
    ~MainSceneLoader() = default;

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
            opaqueLayer.add(&wrappedFunction, model, translate, scale, LandBrush(), minSize, simplifier, opaqueHandler);
        }


        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(0), minSize*2.0f, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 256);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(5), minSize*0.5f, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 128);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            opaqueLayer.del(&wrappedFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.del(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+3.0f*len/4.0f, 128);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            opaqueLayer.del(&wrappedFunction, model, translate, scale, SimpleBrush(4), minSize, simplifier, opaqueHandler);
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
            opaqueLayer.del(&distortedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\ttransparentLayer.add(sphere)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,500);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingSphere sphere = BoundingSphere(min+len, 64);
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model = Transformation(glm::vec3(sphere.radius), sphere.center, 0, 0, 0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            transparentLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(0), minSize*0.1f, simplifier, transparentHandler);
        }

        {
            std::cout << "\topaqueLayer.add(octahedron)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*0);
            float radius = 256.0f;
            OctahedronDistanceFunction function = OctahedronDistanceFunction();
            Transformation model = Transformation(glm::vec3(radius), center, 0, 0, 0);
            WrappedOctahedron wrappedFunction = WrappedOctahedron(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(pyramid)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*1);
            float radius = 256.0f;
            PyramidDistanceFunction function = PyramidDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedPyramid wrappedFunction = WrappedPyramid(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(torus)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*2);
            float radius = 256.0f;
            TorusDistanceFunction function = TorusDistanceFunction(glm::vec2(0.5, 0.25));
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedTorus wrappedFunction = WrappedTorus(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }


        {
            std::cout << "\topaqueLayer.add(cone)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*3);
            float radius = 256.0f;
            ConeDistanceFunction function = ConeDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedCone wrappedFunction = WrappedCone(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(cylinder)"<< std::endl;
            glm::vec3 center = glm::vec3(0,512, 512*4);
            float radius = 256.0f;
            CylinderDistanceFunction function = CylinderDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedCylinder wrappedFunction = WrappedCylinder(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(7), minSize, simplifier, opaqueHandler);
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
            opaqueLayer.add(&distortedFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
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
            opaqueLayer.add(&carvedFunction, model, translate, scale, SimpleBrush(7), minSize*0.2f, simplifier, opaqueHandler);
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
            opaqueLayer.add(&carvedFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }
        {
            std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*3);
            float radius = 200.0f;
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            WrappedVoronoiCarveDistanceEffect distortFunction = WrappedVoronoiCarveDistanceEffect(&wrappedFunction, 64.0f, 64.0f, glm::vec3(0), 0.0f, 1.0f);
            opaqueLayer.add(&distortFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
        }
        {
            std::cout << "\topaqueLayer.add(voronoiDistort)"<< std::endl;
            glm::vec3 center = glm::vec3(512,512, 512*4);
            float radius = 200.0f;
            SphereDistanceFunction function = SphereDistanceFunction();
            Transformation model(glm::vec3(radius), center, 0,0,0);
            WrappedSphere wrappedFunction = WrappedSphere(&function);
            WrappedVoronoiCarveDistanceEffect distortFunction = WrappedVoronoiCarveDistanceEffect(&wrappedFunction, 64.0f, 64.0f, glm::vec3(0), 0.0f, -1.0f);
            opaqueLayer.add(&distortFunction, model, translate, scale, SimpleBrush(7), minSize*0.25f, simplifier, opaqueHandler);
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
            transparentLayer.add(&wrappedFunction, model, translate, scale, WaterBrush(0), minSize, simplifier, transparentHandler);
        }

        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(1500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(0), minSize*4, simplifier, opaqueHandler);
        }

        {
            std::cout << "\topaqueLayer.add(box)"<< std::endl;
            glm::vec3 min = glm::vec3(2500,0,-1000);
            glm::vec3 len = glm::vec3(512.0f);
            BoundingBox box = BoundingBox(min,min+len);
            BoxDistanceFunction function = BoxDistanceFunction();
            Transformation model = Transformation(box.getLength()*0.5f, box.getCenter(), 0, 0, 0);
            WrappedBox wrappedFunction = WrappedBox(&function);
            opaqueLayer.add(&wrappedFunction, model, translate, scale, SimpleBrush(0), minSize*0.25, simplifier, opaqueHandler);
        }

   
        //brushContext->model.scale = glm::vec3(256.0f);




    }


};