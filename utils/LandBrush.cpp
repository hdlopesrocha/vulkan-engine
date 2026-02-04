#include "LandBrush.hpp"
#include "../math/BrushMode.hpp"

LandBrush::LandBrush() {
    this->underground = DISCARD_BRUSH_INDEX;
    this->grass = 3;      // textures/grass_color.jpg (index 3)
    this->sand = 8;       // textures/sand_color.jpg (index 8)
    this->softSand = 10;  // textures/soft_sand_color.jpg (index 10)
    this->rock = 7;       // textures/rock_color.jpg (index 7)
    this->snow = 9;       // textures/snow_color.jpg (index 9)

    this->grassMixSand = 11;
    this->grassMixSnow = 12;
    this->rockMixGrass = 13;
    this->rockMixSnow = 14;
    this->rockMixSand = 15;
}

int LandBrush::paint(const Vertex &vertex, glm::vec4 translate, glm::vec4 scale) const {
    glm::vec3 n = glm::normalize(vertex.normal);
    float steepness =glm::dot(glm::vec3(0.0f,1.0f,0.0f), n );
    int grassLevel = 256;
    int sandLevel = 16;
    int softSandLevel = 2;
    int brushIndex;
    if (glm::dot(glm::vec3(0.0f,1.0f,0.0f), n ) <=0 ){
        brushIndex = underground;
    } else if(steepness < 0.980 ){
        brushIndex = rock;
    } else if(steepness < 0.985 ){
        if(vertex.position.y < softSandLevel){
            brushIndex = rock;
        } else if(vertex.position.y < sandLevel){
            brushIndex = rockMixSand;
        } else if(vertex.position.y < grassLevel){
            brushIndex = rockMixGrass;
        } else {
            brushIndex = rockMixSnow;
        }
    } else if(vertex.position.y < softSandLevel){
        brushIndex = softSand;
    } else if(vertex.position.y < sandLevel){
        brushIndex = sand;
    } else if(vertex.position.y < sandLevel+4){
        brushIndex = grassMixSand;
    } else if(vertex.position.y < grassLevel){
        brushIndex = grass;
    } else if(vertex.position.y < grassLevel+32){
        brushIndex = grassMixSnow;
    } else {
        brushIndex = snow;
    }

    return brushIndex;
}
