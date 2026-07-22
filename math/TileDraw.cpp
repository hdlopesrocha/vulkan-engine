#include "TileDraw.hpp"


TileDraw::TileDraw(uint index_, glm::vec2 size_, glm::vec2 offset_, glm::vec2 pivot_, float angle_){
    this->index = index_;
    this->size = size_;
    this->offset = offset_;
    this->pivot = pivot_;
    this->angle = angle_;
}
