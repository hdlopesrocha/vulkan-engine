#include "SimpleBrush.hpp"

 
SimpleBrush::SimpleBrush(int brush_){
    this->brush = brush_;
}

int SimpleBrush::paint(const Vertex &vertex) const {
    return brush;
}
