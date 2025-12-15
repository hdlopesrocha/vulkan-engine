#ifndef MATH_BRUSHMODE_HPP
#define MATH_BRUSHMODE_HPP

#define INFO_TYPE_FILE 99
#define INFO_TYPE_REMOVE 0
#define DISCARD_BRUSH_INDEX -1

enum BrushMode { ADD, REMOVE, REPLACE, BrushMode_COUNT };
const char* toString(BrushMode v);

#endif
