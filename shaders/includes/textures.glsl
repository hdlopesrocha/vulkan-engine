// Texture bindings for fragment shaders
// texture arrays: binding 1 = albedo array, 2 = normal array, 3 = height array, 4 = shadow map
layout(set = 1, binding = 1) uniform sampler2DArray albedoArray;
layout(set = 1, binding = 2) uniform sampler2DArray normalArray;
layout(set = 1, binding = 3) uniform sampler2DArray heightArray;
layout(set = 1, binding = 4) uniform sampler2D shadowMap;