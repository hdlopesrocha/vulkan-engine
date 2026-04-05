// Texture bindings for fragment shaders
// texture arrays: binding 1 = albedo array, 2 = normal array, 3 = height array, 4 = shadow map (cascade 0)
// Bind into main descriptor set (set = 0)
layout(set = 0, binding = 1) uniform sampler2DArray albedoArray;
layout(set = 0, binding = 2) uniform sampler2DArray normalArray;
layout(set = 0, binding = 3) uniform sampler2DArray heightArray;
layout(set = 0, binding = 4) uniform sampler2D shadowMap;
layout(set = 0, binding = 8) uniform sampler2D shadowMap1;  // cascade 1 (4x ortho0)
layout(set = 0, binding = 9) uniform sampler2D shadowMap2;  // cascade 2 (16x ortho0)