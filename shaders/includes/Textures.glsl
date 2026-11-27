// Texture bindings for fragment shaders
// texture arrays: binding 1 = albedo array, 2 = normal array, 3 = height array
// Bind into main descriptor set (set = 0)
// NOTE: binding 11 (legacy 360° environment cubemap) is REMOVED. Reflections
// use the procedural sky as the environment term. Bindings 4/8/9 (the retired
// CSM cascades) are intentionally absent: directional visibility is SDF
// raymarched, not sampled from a shadow map.
layout(set = 0, binding = 1) uniform sampler2DArray albedoArray;
layout(set = 0, binding = 2) uniform sampler2DArray normalArray;
layout(set = 0, binding = 3) uniform sampler2DArray heightArray;
layout(set = 0, binding = 12) uniform sampler2DArray roughnessArray;
layout(set = 0, binding = 13) uniform sampler2DArray aoArray;