// Hybrid RT shared declarations (raster ray queries + RT pipeline).
// The struct layout must match RayTracingParams (C++). Each shader declares
// its own uniform BLOCK with the set/binding of its pipeline; this file only
// defines the struct + helpers so raster (set 0) and RT-pipeline sets agree.

struct RayTracingParamsGLSL {
    vec4 toggles;      // x=reflections y=refractions z=thickness w=localShadows
    vec4 distances;    // x=maxReflect y=maxRefract z=maxShadowDist w=roughnessThreshold
    vec4 water;        // x=IOR, y=maxWaterThickness (hit clamp), zw reserved
    vec4 absorption;   // rgb=Beer-Lambert coeff, a=thicknessScale
    vec4 debug;        // x=RT debug view, y=tlasReady, z=selfSkipDist, w=useWaterPipeline
    mat4 invViewProj;
    vec4 viewPos;
    vec4 rtResolution; // xy=size, zw=1/size
    vec4 clipPlanes;   // x=near, y=far
    vec4 sunDir;       // xyz=direction TO sun
    vec4 sunColor;
};

struct RTProxyMetaGLSL {
    vec4 minAndMatId;  // xyz=AABB min, w=material id
    vec4 maxAndFlags;  // xyz=AABB max, w=flags
    vec4 albedoRough;  // rgb=avg albedo, a=roughness
    vec4 extra;
};

// Proxy layout: solid boxes occupy metadata slots [0, RT_WATER_BOX_START),
// water volumes [RT_WATER_BOX_START, ...). The TLAS carries one instance per
// layer (custom index 0 = solid, 1 = water); instance masks select layers per
// ray type (solids see all, water-originated rays see solids only).
const uint RT_WATER_BOX_START = 3584u;
const uint RT_RAY_MASK_SOLID = 0x01u;
const uint RT_RAY_MASK_WATER = 0x02u;
const uint RT_RAY_MASK_ALL = 0x03u;

// Refraction "deep water" sentinel. A ray that misses the terrestrial proxies
// has still travelled through the water volume (unresolved exit); encoding it
// as an in-band path length (maxRefract) over-attenuates Beer-Lambert to
// black. This marker lets water.frag substitute the deep-water tint instead
// (see SceneRenderer::updateRTParams / rtMaxWaterThickness).
const float RT_DEEP_WATER = 1e30;

// Global metadata index for a triangle hit (primitiveID is per-BLAS local).
uint rtBoxIndex(uint primitiveId, uint instanceCustomIndex) {
    return primitiveId / 12u + (instanceCustomIndex == 1u ? RT_WATER_BOX_START : 0u);
}

// Schlick Fresnel with dielectric base reflectance.
float rtSchlickFresnel(float cosTheta, float f0) {
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0);
}

// Analytic box-face normal for a proxy hit (closest face to hitPos).
// flipForExit negates it when the ray exits the volume (backface hit).
vec3 rtBoxNormal(vec3 hitPos, vec3 boxMin, vec3 boxMax, bool flipForExit) {
    vec3 dMin = hitPos - boxMin;
    vec3 dMax = boxMax - hitPos;
    vec3 n = vec3(0.0);
    float m = dMin.x;
    m = min(m, dMin.y);
    m = min(m, dMin.z);
    m = min(m, dMax.x);
    m = min(m, dMax.y);
    m = min(m, dMax.z);
    if (m == dMin.x) n = vec3(-1.0, 0.0, 0.0);
    else if (m == dMax.x) n = vec3(1.0, 0.0, 0.0);
    else if (m == dMin.y) n = vec3(0.0, -1.0, 0.0);
    else if (m == dMax.y) n = vec3(0.0, 1.0, 0.0);
    else if (m == dMin.z) n = vec3(0.0, 0.0, -1.0);
    else n = vec3(0.0, 0.0, 1.0);
    return flipForExit ? -n : n;
}

// Slab-test exit distance for ray (origin, dir) through box. Returns the
// parametric exit t (>= hitT when origin is outside or on the surface).
float rtBoxExitT(vec3 origin, vec3 dir, vec3 boxMin, vec3 boxMax) {
    vec3 invD = 1.0 / max(abs(dir), vec3(1e-8)) * sign(dir + vec3(1e-9));
    vec3 t0 = (boxMin - origin) * invD;
    vec3 t1 = (boxMax - origin) * invD;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    return max(max(tmax3.x, tmax3.y), tmax3.z);
}

// Equirectangular UV for a direction (matches postprocess dirToEquirectUV).
vec2 rtDirToEquirectUV(vec3 dir) {
    const float PI = 3.14159265358979;
    vec2 uv;
    uv.x = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return uv;
}

// Procedural sky gradient fallback (solid miss when no sky texture is bound).
// Matches the Sky gradient mode (horizon/zenith) approximately.
vec3 rtProceduralSky(vec3 dir, vec3 horizon, vec3 zenith, float exponent) {
    float h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(horizon, zenith, pow(h, max(exponent, 0.01)));
}
