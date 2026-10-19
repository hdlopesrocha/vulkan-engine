#ifndef BLEND_MODES_GLSL
#define BLEND_MODES_GLSL

// -----------------------------------------------------------------------------
// BlendModes.glsl
//
// Photoshop/W3C-style blend modes for fragment shaders.
//
// Mode index map (must match the CPU enum `MaterialBlendMode` in
// types/MaterialProperties.hpp):
//
//   0  Normal        10 Overlay
//   1  Darken        11 Difference
//   2  Multiply      12 Exclusion
//   3  ColorBurn     13 Subtract
//   4  LinearBurn    14 Divide
//   5  DarkerColor   15 Hue
//   6  Lighten       16 Saturation
//   7  Screen        17 Color
//   8  ColorDodge    18 Luminosity
//   9  LighterColor
//
// All functions take non-premultiplied colors in [0,1] and return [0,1].
// Alpha (source-over compositing) is the caller's responsibility. Modes marked
// non-separable below operate in HSL space per the W3C compositing spec.
// -----------------------------------------------------------------------------

const float BLEND_EPSILON = 1e-5;

// ---------------------------------------------------------------------------
// Separable blend modes
// ---------------------------------------------------------------------------

// 0: Normal - source replaces base (source-over color).
vec3 blendNormal(vec3 base, vec3 blend)
{
    return clamp(blend, 0.0, 1.0);
}

// 1: Darken - per-channel minimum.
vec3 blendDarken(vec3 base, vec3 blend)
{
    return min(base, blend);
}

// 2: Multiply - per-channel product.
vec3 blendMultiply(vec3 base, vec3 blend)
{
    return base * blend;
}

// 3: ColorBurn - darkens base by the inverse of blend.
vec3 blendColorBurn(vec3 base, vec3 blend)
{
    vec3 safeBlend = max(blend, vec3(BLEND_EPSILON));
    vec3 result = 1.0 - min(vec3(1.0), (1.0 - base) / safeBlend);
    // b == 0 burns to pure black.
    result = mix(result, vec3(0.0), step(blend, vec3(0.0)));
    return clamp(result, 0.0, 1.0);
}

// 4: LinearBurn - per-channel additive darken.
vec3 blendLinearBurn(vec3 base, vec3 blend)
{
    return clamp(base + blend - 1.0, 0.0, 1.0);
}

// 5: DarkerColor - returns the color with the lower luminance.
vec3 blendDarkerColor(vec3 base, vec3 blend)
{
    float baseLum  = dot(base,  vec3(0.3, 0.59, 0.11));
    float blendLum = dot(blend, vec3(0.3, 0.59, 0.11));
    return mix(blend, base, step(baseLum, blendLum));
}

// 6: Lighten - per-channel maximum.
vec3 blendLighten(vec3 base, vec3 blend)
{
    return max(base, blend);
}

// 7: Screen - inverse multiply.
vec3 blendScreen(vec3 base, vec3 blend)
{
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

// 8: ColorDodge - brightens base by the inverse of blend.
vec3 blendColorDodge(vec3 base, vec3 blend)
{
    vec3 safeInverse = max(1.0 - blend, vec3(BLEND_EPSILON));
    vec3 result = min(vec3(1.0), base / safeInverse);
    // b == 1 dodges to pure white.
    result = mix(result, vec3(1.0), step(vec3(1.0), blend));
    return clamp(result, 0.0, 1.0);
}

// 9: LighterColor - returns the color with the higher luminance.
vec3 blendLighterColor(vec3 base, vec3 blend)
{
    float baseLum  = dot(base,  vec3(0.3, 0.59, 0.11));
    float blendLum = dot(blend, vec3(0.3, 0.59, 0.11));
    return mix(base, blend, step(baseLum, blendLum));
}

// 10: Overlay - multiply on shadows, screen on highlights.
vec3 blendOverlay(vec3 base, vec3 blend)
{
    vec3 low  = 2.0 * base * blend;
    vec3 high = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
    return mix(high, low, step(base, vec3(0.5)));
}

// 11: Difference - absolute per-channel difference.
vec3 blendDifference(vec3 base, vec3 blend)
{
    return abs(base - blend);
}

// 12: Exclusion - lower-contrast difference.
vec3 blendExclusion(vec3 base, vec3 blend)
{
    return base + blend - 2.0 * base * blend;
}

// 13: Subtract - per-channel base minus blend.
vec3 blendSubtract(vec3 base, vec3 blend)
{
    return max(base - blend, vec3(0.0));
}

// 14: Divide - per-channel base divided by blend.
vec3 blendDivide(vec3 base, vec3 blend)
{
    return clamp(base / max(blend, vec3(BLEND_EPSILON)), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Non-separable (HSL) helpers, per W3C compositing spec
// ---------------------------------------------------------------------------

// Relative luminance weights used by the HSL modes and the color comparisons.
float lum(vec3 c)
{
    return dot(c, vec3(0.3, 0.59, 0.11));
}

// Clamp the color to [0,1] while preserving its luminance.
vec3 clipColor(vec3 c)
{
    float l = lum(c);
    float n = min(min(c.r, c.g), c.b);
    float x = max(max(c.r, c.g), c.b);

    if (n < 0.0)
    {
        c = l + (c - l) * l / max(l - n, BLEND_EPSILON);
    }
    if (x > 1.0)
    {
        c = l + (c - l) * (1.0 - l) / max(x - l, BLEND_EPSILON);
    }

    return clamp(c, 0.0, 1.0);
}

// Replace the luminance of c with l, keeping its hue and saturation.
vec3 setLum(vec3 c, float l)
{
    float d = l - lum(c);
    return clipColor(c + d);
}

// HSL saturation of a color.
float sat(vec3 c)
{
    return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
}

// Replace the HSL saturation of c with s, keeping its hue and lightness.
vec3 setSat(vec3 c, float s)
{
    float cMin = min(min(c.r, c.g), c.b);
    float cMid = c.r + c.g + c.b - cMin - max(max(c.r, c.g), c.b);
    float cMax = max(max(c.r, c.g), c.b);
    float range = cMax - cMin;

    if (range > BLEND_EPSILON)
    {
        cMid = (cMid - cMin) * s / range;
        cMax = s;
    }
    else
    {
        cMid = 0.0;
        cMax = 0.0;
    }

    // Restore the per-channel ordering (min/mid/max positions).
    vec3 result;
    if (c.r >= c.g && c.r >= c.b)
    {
        result.r = cMax;
        if (c.g >= c.b)
        {
            result.g = cMid;
            result.b = 0.0;
        }
        else
        {
            result.g = 0.0;
            result.b = cMid;
        }
    }
    else if (c.g >= c.r && c.g >= c.b)
    {
        result.g = cMax;
        if (c.r >= c.b)
        {
            result.r = cMid;
            result.b = 0.0;
        }
        else
        {
            result.r = 0.0;
            result.b = cMid;
        }
    }
    else
    {
        result.b = cMax;
        if (c.r >= c.g)
        {
            result.r = cMid;
            result.g = 0.0;
        }
        else
        {
            result.r = 0.0;
            result.g = cMid;
        }
    }

    return clamp(result, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Non-separable blend modes
// ---------------------------------------------------------------------------

// 15: Hue - take the hue of blend, saturation and luminance of base.
vec3 blendHue(vec3 base, vec3 blend)
{
    return setLum(setSat(blend, sat(base)), lum(base));
}

// 16: Saturation - take the saturation of blend, hue and luminance of base.
vec3 blendSaturation(vec3 base, vec3 blend)
{
    return setLum(setSat(base, sat(blend)), lum(base));
}

// 17: Color - take hue and saturation of blend, luminance of base.
vec3 blendColor(vec3 base, vec3 blend)
{
    return setLum(blend, lum(base));
}

// 18: Luminosity - take the luminance of blend, hue and saturation of base.
vec3 blendLuminosity(vec3 base, vec3 blend)
{
    return setLum(base, lum(blend));
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

// Selects the blend mode by its MaterialBlendMode index. Returns the blended
// color; the caller remains responsible for alpha compositing.
vec3 blendModeColor(int mode, vec3 base, vec3 blend)
{
    switch (mode)
    {
        case 0:  return blendNormal(base, blend);
        case 1:  return blendDarken(base, blend);
        case 2:  return blendMultiply(base, blend);
        case 3:  return blendColorBurn(base, blend);
        case 4:  return blendLinearBurn(base, blend);
        case 5:  return blendDarkerColor(base, blend);
        case 6:  return blendLighten(base, blend);
        case 7:  return blendScreen(base, blend);
        case 8:  return blendColorDodge(base, blend);
        case 9:  return blendLighterColor(base, blend);
        case 10: return blendOverlay(base, blend);
        case 11: return blendDifference(base, blend);
        case 12: return blendExclusion(base, blend);
        case 13: return blendSubtract(base, blend);
        case 14: return blendDivide(base, blend);
        case 15: return blendHue(base, blend);
        case 16: return blendSaturation(base, blend);
        case 17: return blendColor(base, blend);
        case 18: return blendLuminosity(base, blend);
        default: return blendNormal(base, blend);
    }
}

// Inline alias used by material blending code paths.
vec3 applyMaterialBlend(int mode, vec3 base, vec3 blendColor)
{
    return blendModeColor(mode, base, blendColor);
}

#endif // BLEND_MODES_GLSL
