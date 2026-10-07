// HSV conversion utilities (H in [0,360], S/V in [0,1])

// Convert HSV to RGB (H in [0,360], S/V in [0,1])
vec3 hsvToRgb(vec3 hsv) {
    vec3 c = clamp(hsv, vec3(0.0), vec3(360.0, 1.0, 1.0));
    float h = c.x / 60.0;
    float s = c.y;
    float v = c.z;
    float hi = floor(h);
    float f = h - hi;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    int i = int(hi) % 6;
    if (i == 0) return vec3(v, t, p);
    if (i == 1) return vec3(q, v, p);
    if (i == 2) return vec3(p, v, t);
    if (i == 3) return vec3(p, q, v);
    if (i == 4) return vec3(t, p, v);
    return vec3(v, p, q);
}

// Convert RGB to HSV (H in [0,360], S/V in [0,1])
vec3 rgbToHsv(vec3 rgb) {
    vec3 c = rgb;
    float cmax = max(c.r, max(c.g, c.b));
    float cmin = min(c.r, min(c.g, c.b));
    float delta = cmax - cmin;
    float v = cmax;
    float s = cmax == 0.0 ? 0.0 : delta / cmax;
    float h = 0.0;
    if (delta > 0.0001) {
        if (cmax == c.r)
            h = 60.0 * mod((c.g - c.b) / delta, 6.0);
        else if (cmax == c.g)
            h = 60.0 * ((c.b - c.r) / delta + 2.0);
        else
            h = 60.0 * ((c.r - c.g) / delta + 4.0);
        if (h < 0.0) h += 360.0;
    }
    return vec3(h, s, v);
}
