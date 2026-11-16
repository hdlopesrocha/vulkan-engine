#pragma once

// WindRaymarchRenderer settings.
//
// Renders the shared 3D fBm wind velocity as a raymarched isosurface: the
// shape is the set of world points where |windVelocity(p)| equals the target
// speed. The target speed is the primary adjustable parameter; the march
// budget/range and the surface opacity control cost and readability.
struct WindRaymarchSettings {
    // Isosurface speed (m/s): the rendered shape is |wind| == targetSpeed.
    // Raise it to show only the strong-gust/tornado cores; lower it to show
    // the broader flow structure.
    float targetSpeed = 12.0f;

    // Furthest distance marched per pixel (m). The scene depth clips the march
    // first, so foreground geometry occludes the volume.
    float maxDistance = 400.0f;

    // March samples per pixel (quality vs cost, clamped to [8, 128]).
    int steps = 48;

    // Surface alpha (0..1).
    float opacity = 0.85f;
};
