#pragma once

// Wind-trail particle visualization settings (WindRenderer).
//
// The simulation itself lives on the GPU (shaders/computation/wind/
// WindParticles.comp): every particle samples the SAME shared wind field the
// vegetation/fire/clouds use (set=0 binding 27, vulkan/ubo/WindField.hpp) and
// leaves a fading, time-stamped trail. Only presentation is configured here.
//
// Wind parameters are NOT duplicated in this struct: the shared 3D fBm field
// (frequency/octaves/lacunarity/gain/amplitude/vertical/seed) and the tornado
// list stay owned by VegetationRenderer (WindSettings + WindFieldSettings), so
// the widgets edit the one shared source of truth.
struct WindParticlesSettings {
    // Simulated particles. Clamped to WindRenderer::kMaxParticles by the
    // renderer; the GPU dispatch and the trail draw both use this count.
    int particleCount = 2000;

    // How long a trail lives (seconds). The trail is sampled at a FIXED rate
    // (WindRenderer::kTrailSampleInterval), not once per frame, so this is an
    // absolute lifetime independent of the frame rate.
    float trailDuration = 2.0f;

    // Speed -> color ramp: trails are colored by the local wind speed using a
    // 4-stop ramp (blue = slow -> green -> yellow -> red = fast). This is the
    // speed (m/s) at which the ramp reaches full red; lower it to make the
    // field look "hotter", raise it to keep everything blue/green. Defaults to
    // the default fBm amplitude (16 m/s) so the whole ramp is used.
    float colorSpeedMax = 16.0f;

    // Peak alpha of a trail.
    float opacity = 0.65f;

    // Trail width in screen pixels (screen-space expanded quads, so any
    // hardware supports it without the wideLines feature).
    float lineWidth = 2.0f;

    // Particle band along the view ray (metres from the camera):
    //   spawnNearDistance = closest respawn distance (also the only near bound:
    //     particles may drift closer while they live),
    //   spawnFarDistance  = the particle FAR PLANE: respawns happen inside
    //     [near, far] and a particle that drifts past it is respawned inside
    //     the band. Clamped on the host to [0, Settings::farPlane] (the camera
    //     far plane), every frame.
    float spawnNearDistance = 8.0f;
    float spawnFarDistance = 900.0f;
};
