# Video assets

Source clips for the engine's video textures. Each loaded clip owns one slot of
the dedicated video texture array, i.e. an animated material selected through
`MaterialProperties::textureIndex`.

## Layout

| Path | Committed | Purpose |
|---|---|---|
| `base/` | yes | The three shortest source clips. `MyApp` loads them by default at setup; they become animated video materials. |
| `extra/` | no (only `.gitkeep`) | Optional/local source clips. They can be very large, so they are not committed; users drop their own clips here. |
| `*.mp4` (flat) | no | Converted clips produced by `make videos` / `make videos-extra`. They live next to the source folders and are ignored by `videos/.gitignore`. |

`videos/.gitignore` ignores the flat converted `*.mp4` and `extra/*` while
keeping `extra/.gitkeep`; the committed sources in `base/` are **not** ignored.

## Converting

```sh
make videos        # videos/base/*  -> videos/*.mp4
make videos-extra  # videos/extra/* -> videos/*.mp4
```

Both targets use the same settings:

* square 1024x1024 center crop;
* H.264 (`libx264`, `-crf 26 -preset veryfast`), `yuv420p`, no audio;
* short GOP: `-g 30 -keyint_min 15 -sc_threshold 0`;
* `+faststart` for immediate playback.

An asset is re-encoded only when its source is newer, so `make videos` is a
no-op once `videos/*.mp4` is up to date. The normal build copies the converted
clips into `bin/videos/` next to the executable.

`make videos-clean` deletes the converted `videos/*.mp4` files only; it never
touches the `base/` and `extra/` sources.

## Runtime lookup and loading more clips

Converted clips live flat in `videos/`, so the runtime checks `videos/` and
`../videos/` (next to the executable and in the repository root); the loader
then scans `videos/base` and `videos/extra` for the source clips and prefers
the converted `videos/<name>.mp4` when present. The lookup is **not
recursive**: keep basenames unique across `base/` and `extra/`.

`MyApp` loads the three `base/` clips into video slots at setup. The Video
Textures widget lists every discovered clip and can load/unload clips at runtime
without a restart, up to the video-array capacity (`VULKAN_VIDEO_SLOTS`, default
8):

1. Put the source clip(s) in `videos/extra/` (or drop an already-converted
   `.mp4` directly into `videos/`).
2. Run `make videos-extra` (or `make videos` for a base clip).
3. Open the Video Textures widget and press **Load** next to the clip.

In the **Video Textures** widget, select a loaded clip to play/pause it, change
rate or looping, seek, inspect the decoder/diagnostics and unload it. To use a
clip in the scene, set a material's kind to **Video** in the Materials widget and
pick the clip's slot as its texture index (the video layer renders it with the
material's blend mode and texture-coordinate technique).
