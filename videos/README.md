# Video assets

Source clips for the engine's video textures. Each loaded clip owns one slot of
the dedicated video texture array, i.e. an animated material selected through
`MaterialProperties::textureIndex`.

## Layout

| Path | Committed | Purpose |
|---|---|---|
| `base/` | yes | The three shortest source clips. `MyApp` loads them by default at setup; they become animated video materials. |
| `extra/` | no (only `.gitkeep`) | Optional/local source clips. They can be very large, so they are not committed; users drop their own clips here. |

Converted clips are **not** written into this folder: `make videos` puts them
directly under `bin/videos/base/` and `bin/videos/extra/`, next to the
executable. `videos/.gitignore` only ignores the optional `extra/` library.

## Converting

```sh
make videos        # videos/base/*  -> bin/videos/base/*.mp4
make videos-extra  # videos/extra/* -> bin/videos/extra/*.mp4
```

Both targets use the same settings:

* square 1024x1024 center crop;
* H.264 (`libx264`, `-crf 26 -preset veryfast`), `yuv420p`, no audio;
* short GOP: `-g 30 -keyint_min 15 -sc_threshold 0`;
* `+faststart` for immediate playback.

An asset is re-encoded only when its source is newer, so `make videos` is a
no-op once `bin/videos/base/` is up to date. `make clean` removes `bin/`, so run
`make videos` again afterwards.

`make videos-clean` deletes the converted `bin/videos/base/` and
`bin/videos/extra/` folders only; it never touches the `base/` and `extra/`
sources.

## Runtime lookup and loading more clips

The app normally runs from `bin/`, so the relative paths resolve as:

| Relative path | Contents |
|---|---|
| `videos/base/`, `videos/extra/` | converted clips (preferred) |
| `../videos/base/`, `../videos/extra/` | source clips in the repository |

`VideoTextureService::listAvailableVideos()` scans the converted folders first
and deduplicates by name, so the optimized clips win when both exist. The lookup
is **not recursive** beyond those folders: keep basenames unique across `base/`
and `extra/`.

`MyApp` loads the three `base/` clips into video slots at setup. The Video
Textures widget lists every discovered clip and can load/unload clips at runtime
without a restart, up to the video-array capacity (`VULKAN_VIDEO_SLOTS`, default
8):

1. Put the source clip(s) in `videos/extra/`.
2. Run `make videos-extra` (or `make videos` for a base clip).
3. Open the Video Textures widget and press **Load** next to the clip.

In the **Video Textures** widget, select a loaded clip to play/pause it, change
rate or looping, seek, inspect the decoder/diagnostics and unload it. To use a
clip in the scene, set a material's kind to **Video** in the Materials widget and
pick the clip's slot as its texture index (the video layer renders it with the
material's blend mode and texture-coordinate technique).
