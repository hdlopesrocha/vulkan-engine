# Video assets

Source clips for the engine's video textures. Each loaded clip owns one slot of
the dedicated video texture array, i.e. an animated material selected through
`MaterialProperties::textureIndex`.

## Layout

| Path | Committed | Purpose |
|---|---|---|
| `base/` | yes | The three shortest clips. `MyApp` loads them by default at setup; they become animated brush/video materials. |
| `extra/` | no (only `.gitkeep`) | Optional/local clips. They can be very large, so they are not committed; users drop their own clips here. |

`videos/.gitignore` ignores `extra/*` but keeps `extra/.gitkeep`, so the folder
itself is tracked. `base/*.mp4` are **not** ignored and must stay committed.

## Converting

```sh
make videos        # videos/base/*  -> video_textures/*.mp4
make videos-extra  # videos/extra/* -> video_textures/*.mp4
```

Both targets use the same settings:

* square 1024x1024 center crop;
* H.264 (`libx264`, `-crf 26 -preset veryfast`), `yuv420p`, no audio;
* short GOP: `-g 30 -keyint_min 15 -sc_threshold 0`;
* `+faststart` for immediate playback.

An asset is re-encoded only when its source is newer, so `make videos` is a
no-op once `video_textures/` is up to date. The normal build copies the whole
`video_textures/` directory into `bin/` next to the executable.
`make videos-clean` deletes the converted assets in `video_textures/` only; it
never deletes sources.

## Runtime lookup and loading more clips

`VideoTextureService::findAssetDirectory()` checks, in order:

1. `video_textures/`
2. `../video_textures/`
3. `videos/`
4. `../videos/`

The first directory that contains at least one supported video file wins, so
converted assets are preferred over raw sources, and `make run` (from `bin/`)
and a manual run from the repository root both work. The lookup is **not
recursive**, and `video_textures/` is flat: keep basenames unique across
`base/` and `extra/`.

`MyApp` loads the three `base/` clips into video slots at setup. The Video
Textures widget lists every discovered clip and can load/unload clips at runtime
without a restart, up to the video-array capacity (`VULKAN_VIDEO_SLOTS`, default
8):

1. Put the source clip(s) in `videos/extra/` (or drop an already-converted
   `.mp4` directly into `video_textures/`).
2. Run `make videos-extra` (or `make videos` for a base clip).
3. Open the Video Textures widget and press **Load** next to the clip.

In the **Video Textures** widget, select a loaded clip to play/pause it, change
rate or looping, seek, inspect the decoder/diagnostics and unload it. To use a
clip in the scene, set a material's kind to **Video** in the Materials widget and
pick the clip's slot as its texture index (the video layer renders it with the
material's blend mode and texture-coordinate technique).
