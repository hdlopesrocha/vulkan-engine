# Video assets

Source clips for the engine's video textures. Each loaded clip owns one slot of
the dedicated video texture array, i.e. an animated material selected through
`MaterialProperties::textureIndex`.

The engine only loads **converted** clips: square 1024×1024, H.264 with a short
keyframe interval, tagged `engine-video-texture`. Raw masters never reach a
runtime folder.

## Layout

| Path | Committed | Purpose |
|---|---|---|
| `base/` | yes | The three default **converted** clips. `MyApp` loads them at setup; they become animated video materials. |
| `extra/` | no (only `.gitkeep`) | Optional local **converted** library. |
| `raw/base/` | no | Raw masters for the default clips (local conversion inputs). |
| `raw/extra/` | no | Raw masters for the extra library (local conversion inputs; can be very large). |

`raw/` is ignored by git. The runtime folders `base/` and `extra/` must contain
converted clips only; `scripts/convert_videos.sh` keeps it that way.

## Converting

```sh
scripts/convert_videos.sh          # videos/raw/base -> videos/base, then stage bin/videos
scripts/convert_videos.sh --extra  # also videos/raw/extra -> videos/extra
scripts/convert_videos.sh --stage  # copy converted clips to bin/videos (no conversion)
scripts/convert_videos.sh --force  # re-encode even already converted clips
scripts/convert_videos.sh --clean  # remove bin/videos
```

`make videos` / `make videos-extra` are thin wrappers around the same script.
A normal `make` / `make debug` runs **`--stage`** automatically, so
`bin/videos/` always contains the converted clips and the app never needs to
decode a raw master. Staging is a copy (and only when a clip changes), so it
needs no ffmpeg and adds no build time worth mentioning.

Raw clips dropped directly into `videos/base/` or `videos/extra/` are converted
in place; the script moves the master to `videos/raw/<category>/` first (any
container), so the runtime folder only ever holds converted clips and no master
is lost.

Defaults (override through the environment):

| Setting | Default |
|---|---|
| `VIDEO_SIZE` | `1024` (matches the default texture-array size) |
| `VIDEO_CRF` | `26` |
| `VIDEO_PRESET` | `veryfast` |

Conversion also crops to a centered square, drops audio, sets
`-g 30 -keyint_min 15 -sc_threshold 0` (≈0.5 s seek latency at 30 fps) and
`+faststart` for immediate playback.

## Runtime lookup and loading more clips

The app normally runs from `bin/`, so the relative paths resolve as:

| Relative path | Contents |
|---|---|
| `videos/base/`, `videos/extra/` | staged converted clips (preferred) |
| `../videos/base/`, `../videos/extra/` | converted clips in the repository |

Both sources contain converted clips only. `VideoTextureService::listAvailableVideos()`
scans the staged folders first and deduplicates by name. The lookup is **not
recursive** beyond those folders: keep basenames unique across `base/` and
`extra/`.

`MyApp` loads the three `base/` clips into video slots at setup. The Video
Textures widget lists every discovered clip and can load/unload clips at runtime
without a restart, up to the video-array capacity (`VULKAN_VIDEO_SLOTS`, default
8):

1. Put the raw clip(s) in `videos/raw/extra/`.
2. Run `scripts/convert_videos.sh --extra`.
3. Open the Video Textures widget and press **Load** next to the clip.

In the **Video Textures** widget, select a loaded clip to play/pause it, change
rate or looping, seek, inspect the decoder/diagnostics and unload it. To use a
clip in the scene, set a material's kind to **Video** in the Materials widget and
pick the clip's slot as its texture index (the video layer renders it with the
material's blend mode and texture-coordinate technique).
