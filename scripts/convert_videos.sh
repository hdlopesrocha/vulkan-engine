#!/bin/bash
# Convert video clips into texture-ready engine assets.
#
# The engine only ever loads CONVERTED clips: square, short-GOP H.264 with the
# "engine-video-texture" metadata tag written by this script. Raw masters stay
# out of the runtime folders:
#
#   videos/raw/base/*  (optional local masters, ignored by git)
#   videos/raw/extra/* (optional local masters, ignored by git)
#        │
#        ├─ scripts/convert_videos.sh ──> videos/base/*.mp4   (committed)
#        └─ scripts/convert_videos.sh --extra ──> videos/extra/*.mp4 (ignored)
#
# A raw clip dropped directly into videos/base or videos/extra is converted in
# place, with the master preserved under videos/raw/<cat>/ first.
# `make` runs `--stage`, which copies the converted clips into bin/videos/ so
# the app always sees texture-ready assets; the build itself never invokes
# ffmpeg.
#
# Usage:
#   scripts/convert_videos.sh            convert videos/base, then stage bin/videos
#   scripts/convert_videos.sh --extra    also convert videos/extra
#   scripts/convert_videos.sh --all      same as --extra
#   scripts/convert_videos.sh --stage    copy converted clips only (used by make)
#   scripts/convert_videos.sh --force    re-encode even already converted clips
#   scripts/convert_videos.sh --clean    remove bin/videos
#
# Environment:
#   VIDEO_SIZE    output edge length (default 1024, the texture-array size)
#   VIDEO_CRF     x264 quality (default 26)
#   VIDEO_PRESET  x264 preset (default veryfast)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIDEO_SIZE="${VIDEO_SIZE:-1024}"
VIDEO_CRF="${VIDEO_CRF:-26}"
VIDEO_PRESET="${VIDEO_PRESET:-veryfast}"
VIDEO_MARKER="engine-video-texture"
VIDEO_EXTS=(mp4 mov mkv webm)
BASE_DIR="videos/base"
EXTRA_DIR="videos/extra"
RAW_BASE_DIR="videos/raw/base"
RAW_EXTRA_DIR="videos/raw/extra"
STAGE_DIR="bin/videos"

usage() {
    cat <<'EOF'
Convert video clips into texture-ready engine assets.

Usage:
  scripts/convert_videos.sh            convert videos/base, then stage bin/videos
  scripts/convert_videos.sh --extra    also convert videos/extra
  scripts/convert_videos.sh --all      same as --extra
  scripts/convert_videos.sh --stage    copy converted clips only (used by make)
  scripts/convert_videos.sh --force    re-encode even already converted clips
  scripts/convert_videos.sh --clean    remove bin/videos

Environment:
  VIDEO_SIZE    output edge length (default 1024)
  VIDEO_CRF     x264 quality (default 26)
  VIDEO_PRESET  x264 preset (default veryfast)
EOF
}

do_base=1
do_extra=0
stage_only=0
force=0

while (($# > 0)); do
    case "$1" in
        --base)   do_base=1; do_extra=0 ;;
        --extra)  do_extra=1 ;;
        --all)    do_base=1; do_extra=1 ;;
        --stage)  stage_only=1 ;;
        --force)  force=1 ;;
        --clean)  rm -rf "$STAGE_DIR"; echo "Removed $STAGE_DIR"; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# A failed/aborted conversion must not leave a hidden temp file next to the
# assets (e.g. when the script is killed while ffmpeg is running).
TMP_FILE=""
cleanup_tmp() { [[ -n "$TMP_FILE" ]] && rm -f "$TMP_FILE"; return 0; }
trap cleanup_tmp EXIT
have_ffmpeg=0
command -v ffmpeg >/dev/null 2>&1 && have_ffmpeg=1
have_ffprobe=0
command -v ffprobe >/dev/null 2>&1 && have_ffprobe=1

# True when the file carries the conversion tag. Without ffprobe we cannot
# inspect the tag, so callers trust the folder convention (the committed clips
# in videos/base are converted by definition).
is_converted() {
    local file=$1
    (( have_ffprobe )) || return 0
    local comment
    comment="$(ffprobe -v error -show_entries format_tags=comment \
        -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null | head -n1 || true)"
    [[ "$comment" == "$VIDEO_MARKER" ]]
}

# Null-separated list of video files directly in a folder (sorted, no find error
# when the folder does not exist).
list_videos() {
    local dir=$1
    [[ -d "$dir" ]] || return 0
    local args=()
    local ext
    for ext in "${VIDEO_EXTS[@]}"; do
        args+=(-o -name "*.$ext")
    done
    find "$dir" -maxdepth 1 -type f \( "${args[@]:1}" \) -print0 2>/dev/null | sort -z
}

# convert_one <source> <destination.mp4>
convert_one() {
    local src=$1 dst=$2
    local dir stem tmp
    dir="$(dirname "$dst")"
    stem="$(basename "$dst" .mp4)"
    mkdir -p "$dir"
    tmp="$(mktemp "$dir/.$stem.XXXXXX")"
    TMP_FILE="$tmp"
    echo "Converting video texture: $src -> $dst"
    if ! ffmpeg -nostdin -hide_banner -loglevel error -y -i "$src" -map_metadata -1 \
        -vf "crop=min(iw\,ih):min(iw\,ih),scale=$VIDEO_SIZE:$VIDEO_SIZE:flags=lanczos" \
        -an -c:v libx264 -preset "$VIDEO_PRESET" -crf "$VIDEO_CRF" \
        -g 30 -keyint_min 15 -sc_threshold 0 -pix_fmt yuv420p \
        -movflags +faststart -metadata comment="$VIDEO_MARKER" -f mp4 "$tmp"; then
        rm -f "$tmp"
        TMP_FILE=""
        echo "error: failed to convert $src" >&2
        return 1
    fi
    mv -f "$tmp" "$dst"
    TMP_FILE=""
    chmod 644 "$dst"
    echo "  -> $dst ($(du -h "$dst" | cut -f1))"
}

# process_category <name> <converted-folder> <raw-folder>
process_category() {
    local name=$1 dir=$2 raw_dir=$3
    local src dst base file

    # 1. Clips dropped directly into the runtime folder: convert in place.
    while IFS= read -r -d '' src; do
        if (( force )) || ! is_converted "$src"; then
            case "$src" in
                *.mp4)
                    # Preserve the master under videos/raw/<category>/ before
                    # replacing the runtime-file copy with the converted clip.
                    file="$(basename "$src")"
                    mkdir -p "$raw_dir"
                    if [[ ! -f "$raw_dir/$file" ]]; then
                        mv -f "$src" "$raw_dir/$file"
                        convert_one "$raw_dir/$file" "$src"
                    else
                        convert_one "$src" "$src"
                    fi
                    ;;
                *)
                    # Keep the original container as a local master instead of
                    # leaving a non-converted file in the runtime folder.
                    dst="$dir/$(basename "${src%.*}").mp4"
                    convert_one "$src" "$dst"
                    mkdir -p "$raw_dir"
                    mv -f "$src" "$raw_dir/"
                    echo "  raw master moved to $raw_dir/"
                    ;;
            esac
        fi
    done < <(list_videos "$dir")

    # 2. Local masters in videos/raw/<category>/: convert when the runtime copy
    #    is missing, stale, not converted, or --force was given.
    while IFS= read -r -d '' src; do
        base="$(basename "${src%.*}")"
        dst="$dir/$base.mp4"
        if (( force )) || [[ ! -f "$dst" ]] || ! is_converted "$dst" || [[ "$src" -nt "$dst" ]]; then
            convert_one "$src" "$dst"
        fi
    done < <(list_videos "$raw_dir")
}

# stage_category <converted-folder> <category-subdir-of-bin>
stage_category() {
    local dir=$1 name=$2 dst_dir="$STAGE_DIR/$2" src dst copied=0 fresh=0 skipped=0 pruned=0
    [[ -d "$dir" ]] || return 0
    mkdir -p "$dst_dir"

    while IFS= read -r -d '' src; do
        [[ "$src" == *.mp4 ]] || continue
        if ! is_converted "$src"; then
            echo "warning: skipping non-converted clip '$src' (run scripts/convert_videos.sh)" >&2
            skipped=$((skipped + 1))
            continue
        fi
        dst="$dst_dir/$(basename "$src")"
        if [[ ! -f "$dst" ]] || [[ "$src" -nt "$dst" ]]; then
            cp -p "$src" "$dst"
            copied=$((copied + 1))
        else
            fresh=$((fresh + 1))
        fi
    done < <(list_videos "$dir")

    # Drop clips that no longer exist in the source folder so a removed clip
    # cannot linger in bin/videos and appear in the widget.
    for dst in "$dst_dir"/*; do
        [[ -f "$dst" ]] || continue
        if [[ ! -f "$dir/$(basename "$dst")" ]]; then
            rm -f "$dst"
            pruned=$((pruned + 1))
        fi
    done

    echo "Staged $name: $copied copied, $fresh up to date, $skipped skipped, $pruned pruned"
}

# ---------------------------------------------------------------------------
# Convert / stage
# ---------------------------------------------------------------------------
if (( stage_only )); then
    stage_category "$BASE_DIR" base
    stage_category "$EXTRA_DIR" extra
    exit 0
fi

if (( have_ffmpeg == 0 )); then
    echo "error: ffmpeg is required to convert videos (apt install ffmpeg)" >&2
    exit 1
fi

if (( have_ffprobe == 0 )); then
    echo "warning: ffprobe not found; already converted clips cannot be detected" >&2
fi

if (( do_base )); then
    process_category base "$BASE_DIR" "$RAW_BASE_DIR"
fi
if (( do_extra )); then
    process_category extra "$EXTRA_DIR" "$RAW_EXTRA_DIR"
fi

stage_category "$BASE_DIR" base
stage_category "$EXTRA_DIR" extra
