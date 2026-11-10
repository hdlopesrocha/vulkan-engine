#!/bin/sh
# Speak dynamic alert text (e.g. the current line count) through the same
# queue as the checked-in alert sounds.
#
# Usage: say.sh <text...>
#        say.sh --save <out.mp3> <text...>   synthesize only, no playback
#
# `make cloc` and `make commit` use the first form because the spoken text only
# exists at run time and cannot come from a pre-rendered mp3. The phrase is
# synthesized with the same Edge neural voice as the checked-in alerts
# (en-US-AriaNeural by default, override with SOUND_VOICE) and cached under
# .dynamic/, so repeated runs are instant. `make sounds` uses --save to render
# the checked-in mp3s through this exact pipeline.
#
# There is deliberately no flite/espeak/spd-say fallback: a missing or broken
# engine leaves the alert silent instead of degrading the voice to something
# robotic.
#
# Synthesis happens before playback, and playback is serialized on the same
# queue lock as every other alert, so dynamic speech never overlaps the
# checked-in phase voices. Callers background this script; it blocks as the
# drainer until the queue is empty.

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 0
cache=$dir/.dynamic
voice=${SOUND_VOICE:-en-US-AriaNeural}
rate=${SOUND_TTS_RATE:-0}

save=
if [ "${1:-}" = "--save" ]; then
	save=${2:-}
	[ -n "$save" ] || exit 1
	shift 2
fi
text=$*
[ -n "$text" ] || exit 0
mkdir -p "$cache" || exit 0

# Locate a Python that can import edge-tts. Prefer an explicit override
# (SOUND_TTS_PYTHON), then pipx's venv (the documented install), then an
# edge-tts launcher's shebang (SOUND_EDGE_TTS or the usual paths), then any
# python on PATH. The launcher is not executed directly because older
# edge-tts versions still point at the legacy host that returns 403 in some
# regions; edge_tts_say.py retries the new host first.
find_python() {
	for candidate in \
		"${SOUND_TTS_PYTHON:-}" \
		"${PIPX_HOME:-${HOME:-/nonexistent}/.local/share/pipx}/venvs/edge-tts/bin/python"; do
		[ -n "$candidate" ] || continue
		[ -x "$candidate" ] || continue
		"$candidate" -c 'import edge_tts' >/dev/null 2>&1 || continue
		printf '%s' "$candidate"
		return 0
	done
	for launcher in "${SOUND_EDGE_TTS:-}" \
		"$(command -v edge-tts 2>/dev/null)" \
		"${HOME:-/nonexistent}/.local/bin/edge-tts"; do
		[ -n "$launcher" ] || continue
		[ -x "$launcher" ] || continue
		candidate=$(sed -n '1s/^#!//p' "$launcher")
		[ -n "$candidate" ] || continue
		[ -x "$candidate" ] || continue
		"$candidate" -c 'import edge_tts' >/dev/null 2>&1 || continue
		printf '%s' "$candidate"
		return 0
	done
	for candidate in "$(command -v python3 2>/dev/null)" \
		"$(command -v python 2>/dev/null)"; do
		[ -n "$candidate" ] || continue
		[ -x "$candidate" ] || continue
		"$candidate" -c 'import edge_tts' >/dev/null 2>&1 || continue
		printf '%s' "$candidate"
		return 0
	done
	return 1
}

die() {
	printf 'sounds: %s\n' "$*" >&2
	exit 1
}

python=$(find_python) || python=
if [ -z "$python" ]; then
	[ -z "$save" ] && exit 0
	die "edge-tts is not installed for any Python (pipx install edge-tts, or set SOUND_TTS_PYTHON)"
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
	[ -z "$save" ] && exit 0
	die "ffmpeg is required to encode alert audio (apt install ffmpeg)"
fi

# MD5 of voice+rate+text names the cache file (the unit separator avoids
# collisions between the parts).
cache_key() {
	printf '%s\037%s\037%s' "$1" "$2" "$text" | md5sum | cut -d' ' -f1
}

# encode <out> <ffmpeg input args...>: normalize to the same loudness and
# format as the checked-in alert sounds (mono 24 kHz 64 kbps MP3). The target
# variable is named dst because POSIX sh functions have no local variables.
encode() {
	dst=$1
	shift
	ffmpeg -hide_banner -loglevel error -y "$@" \
		-af "loudnorm=I=-16:TP=-1.5:LRA=11" \
		-ac 1 -ar 24000 -c:a libmp3lame -b:a 64k "$dst" \
		</dev/null >/dev/null 2>&1
}

# synthesize <dst>: Edge neural voice -> temporary raw mp3 -> normalized mp3.
# No fallback voice: the caller decides whether silence is acceptable.
synthesize() {
	target=$1
	raw=$cache/.raw.$$.mp3
	tmp=$cache/.enc.$$.mp3
	printf '%s' "$text" | "$python" "$dir/edge_tts_say.py" \
		--voice "$voice" --rate "$rate" --out "$raw" || {
		rm -f "$raw" "$tmp"
		return 1
	}
	if [ ! -s "$raw" ] || ! encode "$tmp" -i "$raw" || [ ! -s "$tmp" ]; then
		rm -f "$raw" "$tmp"
		return 1
	fi
	rm -f "$raw"
	mv -f "$tmp" "$target"
}

if [ -n "$save" ]; then
	synthesize "$save" || die "could not synthesize the alert with $voice"
	exit 0
fi

out=$cache/$(cache_key "$voice" "$rate").mp3
if [ ! -s "$out" ]; then
	synthesize "$out" || exit 0
fi
[ -s "$out" ] || exit 0

exec sh "$dir/play.sh" "$out"
