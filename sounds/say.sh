#!/bin/sh
# Speak dynamic alert text (e.g. the current line count) through the same
# queue as the checked-in alert sounds.
#
# Usage: say.sh <text...>
#
# `make cloc` uses this because the spoken number only exists at run time and
# cannot come from a pre-rendered mp3. The phrase is synthesized with edge-tts
# (same voice as `make sounds`, loudness-normalized the same way) and cached
# under .dynamic/, so repeated runs are instant. When edge-tts is missing,
# broken or offline, ffmpeg's built-in flite voice is used instead (robotic
# but exact).
# If neither can synthesize the phrase nothing is played: PLAY_SOUND has
# already announced the phase, so the silence is graceful.
#
# Synthesis happens before playback, and playback is serialized on the same
# queue lock as every other alert, so dynamic speech never overlaps the
# checked-in phase voices. Callers background this script; it blocks as the
# drainer until the queue is empty.

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 0
cache=$dir/.dynamic
voice=${SOUND_VOICE:-en-US-AvaMultilingualNeural}
text=$*
[ -n "$text" ] || exit 0
mkdir -p "$cache" || exit 0

# MD5 of method+voice+text names the cache file (the unit separator avoids
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

# Preferred path: edge-tts neural synthesis. Resolve the binary instead of
# trusting `command -v` alone: pipx installs it in ~/.local/bin (not always on
# the PATH of non-login shells), and a stale symlink left behind by a removed
# venv makes `command -v` return nothing. That silently degraded every error
# announcement to the robotic flite voice, so probe --version and skip a
# broken install.
find_edge_tts() {
	for candidate in "${SOUND_EDGE_TTS:-}" \
		"$(command -v edge-tts 2>/dev/null)" \
		"${HOME:-/nonexistent}/.local/bin/edge-tts"; do
		[ -n "$candidate" ] || continue
		[ -x "$candidate" ] || continue
		"$candidate" --version >/dev/null 2>&1 || continue
		printf '%s' "$candidate"
		return 0
	done
	return 1
}

out=
edge=$(find_edge_tts) || edge=
if [ -n "$edge" ]; then
	out=$cache/$(cache_key edge "$voice").mp3
	if [ ! -s "$out" ]; then
		raw=$cache/.raw.$$.mp3
		tmp=$cache/.enc.$$.mp3
		"$edge" --voice "$voice" --text "$text" --write-media "$raw" </dev/null >/dev/null 2>&1
		if [ -s "$raw" ] && encode "$tmp" -i "$raw" && [ -s "$tmp" ]; then
			mv -f "$tmp" "$out"
		fi
		rm -f "$raw" "$tmp"
	fi
	[ -s "$out" ] || out=
fi

# Offline fallback: ffmpeg's flite voice. The text goes through a temp
# textfile because flite's textfile option avoids filtergraph escaping issues
# with multi-word text.
if [ -z "$out" ] && command -v ffmpeg >/dev/null 2>&1; then
	out=$cache/$(cache_key flite slt).mp3
	if [ ! -s "$out" ]; then
		txt=$cache/.txt.$$
		tmp=$cache/.enc.$$.mp3
		printf '%s' "$text" > "$txt"
		if encode "$tmp" -f lavfi -i "flite=textfile=$txt:voice=slt" && [ -s "$tmp" ]; then
			mv -f "$tmp" "$out"
		fi
		rm -f "$txt" "$tmp"
	fi
	[ -s "$out" ] || out=
fi

[ -n "$out" ] && [ -s "$out" ] || exit 0

exec sh "$dir/play.sh" "$out"
