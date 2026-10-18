#!/bin/sh
# Alert-sound queue: enqueue / drain.
#
# Usage: play.sh --enqueue <sound.mp3>   register a sound (fast, synchronous)
#        play.sh --drain                 play queued sounds (long-running)
#        play.sh <sound.mp3>             enqueue + drain (manual use)
#
# Requests are queued as uniquely named entries in .queue.d and played
# strictly one at a time in request-time order: entry names start with a
# nanosecond timestamp, so a directory sort is chronological.
#
# Enqueue is synchronous on purpose. The Makefile PLAY_SOUND macro and the
# app's setup alerts (playSetupAlert in MyApp.cpp) call it in program order
# and only background the drainer, so queue order matches request order even
# with several producers (make and the app share this one queue and play no
# voices in parallel). Stored paths are absolute so any drainer can play
# them regardless of its working directory.
#
# A single player lock (.queue.lock) makes sure only one process plays at a
# time; waiting happens in the background, so builds and the app never block
# on audio. A drainer that finds an entry created after its last look wakes
# up its own caller, which is blocked on the lock, so no request is lost.
#
# The first available player wins: mpv, ffplay, mpg123, cvlc. When none is
# installed the queue is cleared, so requests never pile up.

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 0
qdir=$dir/.queue.d
lock=$dir/.queue.lock

mode=drain
file=
if [ $# -ge 1 ]; then
	case $1 in
	--enqueue) mode=enqueue; file=${2:-} ;;
	--drain)   mode=drain ;;
	*)         mode=both; file=$1 ;;
	esac
fi

player=
for p in mpv ffplay mpg123 cvlc; do
	if command -v "$p" >/dev/null 2>&1; then
		player=$p
		break
	fi
done
[ -n "$player" ] || { rm -f "$qdir"/* 2>/dev/null; exit 0; }

# Enqueue (modes: enqueue, both): canonicalize to an absolute path because
# the entry may be drained by a process with a different working directory.
if [ -n "$file" ]; then
	[ -f "$file" ] || exit 0
	mkdir -p "$qdir" || exit 0
	abs=$(CDPATH= cd -- "$(dirname -- "$file")" && pwd)/$(basename -- "$file") || exit 0
	printf '%s' "$abs" > "$qdir/$(date +%s%N).$$" || exit 0
	[ "$mode" = enqueue ] && exit 0
fi

# Players run in the foreground: waiting for one to finish is what serializes
# the queue (callers background this script, not the player).
play() {
	case $player in
	mpv)    mpv --really-quiet --no-video "$1" ;;
	ffplay) ffplay -nodisp -autoexit -loglevel quiet "$1" ;;
	mpg123) mpg123 -q "$1" ;;
	cvlc)   cvlc --play-and-exit --intf dummy "$1" ;;
	esac
}

# Degraded mode without flock (uncommon outside util-linux): no
# serialization, but entries are claimed by rename so none is played twice.
if ! command -v flock >/dev/null 2>&1; then
	while :; do
		entry=$(ls "$qdir" 2>/dev/null | sort | head -n 1)
		[ -n "$entry" ] || break
		claimed=$qdir/.$entry.$$
		mv "$qdir/$entry" "$claimed" 2>/dev/null || continue
		path=$(cat "$claimed" 2>/dev/null)
		rm -f "$claimed"
		[ -n "$path" ] || continue
		play "$path"
	done
	exit 0
fi

# Become the player (blocking) and drain in timestamp order. The lock is
# held across playback, so concurrent callers wait their turn instead of
# overlapping.
exec 9>"$lock" || exit 0
flock 9 || exit 0

while :; do
	entry=$(ls "$qdir" 2>/dev/null | sort | head -n 1)
	[ -n "$entry" ] || break
	path=$(cat "$qdir/$entry" 2>/dev/null)
	rm -f "$qdir/$entry"
	[ -n "$path" ] || continue
	play "$path"
done
