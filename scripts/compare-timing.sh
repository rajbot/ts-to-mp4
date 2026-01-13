#!/bin/bash
# Compare frame timing between ts-to-mp4 and ffmpeg outputs
#
# Usage: ./scripts/compare-timing.sh input.ts

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <input.ts>"
    exit 1
fi

INPUT="$1"
BASENAME=$(basename "$INPUT" .ts)
TMPDIR=$(mktemp -d)

echo "=== Comparing frame timing for: $INPUT ==="
echo "Temp directory: $TMPDIR"
echo

# Convert with ts-to-mp4
echo "Converting with ts-to-mp4..."
cargo run --features cli --quiet -- "$INPUT" -o "$TMPDIR/${BASENAME}_ts2mp4.mp4"

# Convert with ffmpeg (stream copy, no transcoding)
echo "Converting with ffmpeg..."
ffmpeg -i "$INPUT" -c copy -y "$TMPDIR/${BASENAME}_ffmpeg.mp4" 2>/dev/null

# Extract video frame timing
echo
echo "Extracting video frame timing..."

ffprobe -select_streams v:0 -show_entries frame=pts,pkt_dts,pict_type \
    -of csv=p=0 "$TMPDIR/${BASENAME}_ts2mp4.mp4" 2>/dev/null > "$TMPDIR/ts2mp4_video.csv"

ffprobe -select_streams v:0 -show_entries frame=pts,pkt_dts,pict_type \
    -of csv=p=0 "$TMPDIR/${BASENAME}_ffmpeg.mp4" 2>/dev/null > "$TMPDIR/ffmpeg_video.csv"

# Extract audio frame timing
echo "Extracting audio frame timing..."

ffprobe -select_streams a:0 -show_entries frame=pts,pkt_dts \
    -of csv=p=0 "$TMPDIR/${BASENAME}_ts2mp4.mp4" 2>/dev/null > "$TMPDIR/ts2mp4_audio.csv" || true

ffprobe -select_streams a:0 -show_entries frame=pts,pkt_dts \
    -of csv=p=0 "$TMPDIR/${BASENAME}_ffmpeg.mp4" 2>/dev/null > "$TMPDIR/ffmpeg_audio.csv" || true

# Compare results
echo
echo "=== VIDEO FRAME COMPARISON ==="
echo "ts-to-mp4 frames: $(wc -l < "$TMPDIR/ts2mp4_video.csv")"
echo "ffmpeg frames:    $(wc -l < "$TMPDIR/ffmpeg_video.csv")"
echo

if diff -q "$TMPDIR/ts2mp4_video.csv" "$TMPDIR/ffmpeg_video.csv" > /dev/null 2>&1; then
    echo "VIDEO: IDENTICAL timing"
else
    echo "VIDEO: DIFFERENCES found"
    echo
    echo "First 10 differences:"
    diff --side-by-side --width=80 "$TMPDIR/ts2mp4_video.csv" "$TMPDIR/ffmpeg_video.csv" | head -20
    echo
    echo "Summary of differences:"
    diff "$TMPDIR/ts2mp4_video.csv" "$TMPDIR/ffmpeg_video.csv" | grep -c "^[<>]" || echo "0"
    echo "lines differ"
fi

echo
echo "=== AUDIO FRAME COMPARISON ==="
if [ -s "$TMPDIR/ts2mp4_audio.csv" ] && [ -s "$TMPDIR/ffmpeg_audio.csv" ]; then
    echo "ts-to-mp4 frames: $(wc -l < "$TMPDIR/ts2mp4_audio.csv")"
    echo "ffmpeg frames:    $(wc -l < "$TMPDIR/ffmpeg_audio.csv")"
    echo

    if diff -q "$TMPDIR/ts2mp4_audio.csv" "$TMPDIR/ffmpeg_audio.csv" > /dev/null 2>&1; then
        echo "AUDIO: IDENTICAL timing"
    else
        echo "AUDIO: DIFFERENCES found"
        echo
        echo "First 10 differences:"
        diff --side-by-side --width=80 "$TMPDIR/ts2mp4_audio.csv" "$TMPDIR/ffmpeg_audio.csv" | head -20
    fi
else
    echo "No audio track or audio extraction failed"
fi

# Show detailed timing info
echo
echo "=== DETAILED TIMING INFO ==="
echo
echo "ts-to-mp4 video (first 5 frames): pts,dts,type"
head -5 "$TMPDIR/ts2mp4_video.csv"
echo
echo "ffmpeg video (first 5 frames): pts,dts,type"
head -5 "$TMPDIR/ffmpeg_video.csv"

# Show stream info
echo
echo "=== STREAM INFO ==="
echo
echo "ts-to-mp4:"
ffprobe -show_entries stream=index,codec_name,codec_type,time_base,duration_ts,duration \
    -of compact "$TMPDIR/${BASENAME}_ts2mp4.mp4" 2>/dev/null | grep -v "^$"
echo
echo "ffmpeg:"
ffprobe -show_entries stream=index,codec_name,codec_type,time_base,duration_ts,duration \
    -of compact "$TMPDIR/${BASENAME}_ffmpeg.mp4" 2>/dev/null | grep -v "^$"

# Save files for manual inspection
echo
echo "=== OUTPUT FILES ==="
echo "ts-to-mp4 output: $TMPDIR/${BASENAME}_ts2mp4.mp4"
echo "ffmpeg output:    $TMPDIR/${BASENAME}_ffmpeg.mp4"
echo "Video timing CSV: $TMPDIR/ts2mp4_video.csv, $TMPDIR/ffmpeg_video.csv"
echo "Audio timing CSV: $TMPDIR/ts2mp4_audio.csv, $TMPDIR/ffmpeg_audio.csv"
