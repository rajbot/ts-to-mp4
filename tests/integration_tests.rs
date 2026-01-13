//! Integration tests for ts-to-mp4 remuxing
//!
//! These tests use real MPEG-TS files to verify the remuxing process works correctly.

use std::io::Cursor;

/// Load the 738x720 test fixture (video with frame cropping)
fn load_738x720_fixture() -> Vec<u8> {
    include_bytes!("fixtures/test_738x720.ts").to_vec()
}

/// Load the test fixture with audio (video + audio streams)
fn load_with_audio_fixture() -> Vec<u8> {
    include_bytes!("fixtures/test_with_audio.ts").to_vec()
}

/// NAL unit type constants
const NAL_TYPE_SPS: u8 = 7;
const NAL_TYPE_PPS: u8 = 8;
const NAL_TYPE_IDR: u8 = 5;

/// Find NAL units in AVCC format (length-prefixed) data
fn find_avcc_nal_units(data: &[u8]) -> Vec<(u8, usize, usize)> {
    let mut units = Vec::new();
    let mut offset = 0;

    while offset + 4 < data.len() {
        let length = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;

        if length == 0 || offset + 4 + length > data.len() {
            break;
        }

        let nal_type = data[offset + 4] & 0x1F;
        units.push((nal_type, offset + 4, offset + 4 + length));
        offset += 4 + length;
    }

    units
}

/// Parse MP4 boxes to extract mdat content
fn extract_mdat(mp4_data: &[u8]) -> Option<&[u8]> {
    let mut offset = 0;

    while offset + 8 <= mp4_data.len() {
        let size = u32::from_be_bytes([
            mp4_data[offset],
            mp4_data[offset + 1],
            mp4_data[offset + 2],
            mp4_data[offset + 3],
        ]) as usize;

        let box_type = &mp4_data[offset + 4..offset + 8];

        if box_type == b"mdat" {
            return Some(&mp4_data[offset + 8..offset + size]);
        }

        if size == 0 {
            break;
        }
        offset += size;
    }

    None
}

/// Extract video track dimensions from moov/trak/tkhd box
fn extract_video_dimensions(mp4_data: &[u8]) -> Option<(u32, u32)> {
    // Find moov box
    let mut offset = 0;
    while offset + 8 <= mp4_data.len() {
        let size = u32::from_be_bytes([
            mp4_data[offset],
            mp4_data[offset + 1],
            mp4_data[offset + 2],
            mp4_data[offset + 3],
        ]) as usize;

        let box_type = &mp4_data[offset + 4..offset + 8];

        if box_type == b"moov" {
            return find_tkhd_dimensions(&mp4_data[offset + 8..offset + size]);
        }

        if size == 0 {
            break;
        }
        offset += size;
    }
    None
}

fn find_tkhd_dimensions(moov_data: &[u8]) -> Option<(u32, u32)> {
    let mut offset = 0;
    while offset + 8 <= moov_data.len() {
        let size = u32::from_be_bytes([
            moov_data[offset],
            moov_data[offset + 1],
            moov_data[offset + 2],
            moov_data[offset + 3],
        ]) as usize;

        let box_type = &moov_data[offset + 4..offset + 8];

        if box_type == b"trak" {
            if let Some(dims) = find_tkhd_in_trak(&moov_data[offset + 8..offset + size]) {
                // Only return if dimensions are non-zero (video track, not audio)
                if dims.0 > 0 && dims.1 > 0 {
                    return Some(dims);
                }
            }
        }

        if size == 0 {
            break;
        }
        offset += size;
    }
    None
}

fn find_tkhd_in_trak(trak_data: &[u8]) -> Option<(u32, u32)> {
    let mut offset = 0;
    while offset + 8 <= trak_data.len() {
        let size = u32::from_be_bytes([
            trak_data[offset],
            trak_data[offset + 1],
            trak_data[offset + 2],
            trak_data[offset + 3],
        ]) as usize;

        let box_type = &trak_data[offset + 4..offset + 8];

        if box_type == b"tkhd" {
            // tkhd box: version (1) + flags (3) + creation_time (4) + modification_time (4)
            // + track_id (4) + reserved (4) + duration (4) + reserved (8) + layer (2)
            // + alternate_group (2) + volume (2) + reserved (2) + matrix (36) + width (4) + height (4)
            // Width and height are at offset 84 from box start (including header)
            let width_offset = offset + 8 + 76; // 8 for header, 76 for fields before width
            let height_offset = width_offset + 4;

            if height_offset + 4 <= trak_data.len() {
                // Width and height are in 16.16 fixed point format
                let width_fixed = u32::from_be_bytes([
                    trak_data[width_offset],
                    trak_data[width_offset + 1],
                    trak_data[width_offset + 2],
                    trak_data[width_offset + 3],
                ]);
                let height_fixed = u32::from_be_bytes([
                    trak_data[height_offset],
                    trak_data[height_offset + 1],
                    trak_data[height_offset + 2],
                    trak_data[height_offset + 3],
                ]);

                return Some((width_fixed >> 16, height_fixed >> 16));
            }
        }

        if size == 0 {
            break;
        }
        offset += size;
    }
    None
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_remux_738x720_video() {
    // This test verifies remuxing works for a video with non-standard dimensions
    // that require proper SPS parsing (752x720 macroblocks with 14-pixel right crop = 738x720)
    let ts_data = load_738x720_fixture();
    let mut output = Cursor::new(Vec::new());

    let result = ts_to_mp4::remux(Cursor::new(ts_data), &mut output);
    assert!(result.is_ok(), "Remux should succeed: {:?}", result.err());

    let mp4_data = output.into_inner();
    assert!(!mp4_data.is_empty(), "Output should not be empty");

    // Verify it starts with ftyp box
    assert_eq!(&mp4_data[4..8], b"ftyp", "Should start with ftyp box");
}

#[test]
fn test_738x720_dimensions_parsed_correctly() {
    // This test verifies that SPS parsing extracts correct dimensions
    // The video has 752x720 macroblocks but 14-pixel right crop -> 738x720 display
    let ts_data = load_738x720_fixture();
    let mut output = Cursor::new(Vec::new());

    ts_to_mp4::remux(Cursor::new(ts_data), &mut output).expect("Remux should succeed");
    let mp4_data = output.into_inner();

    let dims = extract_video_dimensions(&mp4_data);
    assert!(dims.is_some(), "Should find video dimensions");

    let (width, height) = dims.unwrap();
    assert_eq!(width, 738, "Width should be 738 (752 - 14 crop)");
    assert_eq!(height, 720, "Height should be 720");
}

#[test]
fn test_sps_pps_included_inline() {
    // This test verifies that SPS and PPS NAL units are included inline in the video stream.
    // Some decoders (including QuickTime) require inline SPS/PPS before keyframes.
    let ts_data = load_738x720_fixture();
    let mut output = Cursor::new(Vec::new());

    ts_to_mp4::remux(Cursor::new(ts_data), &mut output).expect("Remux should succeed");
    let mp4_data = output.into_inner();

    // Extract mdat box (contains video/audio data)
    let mdat = extract_mdat(&mp4_data).expect("Should have mdat box");

    // Parse NAL units from mdat
    let nal_units = find_avcc_nal_units(mdat);

    // Count NAL types
    let sps_count = nal_units.iter().filter(|(t, _, _)| *t == NAL_TYPE_SPS).count();
    let pps_count = nal_units.iter().filter(|(t, _, _)| *t == NAL_TYPE_PPS).count();
    let idr_count = nal_units.iter().filter(|(t, _, _)| *t == NAL_TYPE_IDR).count();

    assert!(sps_count > 0, "Should have at least one inline SPS NAL unit");
    assert!(pps_count > 0, "Should have at least one inline PPS NAL unit");
    assert!(idr_count > 0, "Should have at least one IDR frame");

    // SPS should appear before the first IDR
    let first_sps_idx = nal_units.iter().position(|(t, _, _)| *t == NAL_TYPE_SPS);
    let first_idr_idx = nal_units.iter().position(|(t, _, _)| *t == NAL_TYPE_IDR);

    if let (Some(sps_idx), Some(idr_idx)) = (first_sps_idx, first_idr_idx) {
        assert!(
            sps_idx < idr_idx,
            "SPS should appear before first IDR frame"
        );
    }
}

#[test]
fn test_mp4_has_required_boxes() {
    // Verify the output MP4 has all required boxes
    let ts_data = load_738x720_fixture();
    let mut output = Cursor::new(Vec::new());

    ts_to_mp4::remux(Cursor::new(ts_data), &mut output).expect("Remux should succeed");
    let mp4_data = output.into_inner();

    // Parse top-level boxes
    let mut boxes = Vec::new();
    let mut offset = 0;

    while offset + 8 <= mp4_data.len() {
        let size = u32::from_be_bytes([
            mp4_data[offset],
            mp4_data[offset + 1],
            mp4_data[offset + 2],
            mp4_data[offset + 3],
        ]) as usize;

        let box_type = String::from_utf8_lossy(&mp4_data[offset + 4..offset + 8]).to_string();
        boxes.push(box_type);

        if size == 0 {
            break;
        }
        offset += size;
    }

    assert!(boxes.contains(&"ftyp".to_string()), "Should have ftyp box");
    assert!(boxes.contains(&"mdat".to_string()), "Should have mdat box");
    assert!(boxes.contains(&"moov".to_string()), "Should have moov box");
}

#[test]
fn test_remux_produces_valid_output_size() {
    // Verify the output is reasonable size (not empty, not bloated)
    let ts_data = load_738x720_fixture();
    let ts_size = ts_data.len();
    let mut output = Cursor::new(Vec::new());

    ts_to_mp4::remux(Cursor::new(ts_data), &mut output).expect("Remux should succeed");
    let mp4_data = output.into_inner();

    // MP4 should be roughly similar size to TS (remuxing, not transcoding)
    // Allow for some overhead from MP4 container structure
    assert!(
        mp4_data.len() > ts_size / 2,
        "Output should be at least half the input size"
    );
    assert!(
        mp4_data.len() < ts_size * 2,
        "Output should be less than double the input size"
    );
}

// ============================================================================
// Unit Tests for Helper Functions
// ============================================================================

#[test]
fn test_find_avcc_nal_units_parsing() {
    // Test AVCC NAL unit parsing with known data
    // Format: 4-byte length (big endian) + NAL data
    let data = [
        0x00, 0x00, 0x00, 0x05, // Length: 5
        0x67, 0x64, 0x00, 0x1f, 0xac, // SPS NAL (type 7)
        0x00, 0x00, 0x00, 0x03, // Length: 3
        0x68, 0xeb, 0xe3, // PPS NAL (type 8)
    ];

    let units = find_avcc_nal_units(&data);
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].0, NAL_TYPE_SPS);
    assert_eq!(units[1].0, NAL_TYPE_PPS);
}

#[test]
fn test_find_avcc_nal_units_empty() {
    let data: [u8; 0] = [];
    let units = find_avcc_nal_units(&data);
    assert!(units.is_empty());
}

#[test]
fn test_find_avcc_nal_units_truncated() {
    // Truncated data should not panic
    let data = [0x00, 0x00, 0x00, 0x10, 0x67]; // Claims 16 bytes but only has 1
    let units = find_avcc_nal_units(&data);
    assert!(units.is_empty());
}

// ============================================================================
// FFmpeg Comparison Tests
// ============================================================================

/// Extract frame timing (PTS values) from an MP4 file using ffprobe
fn extract_frame_pts(mp4_path: &std::path::Path) -> Vec<i64> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-select_streams", "v:0",
            "-show_entries", "frame=pts",
            "-of", "csv=p=0",
        ])
        .arg(mp4_path)
        .output()
        .expect("ffprobe command failed - is ffmpeg installed?");

    assert!(output.status.success(), "ffprobe failed: {}", String::from_utf8_lossy(&output.stderr));

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<i64>().ok())
        .collect()
}

/// Extract frame DTS values from an MP4 file using ffprobe
fn extract_frame_dts(mp4_path: &std::path::Path) -> Vec<i64> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_dts",
            "-of", "csv=p=0",
        ])
        .arg(mp4_path)
        .output()
        .expect("ffprobe command failed - is ffmpeg installed?");

    assert!(output.status.success(), "ffprobe failed: {}", String::from_utf8_lossy(&output.stderr));

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<i64>().ok())
        .collect()
}

/// Extract audio frame PTS values from an MP4 file using ffprobe
fn extract_audio_pts(mp4_path: &std::path::Path) -> Vec<i64> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-select_streams", "a:0",
            "-show_entries", "frame=pts",
            "-of", "csv=p=0",
        ])
        .arg(mp4_path)
        .output()
        .expect("ffprobe command failed - is ffmpeg installed?");

    assert!(output.status.success(), "ffprobe failed: {}", String::from_utf8_lossy(&output.stderr));

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<i64>().ok())
        .collect()
}

#[test]
fn test_frame_timing_matches_ffmpeg() {
    // This test verifies that ts-to-mp4 produces identical frame timing to ffmpeg.
    // Requires ffmpeg/ffprobe to be installed.

    let ts_data = load_738x720_fixture();
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    let ts_path = temp_dir.path().join("input.ts");
    let ts2mp4_path = temp_dir.path().join("ts2mp4.mp4");
    let ffmpeg_path = temp_dir.path().join("ffmpeg.mp4");

    // Write input TS file
    std::fs::write(&ts_path, &ts_data).expect("Failed to write TS file");

    // Convert with ts-to-mp4
    let mut output = Cursor::new(Vec::new());
    ts_to_mp4::remux(Cursor::new(ts_data), &mut output).expect("ts-to-mp4 remux failed");
    std::fs::write(&ts2mp4_path, output.into_inner()).expect("Failed to write ts2mp4 output");

    // Convert with ffmpeg (stream copy)
    let ffmpeg_result = std::process::Command::new("ffmpeg")
        .args(["-i"])
        .arg(&ts_path)
        .args(["-c", "copy", "-y"])
        .arg(&ffmpeg_path)
        .output()
        .expect("ffmpeg command failed - is ffmpeg installed?");

    assert!(ffmpeg_result.status.success(),
        "ffmpeg conversion failed: {}",
        String::from_utf8_lossy(&ffmpeg_result.stderr));

    // Extract and compare PTS values
    let ts2mp4_pts = extract_frame_pts(&ts2mp4_path);
    let ffmpeg_pts = extract_frame_pts(&ffmpeg_path);

    assert_eq!(ts2mp4_pts.len(), ffmpeg_pts.len(),
        "Frame count mismatch: ts-to-mp4 has {} frames, ffmpeg has {} frames",
        ts2mp4_pts.len(), ffmpeg_pts.len());

    assert!(!ts2mp4_pts.is_empty(), "Should have at least one frame");

    for (i, (ts2mp4, ffmpeg)) in ts2mp4_pts.iter().zip(ffmpeg_pts.iter()).enumerate() {
        assert_eq!(ts2mp4, ffmpeg,
            "PTS mismatch at frame {}: ts-to-mp4={}, ffmpeg={}",
            i, ts2mp4, ffmpeg);
    }

    // Extract and compare DTS values
    let ts2mp4_dts = extract_frame_dts(&ts2mp4_path);
    let ffmpeg_dts = extract_frame_dts(&ffmpeg_path);

    assert_eq!(ts2mp4_dts.len(), ffmpeg_dts.len(),
        "DTS frame count mismatch");

    for (i, (ts2mp4, ffmpeg)) in ts2mp4_dts.iter().zip(ffmpeg_dts.iter()).enumerate() {
        assert_eq!(ts2mp4, ffmpeg,
            "DTS mismatch at frame {}: ts-to-mp4={}, ffmpeg={}",
            i, ts2mp4, ffmpeg);
    }
}

/// Compute deltas (intervals) between consecutive timestamps
fn compute_deltas(timestamps: &[i64]) -> Vec<i64> {
    timestamps.windows(2).map(|w| w[1] - w[0]).collect()
}

#[test]
fn test_audio_timing_matches_ffmpeg() {
    // This test verifies that ts-to-mp4 produces identical audio/video timing intervals to ffmpeg.
    // Uses a synthetic test file with both video and audio streams.
    // Note: ts-to-mp4 normalizes timestamps to start at 0, while ffmpeg preserves original offsets.
    // We compare timing *deltas* (intervals) rather than absolute values.
    // Requires ffmpeg/ffprobe to be installed.

    let ts_data = load_with_audio_fixture();
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    let ts_path = temp_dir.path().join("input_audio.ts");
    let ts2mp4_path = temp_dir.path().join("ts2mp4_audio.mp4");
    let ffmpeg_path = temp_dir.path().join("ffmpeg_audio.mp4");

    // Write input TS file
    std::fs::write(&ts_path, &ts_data).expect("Failed to write TS file");

    // Convert with ts-to-mp4
    let mut output = Cursor::new(Vec::new());
    ts_to_mp4::remux(Cursor::new(ts_data), &mut output).expect("ts-to-mp4 remux failed");
    std::fs::write(&ts2mp4_path, output.into_inner()).expect("Failed to write ts2mp4 output");

    // Convert with ffmpeg (stream copy)
    let ffmpeg_result = std::process::Command::new("ffmpeg")
        .args(["-i"])
        .arg(&ts_path)
        .args(["-c", "copy", "-y"])
        .arg(&ffmpeg_path)
        .output()
        .expect("ffmpeg command failed - is ffmpeg installed?");

    assert!(ffmpeg_result.status.success(),
        "ffmpeg conversion failed: {}",
        String::from_utf8_lossy(&ffmpeg_result.stderr));

    // Extract and compare audio PTS deltas (intervals between frames)
    let ts2mp4_audio_pts = extract_audio_pts(&ts2mp4_path);
    let ffmpeg_audio_pts = extract_audio_pts(&ffmpeg_path);

    assert_eq!(ts2mp4_audio_pts.len(), ffmpeg_audio_pts.len(),
        "Audio frame count mismatch: ts-to-mp4 has {} frames, ffmpeg has {} frames",
        ts2mp4_audio_pts.len(), ffmpeg_audio_pts.len());

    assert!(!ts2mp4_audio_pts.is_empty(), "Should have at least one audio frame");

    let ts2mp4_audio_deltas = compute_deltas(&ts2mp4_audio_pts);
    let ffmpeg_audio_deltas = compute_deltas(&ffmpeg_audio_pts);

    for (i, (ts2mp4, ffmpeg)) in ts2mp4_audio_deltas.iter().zip(ffmpeg_audio_deltas.iter()).enumerate() {
        assert_eq!(ts2mp4, ffmpeg,
            "Audio PTS delta mismatch at frame {}: ts-to-mp4={}, ffmpeg={}",
            i, ts2mp4, ffmpeg);
    }

    // Verify video timing deltas for the audio test file
    let ts2mp4_video_pts = extract_frame_pts(&ts2mp4_path);
    let ffmpeg_video_pts = extract_frame_pts(&ffmpeg_path);

    assert_eq!(ts2mp4_video_pts.len(), ffmpeg_video_pts.len(),
        "Video frame count mismatch in audio test file: ts-to-mp4 has {} frames, ffmpeg has {} frames",
        ts2mp4_video_pts.len(), ffmpeg_video_pts.len());

    let ts2mp4_video_deltas = compute_deltas(&ts2mp4_video_pts);
    let ffmpeg_video_deltas = compute_deltas(&ffmpeg_video_pts);

    for (i, (ts2mp4, ffmpeg)) in ts2mp4_video_deltas.iter().zip(ffmpeg_video_deltas.iter()).enumerate() {
        assert_eq!(ts2mp4, ffmpeg,
            "Video PTS delta mismatch at frame {}: ts-to-mp4={}, ffmpeg={}",
            i, ts2mp4, ffmpeg);
    }
}
