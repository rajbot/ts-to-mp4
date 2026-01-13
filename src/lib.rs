//! # ts-to-mp4
//!
//! Remux MPEG-TS to MP4 without transcoding (stream copy).
//!
//! This library extracts H.264 video and AAC audio from MPEG-TS containers
//! and repackages them into MP4 format, similar to `ffmpeg -c copy`.

use std::io::{Read, Seek, Write};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid TS packet: {0}")]
    InvalidTs(String),
    #[error("No video stream found")]
    NoVideoStream,
    #[error("No audio stream found")]
    NoAudioStream,
    #[error("Invalid H.264 data: {0}")]
    InvalidH264(String),
    #[error("Invalid AAC data: {0}")]
    InvalidAac(String),
}

pub type Result<T> = std::result::Result<T, Error>;

// ============================================================================
// MPEG-TS Demuxer
// ============================================================================

const TS_PACKET_SIZE: usize = 188;
const TS_SYNC_BYTE: u8 = 0x47;
const PAT_PID: u16 = 0x0000;

/// Stream type identifiers from PMT
const STREAM_TYPE_H264: u8 = 0x1B;
const STREAM_TYPE_AAC: u8 = 0x0F;
const STREAM_TYPE_AAC_LATM: u8 = 0x11;

#[derive(Debug, Clone)]
struct ProgramInfo {
    pmt_pid: u16,
    video_pid: Option<u16>,
    audio_pid: Option<u16>,
}

#[derive(Debug)]
struct PesPacket {
    pts: Option<u64>,
    dts: Option<u64>,
    data: Vec<u8>,
}

/// Demux an MPEG-TS stream into H.264 and AAC elementary streams
struct TsDemuxer<R> {
    reader: R,
    program: Option<ProgramInfo>,
    video_pes_buffer: Vec<u8>,
    audio_pes_buffer: Vec<u8>,
    video_pes_started: bool,
    audio_pes_started: bool,
}

impl<R: Read> TsDemuxer<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            program: None,
            video_pes_buffer: Vec::new(),
            audio_pes_buffer: Vec::new(),
            video_pes_started: false,
            audio_pes_started: false,
        }
    }

    /// Demux the entire stream and return video/audio data
    fn demux(&mut self) -> Result<DemuxedStreams> {
        let mut video_samples: Vec<Sample> = Vec::new();
        let mut audio_samples: Vec<Sample> = Vec::new();
        let mut packet = [0u8; TS_PACKET_SIZE];

        loop {
            match self.reader.read_exact(&mut packet) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            if packet[0] != TS_SYNC_BYTE {
                // Try to resync
                continue;
            }

            let pid = ((packet[1] as u16 & 0x1F) << 8) | packet[2] as u16;
            let payload_start = (packet[1] & 0x40) != 0;
            let has_adaptation = (packet[3] & 0x20) != 0;
            let has_payload = (packet[3] & 0x10) != 0;

            if !has_payload {
                continue;
            }

            let mut offset = 4;
            if has_adaptation {
                let adaptation_len = packet[4] as usize;
                offset += 1 + adaptation_len;
            }

            if offset >= TS_PACKET_SIZE {
                continue;
            }

            let payload = &packet[offset..];

            if pid == PAT_PID {
                self.parse_pat(payload, payload_start)?;
            } else if let Some(ref program) = self.program.clone() {
                if pid == program.pmt_pid {
                    self.parse_pmt(payload, payload_start)?;
                } else if Some(pid) == program.video_pid {
                    if payload_start {
                        if self.video_pes_started && !self.video_pes_buffer.is_empty() {
                            if let Some(pes) = self.parse_pes_packet(&self.video_pes_buffer.clone())? {
                                video_samples.push(Sample {
                                    pts: pes.pts,
                                    dts: pes.dts,
                                    data: pes.data,
                                });
                            }
                        }
                        self.video_pes_buffer.clear();
                        self.video_pes_started = true;
                    }
                    if self.video_pes_started {
                        self.video_pes_buffer.extend_from_slice(payload);
                    }
                } else if Some(pid) == program.audio_pid {
                    if payload_start {
                        if self.audio_pes_started && !self.audio_pes_buffer.is_empty() {
                            if let Some(pes) = self.parse_pes_packet(&self.audio_pes_buffer.clone())? {
                                audio_samples.push(Sample {
                                    pts: pes.pts,
                                    dts: pes.dts,
                                    data: pes.data,
                                });
                            }
                        }
                        self.audio_pes_buffer.clear();
                        self.audio_pes_started = true;
                    }
                    if self.audio_pes_started {
                        self.audio_pes_buffer.extend_from_slice(payload);
                    }
                }
            }
        }

        // Flush remaining buffers
        if self.video_pes_started && !self.video_pes_buffer.is_empty() {
            if let Some(pes) = self.parse_pes_packet(&self.video_pes_buffer)? {
                video_samples.push(Sample {
                    pts: pes.pts,
                    dts: pes.dts,
                    data: pes.data,
                });
            }
        }
        if self.audio_pes_started && !self.audio_pes_buffer.is_empty() {
            if let Some(pes) = self.parse_pes_packet(&self.audio_pes_buffer)? {
                audio_samples.push(Sample {
                    pts: pes.pts,
                    dts: pes.dts,
                    data: pes.data,
                });
            }
        }

        Ok(DemuxedStreams {
            video_samples,
            audio_samples,
        })
    }

    fn parse_pat(&mut self, payload: &[u8], payload_start: bool) -> Result<()> {
        let data = if payload_start && !payload.is_empty() {
            let pointer = payload[0] as usize;
            if pointer + 1 >= payload.len() {
                return Ok(());
            }
            &payload[pointer + 1..]
        } else {
            payload
        };

        if data.len() < 8 {
            return Ok(());
        }

        // Skip to program entries (after table header)
        let section_length = ((data[1] as usize & 0x0F) << 8) | data[2] as usize;
        if section_length < 9 || data.len() < 8 {
            return Ok(());
        }

        // Programs start at offset 8, each is 4 bytes
        let mut offset = 8;
        while offset + 4 <= data.len().min(3 + section_length) - 4 {
            let program_num = ((data[offset] as u16) << 8) | data[offset + 1] as u16;
            let pmt_pid = ((data[offset + 2] as u16 & 0x1F) << 8) | data[offset + 3] as u16;

            if program_num != 0 {
                self.program = Some(ProgramInfo {
                    pmt_pid,
                    video_pid: None,
                    audio_pid: None,
                });
                break;
            }
            offset += 4;
        }

        Ok(())
    }

    fn parse_pmt(&mut self, payload: &[u8], payload_start: bool) -> Result<()> {
        let data = if payload_start && !payload.is_empty() {
            let pointer = payload[0] as usize;
            if pointer + 1 >= payload.len() {
                return Ok(());
            }
            &payload[pointer + 1..]
        } else {
            payload
        };

        if data.len() < 12 {
            return Ok(());
        }

        let section_length = ((data[1] as usize & 0x0F) << 8) | data[2] as usize;
        let program_info_length = ((data[10] as usize & 0x0F) << 8) | data[11] as usize;

        let mut offset = 12 + program_info_length;

        while offset + 5 <= data.len().min(3 + section_length) - 4 {
            let stream_type = data[offset];
            let elementary_pid = ((data[offset + 1] as u16 & 0x1F) << 8) | data[offset + 2] as u16;
            let es_info_length = ((data[offset + 3] as usize & 0x0F) << 8) | data[offset + 4] as usize;

            if let Some(ref mut program) = self.program {
                if stream_type == STREAM_TYPE_H264 {
                    program.video_pid = Some(elementary_pid);
                } else if stream_type == STREAM_TYPE_AAC || stream_type == STREAM_TYPE_AAC_LATM {
                    program.audio_pid = Some(elementary_pid);
                }
            }

            offset += 5 + es_info_length;
        }

        Ok(())
    }

    fn parse_pes_packet(&self, data: &[u8]) -> Result<Option<PesPacket>> {
        if data.len() < 9 {
            return Ok(None);
        }

        // Check PES start code
        if data[0] != 0x00 || data[1] != 0x00 || data[2] != 0x01 {
            return Ok(None);
        }

        let header_data_length = data[8] as usize;
        if data.len() < 9 + header_data_length {
            return Ok(None);
        }

        let pts_dts_flags = (data[7] >> 6) & 0x03;
        let mut pts = None;
        let mut dts = None;

        if pts_dts_flags >= 2 && data.len() >= 14 {
            pts = Some(self.parse_timestamp(&data[9..14]));
        }
        if pts_dts_flags == 3 && data.len() >= 19 {
            dts = Some(self.parse_timestamp(&data[14..19]));
        }

        let payload_start = 9 + header_data_length;
        if payload_start >= data.len() {
            return Ok(None);
        }

        Ok(Some(PesPacket {
            pts,
            dts,
            data: data[payload_start..].to_vec(),
        }))
    }

    fn parse_timestamp(&self, data: &[u8]) -> u64 {
        let mut ts: u64 = 0;
        ts |= ((data[0] as u64 >> 1) & 0x07) << 30;
        ts |= (data[1] as u64) << 22;
        ts |= ((data[2] as u64 >> 1) & 0x7F) << 15;
        ts |= (data[3] as u64) << 7;
        ts |= (data[4] as u64 >> 1) & 0x7F;
        ts
    }
}

#[derive(Debug)]
struct DemuxedStreams {
    video_samples: Vec<Sample>,
    audio_samples: Vec<Sample>,
}

#[derive(Debug, Clone)]
struct Sample {
    pts: Option<u64>,
    dts: Option<u64>,
    data: Vec<u8>,
}

// ============================================================================
// H.264 Processing (Annex B to AVCC)
// ============================================================================

/// NAL unit types
const NAL_TYPE_SPS: u8 = 7;
const NAL_TYPE_PPS: u8 = 8;
const NAL_TYPE_IDR: u8 = 5;

#[derive(Debug, Clone)]
struct H264Config {
    sps: Vec<u8>,
    pps: Vec<u8>,
    width: u32,
    height: u32,
    profile: u8,
    level: u8,
}

#[derive(Debug)]
struct H264Sample {
    pts: Option<u64>,
    dts: Option<u64>,
    data: Vec<u8>,
    is_keyframe: bool,
}

/// Find all NAL units in Annex B data
fn find_nal_units(data: &[u8]) -> Vec<(usize, usize)> {
    let mut units = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Look for start code (0x000001 or 0x00000001)
        if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 {
            let start;
            if data[i + 2] == 1 {
                start = i + 3;
            } else if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                start = i + 4;
            } else {
                i += 1;
                continue;
            }

            // Find end of this NAL unit (next start code or end of data)
            let mut end = data.len();
            let mut j = start;
            while j + 3 <= data.len() {
                if data[j] == 0 && data[j + 1] == 0 && (data[j + 2] == 1 || (j + 3 < data.len() && data[j + 2] == 0 && data[j + 3] == 1)) {
                    end = j;
                    break;
                }
                j += 1;
            }

            if start < end {
                units.push((start, end));
            }
            i = end;
        } else {
            i += 1;
        }
    }

    units
}

/// Extract H.264 configuration (SPS/PPS) and convert samples to AVCC format
fn process_h264_samples(samples: &[Sample]) -> Result<(Option<H264Config>, Vec<H264Sample>)> {
    let mut config: Option<H264Config> = None;
    let mut processed_samples = Vec::new();

    for sample in samples {
        let nal_units = find_nal_units(&sample.data);
        let mut avcc_data = Vec::new();
        let mut is_keyframe = false;

        for (start, end) in nal_units {
            let nal_data = &sample.data[start..end];
            if nal_data.is_empty() {
                continue;
            }

            let nal_type = nal_data[0] & 0x1F;

            match nal_type {
                NAL_TYPE_SPS => {
                    if config.is_none() {
                        let (width, height, profile, level) = parse_sps(nal_data);
                        config = Some(H264Config {
                            sps: nal_data.to_vec(),
                            pps: Vec::new(),
                            width,
                            height,
                            profile,
                            level,
                        });
                    }
                    // Also include SPS in stream for decoder compatibility
                    let len = nal_data.len() as u32;
                    avcc_data.extend_from_slice(&len.to_be_bytes());
                    avcc_data.extend_from_slice(nal_data);
                }
                NAL_TYPE_PPS => {
                    if let Some(ref mut cfg) = config {
                        if cfg.pps.is_empty() {
                            cfg.pps = nal_data.to_vec();
                        }
                    }
                    // Also include PPS in stream for decoder compatibility
                    let len = nal_data.len() as u32;
                    avcc_data.extend_from_slice(&len.to_be_bytes());
                    avcc_data.extend_from_slice(nal_data);
                }
                NAL_TYPE_IDR => {
                    is_keyframe = true;
                    // Write length-prefixed NAL unit (AVCC format)
                    let len = nal_data.len() as u32;
                    avcc_data.extend_from_slice(&len.to_be_bytes());
                    avcc_data.extend_from_slice(nal_data);
                }
                1..=5 => {
                    // Non-IDR slice or other VCL NAL
                    if nal_type == 5 {
                        is_keyframe = true;
                    }
                    let len = nal_data.len() as u32;
                    avcc_data.extend_from_slice(&len.to_be_bytes());
                    avcc_data.extend_from_slice(nal_data);
                }
                _ => {
                    // Other NAL types (SEI, AUD, etc.) - include them
                    let len = nal_data.len() as u32;
                    avcc_data.extend_from_slice(&len.to_be_bytes());
                    avcc_data.extend_from_slice(nal_data);
                }
            }
        }

        if !avcc_data.is_empty() {
            processed_samples.push(H264Sample {
                pts: sample.pts,
                dts: sample.dts,
                data: avcc_data,
                is_keyframe,
            });
        }
    }

    Ok((config, processed_samples))
}

/// Bit reader for parsing H.264 NAL units
struct BitReader<'a> {
    data: &'a [u8],
    byte_offset: usize,
    bit_offset: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_offset: 0,
            bit_offset: 0,
        }
    }

    fn read_bit(&mut self) -> Option<u8> {
        if self.byte_offset >= self.data.len() {
            return None;
        }
        let bit = (self.data[self.byte_offset] >> (7 - self.bit_offset)) & 1;
        self.bit_offset += 1;
        if self.bit_offset == 8 {
            self.bit_offset = 0;
            self.byte_offset += 1;
        }
        Some(bit)
    }

    fn read_bits(&mut self, n: u8) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }
        Some(value)
    }

    /// Read unsigned Exp-Golomb coded value
    fn read_ue(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u8;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return None; // Prevent infinite loop
            }
        }
        if leading_zeros == 0 {
            return Some(0);
        }
        let suffix = self.read_bits(leading_zeros)?;
        Some((1 << leading_zeros) - 1 + suffix)
    }

    /// Read signed Exp-Golomb coded value
    fn read_se(&mut self) -> Option<i32> {
        let ue = self.read_ue()?;
        let value = ((ue + 1) / 2) as i32;
        if ue % 2 == 0 {
            Some(-value)
        } else {
            Some(value)
        }
    }
}

/// Parse SPS to extract video dimensions
fn parse_sps(sps: &[u8]) -> (u32, u32, u8, u8) {
    if sps.len() < 4 {
        return (1280, 720, 66, 30); // Default fallback
    }

    let profile_idc = sps[1];
    let level_idc = sps[3];

    // Skip NAL header byte and start parsing from byte 1
    let mut reader = BitReader::new(&sps[1..]);

    // profile_idc (8 bits) - already read above, skip
    reader.read_bits(8);
    // constraint_set flags (8 bits)
    reader.read_bits(8);
    // level_idc (8 bits)
    reader.read_bits(8);
    // seq_parameter_set_id
    reader.read_ue();

    // Handle high profiles
    if profile_idc == 100 || profile_idc == 110 || profile_idc == 122 ||
       profile_idc == 244 || profile_idc == 44 || profile_idc == 83 ||
       profile_idc == 86 || profile_idc == 118 || profile_idc == 128 ||
       profile_idc == 138 || profile_idc == 139 || profile_idc == 134 {
        let chroma_format_idc = reader.read_ue().unwrap_or(1);
        if chroma_format_idc == 3 {
            reader.read_bits(1); // separate_colour_plane_flag
        }
        reader.read_ue(); // bit_depth_luma_minus8
        reader.read_ue(); // bit_depth_chroma_minus8
        reader.read_bits(1); // qpprime_y_zero_transform_bypass_flag
        let seq_scaling_matrix_present = reader.read_bits(1).unwrap_or(0);
        if seq_scaling_matrix_present == 1 {
            let count = if chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..count {
                let seq_scaling_list_present = reader.read_bits(1).unwrap_or(0);
                if seq_scaling_list_present == 1 {
                    let size = if i < 6 { 16 } else { 64 };
                    skip_scaling_list(&mut reader, size);
                }
            }
        }
    }

    // log2_max_frame_num_minus4
    reader.read_ue();
    // pic_order_cnt_type
    let pic_order_cnt_type = reader.read_ue().unwrap_or(0);
    if pic_order_cnt_type == 0 {
        reader.read_ue(); // log2_max_pic_order_cnt_lsb_minus4
    } else if pic_order_cnt_type == 1 {
        reader.read_bits(1); // delta_pic_order_always_zero_flag
        reader.read_se(); // offset_for_non_ref_pic
        reader.read_se(); // offset_for_top_to_bottom_field
        let num_ref_frames_in_pic_order_cnt_cycle = reader.read_ue().unwrap_or(0);
        for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
            reader.read_se(); // offset_for_ref_frame
        }
    }

    // max_num_ref_frames
    reader.read_ue();
    // gaps_in_frame_num_value_allowed_flag
    reader.read_bits(1);

    // pic_width_in_mbs_minus1
    let pic_width_in_mbs_minus1 = reader.read_ue().unwrap_or(79); // 79 = 1280/16-1
    // pic_height_in_map_units_minus1
    let pic_height_in_map_units_minus1 = reader.read_ue().unwrap_or(44); // 44 = 720/16-1
    // frame_mbs_only_flag
    let frame_mbs_only_flag = reader.read_bits(1).unwrap_or(1);

    if frame_mbs_only_flag == 0 {
        reader.read_bits(1); // mb_adaptive_frame_field_flag
    }

    // direct_8x8_inference_flag
    reader.read_bits(1);

    // frame_cropping_flag
    let frame_cropping_flag = reader.read_bits(1).unwrap_or(0);
    let (crop_left, crop_right, crop_top, crop_bottom) = if frame_cropping_flag == 1 {
        (
            reader.read_ue().unwrap_or(0),
            reader.read_ue().unwrap_or(0),
            reader.read_ue().unwrap_or(0),
            reader.read_ue().unwrap_or(0),
        )
    } else {
        (0, 0, 0, 0)
    };

    // Calculate dimensions
    let width = ((pic_width_in_mbs_minus1 + 1) * 16) - (crop_left + crop_right) * 2;
    let height = ((2 - frame_mbs_only_flag) * (pic_height_in_map_units_minus1 + 1) * 16)
        - (crop_top + crop_bottom) * 2;

    (width, height, profile_idc, level_idc)
}

/// Skip scaling list in SPS (used for high profiles)
fn skip_scaling_list(reader: &mut BitReader, size: usize) {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for _ in 0..size {
        if next_scale != 0 {
            let delta_scale = reader.read_se().unwrap_or(0);
            next_scale = (last_scale + delta_scale + 256) % 256;
        }
        last_scale = if next_scale == 0 { last_scale } else { next_scale };
    }
}

// ============================================================================
// AAC Processing (ADTS to raw)
// ============================================================================

#[derive(Debug, Clone)]
struct AacConfig {
    sample_rate: u32,
    channels: u8,
    profile: u8,
}

#[derive(Debug)]
struct AacSample {
    pts: Option<u64>,
    is_pes_start: bool, // True if this frame starts a new PES packet (has real PTS)
    data: Vec<u8>,
}

/// AAC sample rates table
const AAC_SAMPLE_RATES: [u32; 16] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
    16000, 12000, 11025, 8000, 7350, 0, 0, 0,
];

/// Process AAC samples - strip ADTS headers
fn process_aac_samples(samples: &[Sample]) -> Result<(Option<AacConfig>, Vec<AacSample>)> {
    let mut config: Option<AacConfig> = None;
    let mut processed_samples = Vec::new();

    for sample in samples {
        let mut offset = 0;
        let mut frame_index_in_pes = 0u64;
        let base_pts = sample.pts;

        while offset + 7 <= sample.data.len() {
            // Check ADTS sync word
            if sample.data[offset] != 0xFF || (sample.data[offset + 1] & 0xF0) != 0xF0 {
                offset += 1;
                continue;
            }

            let header = &sample.data[offset..];
            if header.len() < 7 {
                break;
            }

            // Parse ADTS header
            let protection_absent = (header[1] & 0x01) != 0;
            let profile = ((header[2] >> 6) & 0x03) + 1;
            let sample_rate_idx = (header[2] >> 2) & 0x0F;
            let channels = ((header[2] & 0x01) << 2) | ((header[3] >> 6) & 0x03);
            let frame_length = (((header[3] & 0x03) as usize) << 11)
                | ((header[4] as usize) << 3)
                | ((header[5] >> 5) as usize);

            let header_size = if protection_absent { 7 } else { 9 };

            if frame_length <= header_size || offset + frame_length > sample.data.len() {
                break;
            }

            // Extract configuration from first frame
            let sample_rate = if (sample_rate_idx as usize) < AAC_SAMPLE_RATES.len() {
                AAC_SAMPLE_RATES[sample_rate_idx as usize]
            } else {
                44100
            };

            if config.is_none() {
                config = Some(AacConfig {
                    sample_rate,
                    channels,
                    profile,
                });
            }

            // Calculate PTS for this frame within the PES packet
            // Each AAC frame is 1024 samples. PTS is in 90kHz units.
            // PTS increment per frame = 1024 * 90000 / sample_rate
            let pts = base_pts.map(|base| {
                let pts_per_frame = (1024u64 * 90000) / sample_rate as u64;
                base + frame_index_in_pes * pts_per_frame
            });

            // Extract raw AAC frame (without ADTS header)
            let raw_data = sample.data[offset + header_size..offset + frame_length].to_vec();
            if !raw_data.is_empty() {
                processed_samples.push(AacSample {
                    pts,
                    is_pes_start: frame_index_in_pes == 0, // First frame in PES has real PTS
                    data: raw_data,
                });
            }

            offset += frame_length;
            frame_index_in_pes += 1;
        }
    }

    Ok((config, processed_samples))
}

// ============================================================================
// MP4 Muxer
// ============================================================================

/// Calculate video duration from actual timestamps
fn calculate_video_duration(samples: &[H264Sample], default_timescale: u32) -> u64 {
    if samples.is_empty() {
        return 0;
    }

    // Get DTS values (fallback to PTS if DTS not available)
    let dts_values: Vec<u64> = samples
        .iter()
        .map(|s| s.dts.or(s.pts).unwrap_or(0))
        .collect();

    // Check if we have valid timestamps
    let has_valid_timestamps = dts_values.iter().any(|&dts| dts > 0);

    if !has_valid_timestamps {
        // Fall back to hardcoded 30fps
        return samples.len() as u64 * (default_timescale as u64 / 30);
    }

    // Find the first non-zero DTS to use as base
    let first_dts = dts_values.iter().find(|&&d| d > 0).copied().unwrap_or(0);
    let last_dts = *dts_values.last().unwrap_or(&0);

    // Duration is last - first, plus one frame duration (estimate from average)
    let total_span = if last_dts > first_dts {
        last_dts - first_dts
    } else {
        0
    };

    // Add estimated duration of last frame (average frame duration)
    let avg_frame_duration = if samples.len() > 1 && total_span > 0 {
        total_span / (samples.len() as u64 - 1)
    } else {
        default_timescale as u64 / 30 // Default to 30fps
    };

    total_span + avg_frame_duration
}

/// Write MP4 file from processed video and audio
fn write_mp4<W: Write + Seek>(
    writer: &mut W,
    video_config: &H264Config,
    video_samples: &[H264Sample],
    audio_config: Option<&AacConfig>,
    audio_samples: &[AacSample],
) -> Result<()> {
    // Calculate sizes
    let mut mdat_size: u64 = 8; // box header

    for sample in video_samples {
        mdat_size += sample.data.len() as u64;
    }
    for sample in audio_samples {
        mdat_size += sample.data.len() as u64;
    }

    // Write ftyp box
    write_ftyp(writer)?;

    // Remember mdat position (for potential future use)
    let _mdat_pos = writer.stream_position()?;

    // Write mdat header (we'll update size later if needed)
    writer.write_all(&(mdat_size as u32).to_be_bytes())?;
    writer.write_all(b"mdat")?;

    // Write video samples to mdat and collect offsets
    let mut video_offsets = Vec::new();
    let mut video_sizes = Vec::new();
    for sample in video_samples {
        video_offsets.push(writer.stream_position()? as u32);
        video_sizes.push(sample.data.len() as u32);
        writer.write_all(&sample.data)?;
    }

    // Write audio samples to mdat and collect offsets
    let mut audio_offsets = Vec::new();
    let mut audio_sizes = Vec::new();
    for sample in audio_samples {
        audio_offsets.push(writer.stream_position()? as u32);
        audio_sizes.push(sample.data.len() as u32);
        writer.write_all(&sample.data)?;
    }

    // Calculate video timing from actual timestamps
    let video_timescale = 90000u32; // Standard for MPEG-TS (PTS/DTS are in 90kHz)
    let video_duration = calculate_video_duration(video_samples, video_timescale);

    // Calculate audio timing
    let audio_timescale = audio_config.map(|c| c.sample_rate).unwrap_or(44100);
    let audio_frame_duration = 1024u32; // AAC frame size
    let audio_duration = audio_samples.len() as u64 * audio_frame_duration as u64;

    // Write moov box
    write_moov(
        writer,
        video_config,
        &video_offsets,
        &video_sizes,
        video_samples,
        video_timescale,
        video_duration,
        audio_config,
        &audio_offsets,
        &audio_sizes,
        audio_samples,
        audio_timescale,
        audio_duration,
    )?;

    Ok(())
}

fn write_ftyp<W: Write>(writer: &mut W) -> Result<()> {
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"isom"); // major brand
    ftyp.extend_from_slice(&0u32.to_be_bytes()); // minor version
    ftyp.extend_from_slice(b"isom"); // compatible brands
    ftyp.extend_from_slice(b"iso2");
    ftyp.extend_from_slice(b"avc1");
    ftyp.extend_from_slice(b"mp41");

    let size = (8 + ftyp.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"ftyp")?;
    writer.write_all(&ftyp)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_moov<W: Write>(
    writer: &mut W,
    video_config: &H264Config,
    video_offsets: &[u32],
    video_sizes: &[u32],
    video_samples: &[H264Sample],
    video_timescale: u32,
    video_duration: u64,
    audio_config: Option<&AacConfig>,
    audio_offsets: &[u32],
    audio_sizes: &[u32],
    audio_samples: &[AacSample],
    audio_timescale: u32,
    audio_duration: u64,
) -> Result<()> {
    let mut moov_data = Vec::new();

    // mvhd (movie header)
    write_mvhd(&mut moov_data, video_timescale, video_duration)?;

    // Video track
    write_video_trak(
        &mut moov_data,
        video_config,
        video_offsets,
        video_sizes,
        video_samples,
        video_timescale,
        video_duration,
    )?;

    // Audio track (if present)
    if let Some(audio_cfg) = audio_config {
        write_audio_trak(
            &mut moov_data,
            audio_cfg,
            audio_offsets,
            audio_sizes,
            audio_samples,
            audio_timescale,
            audio_duration,
            video_timescale, // movie timescale for tkhd duration
        )?;
    }

    let size = (8 + moov_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"moov")?;
    writer.write_all(&moov_data)?;
    Ok(())
}

fn write_mvhd<W: Write>(writer: &mut W, timescale: u32, duration: u64) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&0u32.to_be_bytes()); // creation time
    data.extend_from_slice(&0u32.to_be_bytes()); // modification time
    data.extend_from_slice(&timescale.to_be_bytes()); // timescale
    data.extend_from_slice(&(duration as u32).to_be_bytes()); // duration
    data.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate (1.0)
    data.extend_from_slice(&0x0100u16.to_be_bytes()); // volume (1.0)
    data.extend_from_slice(&[0; 10]); // reserved

    // Matrix (identity)
    data.extend_from_slice(&0x00010000u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0x00010000u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0x40000000u32.to_be_bytes());

    data.extend_from_slice(&[0; 24]); // pre_defined
    data.extend_from_slice(&3u32.to_be_bytes()); // next_track_id

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"mvhd")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_video_trak<W: Write>(
    writer: &mut W,
    config: &H264Config,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[H264Sample],
    timescale: u32,
    duration: u64,
) -> Result<()> {
    let mut trak_data = Vec::new();

    // tkhd
    write_tkhd(&mut trak_data, 1, config.width, config.height, duration)?;

    // edts (edit list) - needed for proper A/V sync when first PTS != first DTS
    // The media_time should be the composition offset of the first frame
    if let Some(first_sample) = samples.first() {
        let first_pts = first_sample.pts.unwrap_or(0);
        let first_dts = first_sample.dts.or(first_sample.pts).unwrap_or(0);
        let media_time = if first_pts >= first_dts {
            (first_pts - first_dts) as i32
        } else {
            0
        };
        // Only write edts if there's a non-zero offset
        if media_time > 0 {
            write_edts(&mut trak_data, duration, media_time)?;
        }
    }

    // mdia
    write_video_mdia(&mut trak_data, config, offsets, sizes, samples, timescale, duration)?;

    let size = (8 + trak_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"trak")?;
    writer.write_all(&trak_data)?;
    Ok(())
}

/// Write edit list box (edts/elst) for proper playback timing
fn write_edts<W: Write>(writer: &mut W, duration: u64, media_time: i32) -> Result<()> {
    let mut edts_data = Vec::new();

    // elst (edit list)
    let mut elst_data = Vec::new();
    elst_data.push(0); // version
    elst_data.extend_from_slice(&[0, 0, 0]); // flags
    elst_data.extend_from_slice(&1u32.to_be_bytes()); // entry count

    // Single entry: play from media_time for the full duration
    elst_data.extend_from_slice(&(duration as u32).to_be_bytes()); // segment duration (in movie timescale)
    elst_data.extend_from_slice(&media_time.to_be_bytes()); // media time (where to start in track)
    elst_data.extend_from_slice(&0x00010000u32.to_be_bytes()); // media rate (1.0 in 16.16 fixed point)

    let elst_size = (8 + elst_data.len()) as u32;
    edts_data.extend_from_slice(&elst_size.to_be_bytes());
    edts_data.extend_from_slice(b"elst");
    edts_data.extend_from_slice(&elst_data);

    let size = (8 + edts_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"edts")?;
    writer.write_all(&edts_data)?;
    Ok(())
}

fn write_tkhd<W: Write>(writer: &mut W, track_id: u32, width: u32, height: u32, duration: u64) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 3]); // flags (track enabled, in movie)
    data.extend_from_slice(&0u32.to_be_bytes()); // creation time
    data.extend_from_slice(&0u32.to_be_bytes()); // modification time
    data.extend_from_slice(&track_id.to_be_bytes()); // track id
    data.extend_from_slice(&0u32.to_be_bytes()); // reserved
    data.extend_from_slice(&(duration as u32).to_be_bytes()); // duration
    data.extend_from_slice(&[0; 8]); // reserved
    data.extend_from_slice(&0u16.to_be_bytes()); // layer
    data.extend_from_slice(&0u16.to_be_bytes()); // alternate group
    data.extend_from_slice(&0u16.to_be_bytes()); // volume (0 for video)
    data.extend_from_slice(&0u16.to_be_bytes()); // reserved

    // Matrix (identity)
    data.extend_from_slice(&0x00010000u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0x00010000u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0x40000000u32.to_be_bytes());

    // Width and height in 16.16 fixed point
    data.extend_from_slice(&((width << 16) as u32).to_be_bytes());
    data.extend_from_slice(&((height << 16) as u32).to_be_bytes());

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"tkhd")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_video_mdia<W: Write>(
    writer: &mut W,
    config: &H264Config,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[H264Sample],
    timescale: u32,
    duration: u64,
) -> Result<()> {
    let mut mdia_data = Vec::new();

    // mdhd
    write_mdhd(&mut mdia_data, timescale, duration)?;

    // hdlr
    write_hdlr(&mut mdia_data, b"vide", b"VideoHandler")?;

    // minf
    write_video_minf(&mut mdia_data, config, offsets, sizes, samples, timescale)?;

    let size = (8 + mdia_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"mdia")?;
    writer.write_all(&mdia_data)?;
    Ok(())
}

fn write_mdhd<W: Write>(writer: &mut W, timescale: u32, duration: u64) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&0u32.to_be_bytes()); // creation time
    data.extend_from_slice(&0u32.to_be_bytes()); // modification time
    data.extend_from_slice(&timescale.to_be_bytes()); // timescale
    data.extend_from_slice(&(duration as u32).to_be_bytes()); // duration
    data.extend_from_slice(&0x55C4u16.to_be_bytes()); // language (und)
    data.extend_from_slice(&0u16.to_be_bytes()); // pre_defined

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"mdhd")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_hdlr<W: Write>(writer: &mut W, handler_type: &[u8; 4], name: &[u8]) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
    data.extend_from_slice(handler_type);
    data.extend_from_slice(&[0; 12]); // reserved
    data.extend_from_slice(name);
    data.push(0); // null terminator

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"hdlr")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_video_minf<W: Write>(
    writer: &mut W,
    config: &H264Config,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[H264Sample],
    timescale: u32,
) -> Result<()> {
    let mut minf_data = Vec::new();

    // vmhd
    write_vmhd(&mut minf_data)?;

    // dinf
    write_dinf(&mut minf_data)?;

    // stbl
    write_video_stbl(&mut minf_data, config, offsets, sizes, samples, timescale)?;

    let size = (8 + minf_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"minf")?;
    writer.write_all(&minf_data)?;
    Ok(())
}

fn write_vmhd<W: Write>(writer: &mut W) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 1]); // flags
    data.extend_from_slice(&0u16.to_be_bytes()); // graphics mode
    data.extend_from_slice(&[0; 6]); // opcolor

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"vmhd")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_dinf<W: Write>(writer: &mut W) -> Result<()> {
    let mut dinf_data = Vec::new();

    // dref
    let mut dref_data = Vec::new();
    dref_data.push(0); // version
    dref_data.extend_from_slice(&[0, 0, 0]); // flags
    dref_data.extend_from_slice(&1u32.to_be_bytes()); // entry count

    // url box (self-contained)
    dref_data.extend_from_slice(&12u32.to_be_bytes());
    dref_data.extend_from_slice(b"url ");
    dref_data.push(0); // version
    dref_data.extend_from_slice(&[0, 0, 1]); // flags (self-contained)

    let dref_size = (8 + dref_data.len()) as u32;
    dinf_data.extend_from_slice(&dref_size.to_be_bytes());
    dinf_data.extend_from_slice(b"dref");
    dinf_data.extend_from_slice(&dref_data);

    let size = (8 + dinf_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"dinf")?;
    writer.write_all(&dinf_data)?;
    Ok(())
}

fn write_video_stbl<W: Write>(
    writer: &mut W,
    config: &H264Config,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[H264Sample],
    timescale: u32,
) -> Result<()> {
    let mut stbl_data = Vec::new();

    // stsd
    write_video_stsd(&mut stbl_data, config)?;

    // stts - use actual DTS values if available, otherwise assume constant frame rate
    write_video_stts(&mut stbl_data, samples, timescale)?;

    // ctts - composition time offsets (PTS - DTS), needed for B-frames
    write_ctts(&mut stbl_data, samples)?;

    // stss (sync samples / keyframes)
    write_stss(&mut stbl_data, samples)?;

    // stsc
    write_stsc(&mut stbl_data)?;

    // stsz
    write_stsz(&mut stbl_data, sizes)?;

    // stco
    write_stco(&mut stbl_data, offsets)?;

    let size = (8 + stbl_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stbl")?;
    writer.write_all(&stbl_data)?;
    Ok(())
}

fn write_video_stsd<W: Write>(writer: &mut W, config: &H264Config) -> Result<()> {
    let mut stsd_data = Vec::new();

    stsd_data.push(0); // version
    stsd_data.extend_from_slice(&[0, 0, 0]); // flags
    stsd_data.extend_from_slice(&1u32.to_be_bytes()); // entry count

    // avc1 box
    let mut avc1_data = Vec::new();
    avc1_data.extend_from_slice(&[0; 6]); // reserved
    avc1_data.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    avc1_data.extend_from_slice(&[0; 16]); // pre_defined, reserved
    avc1_data.extend_from_slice(&(config.width as u16).to_be_bytes());
    avc1_data.extend_from_slice(&(config.height as u16).to_be_bytes());
    avc1_data.extend_from_slice(&0x00480000u32.to_be_bytes()); // horiz resolution (72 dpi)
    avc1_data.extend_from_slice(&0x00480000u32.to_be_bytes()); // vert resolution (72 dpi)
    avc1_data.extend_from_slice(&0u32.to_be_bytes()); // reserved
    avc1_data.extend_from_slice(&1u16.to_be_bytes()); // frame count
    avc1_data.extend_from_slice(&[0; 32]); // compressor name
    avc1_data.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
    avc1_data.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined

    // avcC box
    let mut avcc_data = Vec::new();
    avcc_data.push(1); // configuration version
    avcc_data.push(config.profile);
    avcc_data.push(0); // profile compatibility
    avcc_data.push(config.level);
    avcc_data.push(0xFF); // length size minus one (3 = 4 bytes) | reserved
    avcc_data.push(0xE1); // num SPS | reserved
    avcc_data.extend_from_slice(&(config.sps.len() as u16).to_be_bytes());
    avcc_data.extend_from_slice(&config.sps);
    avcc_data.push(1); // num PPS
    avcc_data.extend_from_slice(&(config.pps.len() as u16).to_be_bytes());
    avcc_data.extend_from_slice(&config.pps);

    let avcc_size = (8 + avcc_data.len()) as u32;
    avc1_data.extend_from_slice(&avcc_size.to_be_bytes());
    avc1_data.extend_from_slice(b"avcC");
    avc1_data.extend_from_slice(&avcc_data);

    let avc1_size = (8 + avc1_data.len()) as u32;
    stsd_data.extend_from_slice(&avc1_size.to_be_bytes());
    stsd_data.extend_from_slice(b"avc1");
    stsd_data.extend_from_slice(&avc1_data);

    let size = (8 + stsd_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stsd")?;
    writer.write_all(&stsd_data)?;
    Ok(())
}

fn write_stts<W: Write>(writer: &mut W, sample_count: usize, sample_delta: u32) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&1u32.to_be_bytes()); // entry count
    data.extend_from_slice(&(sample_count as u32).to_be_bytes());
    data.extend_from_slice(&sample_delta.to_be_bytes());

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stts")?;
    writer.write_all(&data)?;
    Ok(())
}

/// Write stts box using actual DTS values from samples
fn write_video_stts<W: Write>(writer: &mut W, samples: &[H264Sample], timescale: u32) -> Result<()> {
    if samples.is_empty() {
        return write_stts(writer, 0, 0);
    }

    // Collect DTS values, using PTS as fallback if DTS is not available
    let dts_values: Vec<u64> = samples
        .iter()
        .map(|s| s.dts.or(s.pts).unwrap_or(0))
        .collect();

    // Check if we have valid timestamps (not all zeros)
    let has_valid_timestamps = dts_values.iter().any(|&dts| dts > 0);

    if !has_valid_timestamps {
        // Fall back to constant frame rate (assume 30fps)
        return write_stts(writer, samples.len(), timescale / 30);
    }

    // Calculate deltas between consecutive DTS values
    let mut deltas: Vec<u32> = Vec::new();
    for i in 0..samples.len() {
        let delta = if i + 1 < samples.len() {
            // Delta to next sample
            let current_dts = dts_values[i];
            let next_dts = dts_values[i + 1];
            if next_dts > current_dts {
                (next_dts - current_dts) as u32
            } else {
                // Handle wraparound or out-of-order - use default
                timescale / 30
            }
        } else {
            // Last sample - use previous delta or default
            if !deltas.is_empty() {
                deltas[deltas.len() - 1]
            } else {
                timescale / 30
            }
        };
        deltas.push(delta);
    }

    // Run-length encode the deltas
    let mut entries: Vec<(u32, u32)> = Vec::new(); // (count, delta)
    for delta in deltas {
        if let Some(last) = entries.last_mut() {
            if last.1 == delta {
                last.0 += 1;
                continue;
            }
        }
        entries.push((1, delta));
    }

    let mut data = Vec::new();
    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (count, delta) in entries {
        data.extend_from_slice(&count.to_be_bytes());
        data.extend_from_slice(&delta.to_be_bytes());
    }

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stts")?;
    writer.write_all(&data)?;
    Ok(())
}

/// Write ctts box (composition time to sample) for B-frame reordering
/// This specifies the offset between decode time (DTS) and presentation time (PTS)
fn write_ctts<W: Write>(writer: &mut W, samples: &[H264Sample]) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }

    // Calculate composition time offsets (PTS - DTS)
    // In standard video, PTS >= DTS, so offsets should be non-negative
    let offsets: Vec<u32> = samples
        .iter()
        .map(|s| {
            let pts = s.pts.unwrap_or(0);
            let dts = s.dts.or(s.pts).unwrap_or(0);
            if pts >= dts {
                (pts - dts) as u32
            } else {
                0 // Clamp negative to 0 for version 0 compatibility
            }
        })
        .collect();

    // Check if all offsets are zero - if so, skip ctts box
    if offsets.iter().all(|&o| o == 0) {
        return Ok(());
    }

    // Run-length encode the offsets
    let mut entries: Vec<(u32, u32)> = Vec::new(); // (count, offset)
    for offset in offsets {
        if let Some(last) = entries.last_mut() {
            if last.1 == offset {
                last.0 += 1;
                continue;
            }
        }
        entries.push((1, offset));
    }

    let mut data = Vec::new();
    data.push(0); // version 0 (unsigned offsets, better compatibility)
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (count, offset) in entries {
        data.extend_from_slice(&count.to_be_bytes());
        data.extend_from_slice(&offset.to_be_bytes());
    }

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"ctts")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_stss<W: Write>(writer: &mut W, samples: &[H264Sample]) -> Result<()> {
    let keyframes: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_keyframe)
        .map(|(i, _)| (i + 1) as u32) // 1-indexed
        .collect();

    if keyframes.is_empty() {
        return Ok(());
    }

    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&(keyframes.len() as u32).to_be_bytes());
    for kf in keyframes {
        data.extend_from_slice(&kf.to_be_bytes());
    }

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stss")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_stsc<W: Write>(writer: &mut W) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&1u32.to_be_bytes()); // entry count

    // One sample per chunk
    data.extend_from_slice(&1u32.to_be_bytes()); // first chunk
    data.extend_from_slice(&1u32.to_be_bytes()); // samples per chunk
    data.extend_from_slice(&1u32.to_be_bytes()); // sample description index

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stsc")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_stsz<W: Write>(writer: &mut W, sizes: &[u32]) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&0u32.to_be_bytes()); // sample size (0 = variable)
    data.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
    for &size in sizes {
        data.extend_from_slice(&size.to_be_bytes());
    }

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stsz")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_stco<W: Write>(writer: &mut W, offsets: &[u32]) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&(offsets.len() as u32).to_be_bytes());
    for &offset in offsets {
        data.extend_from_slice(&offset.to_be_bytes());
    }

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stco")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_audio_trak<W: Write>(
    writer: &mut W,
    config: &AacConfig,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[AacSample],
    timescale: u32,
    duration: u64,
    movie_timescale: u32,
) -> Result<()> {
    let mut trak_data = Vec::new();

    // tkhd duration must be in movie timescale (not audio timescale)
    // Convert from audio samples to movie timescale
    let tkhd_duration = (duration * movie_timescale as u64) / timescale as u64;

    // tkhd (audio track)
    write_audio_tkhd(&mut trak_data, 2, tkhd_duration)?;

    // mdia
    write_audio_mdia(&mut trak_data, config, offsets, sizes, samples, timescale, duration)?;

    let size = (8 + trak_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"trak")?;
    writer.write_all(&trak_data)?;
    Ok(())
}

fn write_audio_tkhd<W: Write>(writer: &mut W, track_id: u32, duration: u64) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 3]); // flags
    data.extend_from_slice(&0u32.to_be_bytes()); // creation time
    data.extend_from_slice(&0u32.to_be_bytes()); // modification time
    data.extend_from_slice(&track_id.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes()); // reserved
    data.extend_from_slice(&(duration as u32).to_be_bytes());
    data.extend_from_slice(&[0; 8]); // reserved
    data.extend_from_slice(&0u16.to_be_bytes()); // layer
    data.extend_from_slice(&1u16.to_be_bytes()); // alternate group
    data.extend_from_slice(&0x0100u16.to_be_bytes()); // volume (1.0)
    data.extend_from_slice(&0u16.to_be_bytes()); // reserved

    // Matrix
    data.extend_from_slice(&0x00010000u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0x00010000u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(&0x40000000u32.to_be_bytes());

    data.extend_from_slice(&0u32.to_be_bytes()); // width
    data.extend_from_slice(&0u32.to_be_bytes()); // height

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"tkhd")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_audio_mdia<W: Write>(
    writer: &mut W,
    config: &AacConfig,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[AacSample],
    timescale: u32,
    duration: u64,
) -> Result<()> {
    let mut mdia_data = Vec::new();

    // mdhd
    write_mdhd(&mut mdia_data, timescale, duration)?;

    // hdlr
    write_hdlr(&mut mdia_data, b"soun", b"SoundHandler")?;

    // minf
    write_audio_minf(&mut mdia_data, config, offsets, sizes, samples)?;

    let size = (8 + mdia_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"mdia")?;
    writer.write_all(&mdia_data)?;
    Ok(())
}

fn write_audio_minf<W: Write>(
    writer: &mut W,
    config: &AacConfig,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[AacSample],
) -> Result<()> {
    let mut minf_data = Vec::new();

    // smhd
    write_smhd(&mut minf_data)?;

    // dinf
    write_dinf(&mut minf_data)?;

    // stbl
    write_audio_stbl(&mut minf_data, config, offsets, sizes, samples)?;

    let size = (8 + minf_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"minf")?;
    writer.write_all(&minf_data)?;
    Ok(())
}

fn write_smhd<W: Write>(writer: &mut W) -> Result<()> {
    let mut data = Vec::new();

    data.push(0); // version
    data.extend_from_slice(&[0, 0, 0]); // flags
    data.extend_from_slice(&0u16.to_be_bytes()); // balance
    data.extend_from_slice(&0u16.to_be_bytes()); // reserved

    let size = (8 + data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"smhd")?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_audio_stbl<W: Write>(
    writer: &mut W,
    config: &AacConfig,
    offsets: &[u32],
    sizes: &[u32],
    samples: &[AacSample],
) -> Result<()> {
    let mut stbl_data = Vec::new();

    // stsd
    write_audio_stsd(&mut stbl_data, config)?;

    // stts - use actual PTS values if available
    write_audio_stts(&mut stbl_data, samples, config.sample_rate)?;

    // stsc
    write_stsc(&mut stbl_data)?;

    // stsz
    write_stsz(&mut stbl_data, sizes)?;

    // stco
    write_stco(&mut stbl_data, offsets)?;

    let size = (8 + stbl_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stbl")?;
    writer.write_all(&stbl_data)?;
    Ok(())
}

/// Write audio stts box with constant frame duration
///
/// AAC frames are always 1024 samples. While there can be slight timing drift
/// between audio sample rate and MPEG-TS timestamps, using a constant duration
/// provides the most compatible output for players.
fn write_audio_stts<W: Write>(writer: &mut W, samples: &[AacSample], _sample_rate: u32) -> Result<()> {
    // Use constant 1024 samples per AAC frame
    write_stts(writer, samples.len(), 1024)
}

fn write_audio_stsd<W: Write>(writer: &mut W, config: &AacConfig) -> Result<()> {
    let mut stsd_data = Vec::new();

    stsd_data.push(0); // version
    stsd_data.extend_from_slice(&[0, 0, 0]); // flags
    stsd_data.extend_from_slice(&1u32.to_be_bytes()); // entry count

    // mp4a box
    let mut mp4a_data = Vec::new();
    mp4a_data.extend_from_slice(&[0; 6]); // reserved
    mp4a_data.extend_from_slice(&1u16.to_be_bytes()); // data reference index
    mp4a_data.extend_from_slice(&[0; 8]); // reserved
    mp4a_data.extend_from_slice(&(config.channels as u16).to_be_bytes());
    mp4a_data.extend_from_slice(&16u16.to_be_bytes()); // sample size (bits)
    mp4a_data.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    mp4a_data.extend_from_slice(&0u16.to_be_bytes()); // reserved
    mp4a_data.extend_from_slice(&((config.sample_rate as u32) << 16).to_be_bytes());

    // esds box (elementary stream descriptor)
    let esds = build_esds(config);
    mp4a_data.extend_from_slice(&esds);

    let mp4a_size = (8 + mp4a_data.len()) as u32;
    stsd_data.extend_from_slice(&mp4a_size.to_be_bytes());
    stsd_data.extend_from_slice(b"mp4a");
    stsd_data.extend_from_slice(&mp4a_data);

    let size = (8 + stsd_data.len()) as u32;
    writer.write_all(&size.to_be_bytes())?;
    writer.write_all(b"stsd")?;
    writer.write_all(&stsd_data)?;
    Ok(())
}

fn build_esds(config: &AacConfig) -> Vec<u8> {
    // Build AudioSpecificConfig
    let sample_rate_idx = AAC_SAMPLE_RATES
        .iter()
        .position(|&r| r == config.sample_rate)
        .unwrap_or(4) as u8; // Default to 44100

    let audio_specific_config: [u8; 2] = [
        (config.profile << 3) | (sample_rate_idx >> 1),
        ((sample_rate_idx & 1) << 7) | ((config.channels & 0x0F) << 3),
    ];

    let mut esds = Vec::new();

    esds.push(0); // version
    esds.extend_from_slice(&[0, 0, 0]); // flags

    // ES_Descriptor
    esds.push(0x03); // ES_DescrTag
    esds.push(23 + audio_specific_config.len() as u8); // size
    esds.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    esds.push(0); // flags

    // DecoderConfigDescriptor
    esds.push(0x04); // DecoderConfigDescrTag
    esds.push(15 + audio_specific_config.len() as u8); // size
    esds.push(0x40); // objectTypeIndication (AAC)
    esds.push(0x15); // streamType (5=audio) << 2 | upstream | reserved
    esds.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
    esds.extend_from_slice(&128000u32.to_be_bytes()); // maxBitrate
    esds.extend_from_slice(&128000u32.to_be_bytes()); // avgBitrate

    // DecoderSpecificInfo
    esds.push(0x05); // DecSpecificInfoTag
    esds.push(audio_specific_config.len() as u8);
    esds.extend_from_slice(&audio_specific_config);

    // SLConfigDescriptor
    esds.push(0x06); // SLConfigDescrTag
    esds.push(1);
    esds.push(0x02);

    let size = (8 + esds.len()) as u32;
    let mut result = Vec::new();
    result.extend_from_slice(&size.to_be_bytes());
    result.extend_from_slice(b"esds");
    result.extend_from_slice(&esds);
    result
}

// ============================================================================
// Public API
// ============================================================================

/// Remux an MPEG-TS file to MP4 without transcoding.
///
/// This function reads video (H.264) and audio (AAC) from the input TS file
/// and writes them to an MP4 container, similar to `ffmpeg -c copy`.
///
/// # Arguments
/// * `input` - Reader providing MPEG-TS data
/// * `output` - Writer for the MP4 output (must support seeking)
///
/// # Example
/// ```no_run
/// use std::fs::File;
/// use std::io::BufReader;
///
/// let input = BufReader::new(File::open("input.ts").unwrap());
/// let mut output = File::create("output.mp4").unwrap();
/// ts_to_mp4::remux(input, &mut output).unwrap();
/// ```
pub fn remux<R: Read, W: Write + Seek>(input: R, output: &mut W) -> Result<()> {
    // Demux TS
    let mut demuxer = TsDemuxer::new(input);
    let streams = demuxer.demux()?;

    if streams.video_samples.is_empty() {
        return Err(Error::NoVideoStream);
    }

    // Process H.264
    let (video_config, video_samples) = process_h264_samples(&streams.video_samples)?;
    let video_config = video_config.ok_or_else(|| Error::InvalidH264("No SPS/PPS found".into()))?;

    // Process AAC
    let (audio_config, audio_samples) = process_aac_samples(&streams.audio_samples)?;

    // Write MP4
    write_mp4(
        output,
        &video_config,
        &video_samples,
        audio_config.as_ref(),
        &audio_samples,
    )?;

    Ok(())
}

/// Remux an MPEG-TS file to MP4 using file paths.
///
/// Convenience function that handles file opening.
pub fn remux_file(input_path: &str, output_path: &str) -> Result<()> {
    let input = std::io::BufReader::new(std::fs::File::open(input_path)?);
    let mut output = std::fs::File::create(output_path)?;
    remux(input, &mut output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nal_units() {
        // Test with 4-byte start code
        let data = [0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68, 0x43];
        let units = find_nal_units(&data);
        assert_eq!(units.len(), 2);
        assert_eq!(&data[units[0].0..units[0].1], &[0x67, 0x42]);
        assert_eq!(&data[units[1].0..units[1].1], &[0x68, 0x43]);
    }

    #[test]
    fn test_find_nal_units_3byte() {
        // Test with 3-byte start code
        let data = [0, 0, 1, 0x67, 0x42, 0, 0, 1, 0x68];
        let units = find_nal_units(&data);
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn test_aac_sample_rates() {
        assert_eq!(AAC_SAMPLE_RATES[3], 48000);
        assert_eq!(AAC_SAMPLE_RATES[4], 44100);
    }
}
