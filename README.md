# ts-to-mp4

![Tests](https://github.com/rajbot/ts-to-mp4/actions/workflows/ci.yml/badge.svg)

A pure Rust library and CLI tool for remuxing MPEG-TS to MP4 without transcoding, similar to `ffmpeg -c copy`.

## Features

- **No transcoding** - Copies H.264 video and AAC audio streams directly
- **Pure Rust** - No external dependencies like ffmpeg
- **Simple API** - Just `remux(input, output)`
- **CLI included** - Convert files from the command line

## What it does

1. **Demuxes MPEG-TS** - Parses transport stream packets, PAT/PMT tables, and PES packets
2. **Processes H.264** - Converts from Annex B format (start codes) to AVCC format (length-prefixed NAL units)
3. **Processes AAC** - Strips ADTS headers from audio frames
4. **Muxes to MP4** - Writes a valid MP4 container with proper metadata

## Browser Demo

Try the converter in your browser: **[rajbot.github.io/ts-to-mp4](https://rajbot.github.io/ts-to-mp4)**

All processing happens locally in your browser using WebAssembly - no files are uploaded.

## Command Line Usage

### Install

```bash
cargo install --git https://github.com/rajbot/ts-to-mp4 --features cli
```

### Run

```bash
# Convert input.ts to input.mp4
ts-to-mp4 input.ts

# Specify output filename
ts-to-mp4 input.ts -o output.mp4

# Show help
ts-to-mp4 --help
```

## Library Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ts-to-mp4 = { git = "https://github.com/rajbot/ts-to-mp4" }
```

### From file paths

```rust
ts_to_mp4::remux_file("input.ts", "output.mp4")?;
```

### From readers/writers

```rust
use std::fs::File;
use std::io::BufReader;

let input = BufReader::new(File::open("input.ts")?);
let mut output = File::create("output.mp4")?;
ts_to_mp4::remux(input, &mut output)?;
```

## Supported formats

- **Video**: H.264/AVC
- **Audio**: AAC (ADTS)
- **Container**: MPEG-TS input, MP4/M4V output

## Building WebAssembly

To build and run the web version locally:

```bash
# Build WASM module
./build-wasm.sh

# Start local server
cd web && python3 -m http.server 8000

# Open http://localhost:8000
```

## Limitations

- Video-only files must contain H.264
- Audio-only files are not supported (video track required)
- Does not support other codecs (H.265/HEVC, VP9, etc.)

## License

GNU Affero General Public License v3.0 (AGPL-3.0)
