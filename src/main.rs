//! Command-line tool for remuxing MPEG-TS to MP4

#[cfg(feature = "cli")]
use clap::Parser;

#[cfg(feature = "cli")]
#[derive(Parser)]
#[command(name = "ts-to-mp4")]
#[command(about = "Remux MPEG-TS to MP4 without transcoding", long_about = None)]
struct Args {
    /// Input MPEG-TS file
    input: String,

    /// Output MP4 file (defaults to input with .mp4 extension)
    #[arg(short, long)]
    output: Option<String>,
}

#[cfg(feature = "cli")]
fn main() {
    let args = Args::parse();

    let output = args.output.unwrap_or_else(|| {
        if args.input.ends_with(".ts") {
            args.input.replace(".ts", ".mp4")
        } else {
            format!("{}.mp4", args.input)
        }
    });

    eprintln!("Converting {} -> {}", args.input, output);

    match ts_to_mp4::remux_file(&args.input, &output) {
        Ok(()) => {
            eprintln!("Done!");
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("CLI not enabled. Build with: cargo build --features cli");
    eprintln!("Or use the library API directly.");
    std::process::exit(1);
}
