use std::env;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: remux <input.ts> <output.mp4>");
        std::process::exit(1);
    }

    let input = BufReader::new(File::open(&args[1]).expect("Failed to open input"));
    let mut output = File::create(&args[2]).expect("Failed to create output");

    match ts_to_mp4::remux(input, &mut output) {
        Ok(()) => println!("Successfully remuxed {} -> {}", args[1], args[2]),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
