#!/bin/bash
# Build WASM module for the TS to MP4 converter

set -e

echo "Building WASM module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build the WASM module
wasm-pack build --target web --out-dir web/pkg --features wasm

# Remove unnecessary files
rm -f web/pkg/.gitignore
rm -f web/pkg/package.json
rm -f web/pkg/README.md

echo ""
echo "Build complete! Files are in web/pkg/"
echo ""
echo "To test locally:"
echo "  cd web && python3 -m http.server 8000"
echo "  Then open http://localhost:8000"
