// TS to MP4 Converter - Web Application
// Uses WebAssembly for browser-based conversion

import init, { convert_ts_to_mp4 } from './pkg/ts_to_mp4.js';

let wasmReady = false;
let convertedBlob = null;
let outputFileName = 'output.mp4';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const progressSection = document.getElementById('progress-section');
const resultSection = document.getElementById('result-section');
const errorSection = document.getElementById('error-section');
const fileNameEl = document.getElementById('file-name');
const fileSizeEl = document.getElementById('file-size');
const progressFill = document.getElementById('progress-fill');
const statusText = document.getElementById('status-text');
const downloadButton = document.getElementById('download-button');
const convertAnother = document.getElementById('convert-another');
const tryAgain = document.getElementById('try-again');
const errorMessage = document.getElementById('error-message');

// Initialize WASM module
async function initWasm() {
    try {
        await init();
        wasmReady = true;
        console.log('WASM module loaded successfully');
    } catch (error) {
        console.error('Failed to load WASM module:', error);
        showError('Failed to load converter. Please refresh the page.');
    }
}

// Format file size for display
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Show/hide sections
function showSection(section) {
    progressSection.classList.add('hidden');
    resultSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    dropZone.classList.remove('hidden');

    if (section === 'progress') {
        dropZone.classList.add('hidden');
        progressSection.classList.remove('hidden');
    } else if (section === 'result') {
        dropZone.classList.add('hidden');
        resultSection.classList.remove('hidden');
    } else if (section === 'error') {
        dropZone.classList.add('hidden');
        errorSection.classList.remove('hidden');
    }
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    showSection('error');
}

// Reset to initial state
function reset() {
    convertedBlob = null;
    outputFileName = 'output.mp4';
    progressFill.style.width = '0%';
    statusText.textContent = 'Loading...';
    showSection('drop');
}

// Update progress display
function updateProgress(percent, status) {
    progressFill.style.width = percent + '%';
    statusText.textContent = status;
}

// Convert file
async function convertFile(file) {
    if (!wasmReady) {
        showError('Converter is still loading. Please wait...');
        return;
    }

    if (!file.name.toLowerCase().endsWith('.ts')) {
        showError('Please select a .ts (MPEG-TS) file.');
        return;
    }

    // Set output filename
    outputFileName = file.name.replace(/\.ts$/i, '.mp4');

    // Show progress
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = formatFileSize(file.size);
    showSection('progress');
    updateProgress(10, 'Reading file...');

    try {
        // Read file as ArrayBuffer
        const arrayBuffer = await file.arrayBuffer();
        const inputData = new Uint8Array(arrayBuffer);

        updateProgress(30, 'Converting to MP4...');

        // Convert using WASM
        // Use setTimeout to allow UI to update before blocking conversion
        await new Promise(resolve => setTimeout(resolve, 50));

        const startTime = performance.now();
        const outputData = convert_ts_to_mp4(inputData);
        const endTime = performance.now();

        console.log(`Conversion completed in ${(endTime - startTime).toFixed(0)}ms`);
        console.log(`Input: ${formatFileSize(inputData.length)}, Output: ${formatFileSize(outputData.length)}`);

        updateProgress(90, 'Preparing download...');

        // Create blob for download
        convertedBlob = new Blob([outputData], { type: 'video/mp4' });

        updateProgress(100, 'Complete!');

        // Show result after brief delay
        setTimeout(() => {
            showSection('result');
        }, 500);

    } catch (error) {
        console.error('Conversion error:', error);
        showError('Conversion failed: ' + error.message);
    }
}

// Download converted file
function downloadFile() {
    if (!convertedBlob) return;

    const url = URL.createObjectURL(convertedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = outputFileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Event Listeners

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        convertFile(file);
    }
});

// Drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const file = e.dataTransfer.files[0];
    if (file) {
        convertFile(file);
    }
});

// Click on drop zone to trigger file input
dropZone.addEventListener('click', (e) => {
    if (e.target === dropZone || e.target.closest('.drop-zone-content')) {
        if (!e.target.closest('.file-button')) {
            fileInput.click();
        }
    }
});

// Download button
downloadButton.addEventListener('click', downloadFile);

// Convert another / Try again buttons
convertAnother.addEventListener('click', reset);
tryAgain.addEventListener('click', reset);

// Initialize on page load
initWasm();
