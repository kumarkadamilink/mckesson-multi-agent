"""
debug_chunk.py
Run this in your project root to diagnose the chunk_text MemoryError.
    python debug_chunk.py
"""
import re
from pathlib import Path
from pypdf import PdfReader

CONTRACTS_DIR = Path("data/knowledge/contracts")

def extract_pdf(path):
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)

# Test just the first PDF
files = sorted([f for f in CONTRACTS_DIR.iterdir() if f.suffix.lower() == ".pdf"])
if not files:
    print("No PDFs found in", CONTRACTS_DIR)
else:
    path = files[0]
    print(f"Testing: {path.name}")
    text = extract_pdf(path)
    print(f"Text length : {len(text)} characters")
    print(f"First 300 chars:\n{text[:300]}")
    print(f"\nLast 300 chars:\n{text[-300:]}")

    # Check for anything weird
    print(f"\nNull bytes  : {text.count(chr(0))}")
    print(f"Lines       : {text.count(chr(10))}")
    print(f"Unique chars: {len(set(text))}")

    # Try a tiny manual chunk
    print("\nAttempting manual chunk loop...")
    chunks = []
    CHUNK_SIZE = 400
    OVERLAP = 50
    start = 0
    count = 0
    while start < len(text) and count < 5:
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        print(f"  Chunk {count}: start={start} end={end} len={len(chunk)}")
        chunks.append(chunk)
        start = end - OVERLAP
        count += 1
    print("Manual chunking OK")
