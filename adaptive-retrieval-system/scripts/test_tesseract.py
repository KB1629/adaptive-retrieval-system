#!/usr/bin/env python3
"""
Quick test: Tesseract vs PaddleOCR performance comparison.

Tests both OCR engines on 10 sample pages to compare:
- Speed (time per page)
- Text extraction quality
- Suitability for retrieval
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.text_extractor import TextExtractor, _resize_for_ocr

def test_tesseract_ocr(image: np.ndarray) -> tuple[str, float]:
    """
    Test Tesseract OCR on an image.
    
    Returns:
        (extracted_text, time_seconds)
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Resize image (same as PaddleOCR)
        image = _resize_for_ocr(image)
        
        # Convert to PIL
        pil_img = Image.fromarray(image)
        
        # Run Tesseract
        start = time.time()
        text = pytesseract.image_to_string(pil_img)
        elapsed = time.time() - start
        
        return text.strip(), elapsed
        
    except Exception as e:
        return f"ERROR: {e}", 0.0


def test_paddleocr(image: np.ndarray, extractor: TextExtractor) -> tuple[str, float]:
    """
    Test PaddleOCR on an image.
    
    Returns:
        (extracted_text, time_seconds)
    """
    try:
        start = time.time()
        text = extractor.extract_from_image(image)
        elapsed = time.time() - start
        return text, elapsed
    except Exception as e:
        return f"ERROR: {e}", 0.0


def load_sample_pages(data_dir: str, num_pages: int = 10):
    """Load sample pages from DocVQA dataset."""
    metadata_path = Path(data_dir) / "metadata.json"
    
    # Load metadata (standard JSON array)
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    pages = []
    for i, item in enumerate(metadata[:num_pages]):
        # Get image path - DocVQA uses 'path' field
        img_rel_path = item.get('path')
        if not img_rel_path:
            continue
            
        # Path is already relative to data root, so use absolute from metadata location
        img_path = Path(img_rel_path)
        if not img_path.is_absolute():
            img_path = Path(data_dir).parent.parent / img_rel_path
        
        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
            pages.append({
                'index': i,
                'path': str(img_path),
                'image': np.array(img),
                'size': f"{img.width}x{img.height}"
            })
    
    return pages


def main():
    print("=" * 70)
    print("TESSERACT vs PADDLEOCR COMPARISON TEST")
    print("=" * 70)
    print()
    
    # Check if Tesseract is installed
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract installed: v{version}")
    except Exception as e:
        print(f"✗ Tesseract not available: {e}")
        print("\nInstall with: brew install tesseract")
        return 1
    
    print()
    
    # Load sample pages
    data_dir = "/Users/Kabeleswar.pe/adaptive-retrieval-system/data/docvqa_sample"
    print(f"Loading 10 sample pages from {data_dir}...")
    pages = load_sample_pages(data_dir, num_pages=10)
    print(f"✓ Loaded {len(pages)} pages")
    print()
    
    # Initialize PaddleOCR
    print("Initializing PaddleOCR...")
    paddle_extractor = TextExtractor()
    print("✓ PaddleOCR ready")
    print()
    
    # Test both engines
    print("=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)
    print()
    
    results = []
    
    for page in pages:
        print(f"Page {page['index']+1}/10 ({page['size']}):")
        
        # Test Tesseract
        print("  Testing Tesseract...", end=" ", flush=True)
        tess_text, tess_time = test_tesseract_ocr(page['image'])
        tess_chars = len(tess_text)
        print(f"✓ {tess_time:.2f}s ({tess_chars} chars)")
        
        # Test PaddleOCR
        print("  Testing PaddleOCR...", end=" ", flush=True)
        paddle_text, paddle_time = test_paddleocr(page['image'], paddle_extractor)
        paddle_chars = len(paddle_text)
        print(f"✓ {paddle_time:.2f}s ({paddle_chars} chars)")
        
        # Calculate speedup
        if paddle_time > 0:
            speedup = paddle_time / tess_time
            print(f"  → Tesseract is {speedup:.1f}x faster")
        
        results.append({
            'page': page['index'],
            'size': page['size'],
            'tesseract_time': tess_time,
            'tesseract_chars': tess_chars,
            'paddleocr_time': paddle_time,
            'paddleocr_chars': paddle_chars,
            'speedup': speedup if paddle_time > 0 else 0
        })
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    avg_tess_time = np.mean([r['tesseract_time'] for r in results])
    avg_paddle_time = np.mean([r['paddleocr_time'] for r in results])
    avg_speedup = np.mean([r['speedup'] for r in results])
    
    print(f"Average Tesseract time:  {avg_tess_time:.2f}s/page")
    print(f"Average PaddleOCR time:  {avg_paddle_time:.2f}s/page")
    print(f"Average speedup:         {avg_speedup:.1f}x faster with Tesseract")
    print()
    
    # Character count comparison (rough quality metric)
    avg_tess_chars = np.mean([r['tesseract_chars'] for r in results])
    avg_paddle_chars = np.mean([r['paddleocr_chars'] for r in results])
    char_ratio = (avg_tess_chars / avg_paddle_chars) * 100 if avg_paddle_chars > 0 else 0
    
    print(f"Average chars extracted:")
    print(f"  Tesseract: {avg_tess_chars:.0f} chars")
    print(f"  PaddleOCR: {avg_paddle_chars:.0f} chars")
    print(f"  Ratio: {char_ratio:.1f}% (Tesseract vs PaddleOCR)")
    print()
    
    # Recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()
    
    if avg_tess_time < 5.0 and avg_speedup > 3.0:
        print("✅ RECOMMENDED: Switch to Tesseract!")
        print(f"   - Speed: {avg_tess_time:.2f}s/page (vs ColPali baseline 2.82s/page)")
        print(f"   - {avg_speedup:.1f}x faster than PaddleOCR")
        print(f"   - Text extraction quality: {char_ratio:.0f}% of PaddleOCR")
        print()
        print("   This would make your text path COMPETITIVE with ColPali!")
    elif avg_speedup > 2.0:
        print("⚠️  CONSIDER: Tesseract is faster but may need quality validation")
        print(f"   - Speed: {avg_tess_time:.2f}s/page")
        print(f"   - {avg_speedup:.1f}x faster than PaddleOCR")
        print(f"   - Test retrieval quality on your queries")
    else:
        print("❌ STICK WITH PADDLEOCR: Tesseract not significantly faster")
    
    print()
    
    # Save results
    output_file = "outputs/tesseract_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'avg_tesseract_time': avg_tess_time,
                'avg_paddleocr_time': avg_paddle_time,
                'avg_speedup': avg_speedup,
                'avg_tesseract_chars': avg_tess_chars,
                'avg_paddleocr_chars': avg_paddle_chars,
            },
            'per_page_results': results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
