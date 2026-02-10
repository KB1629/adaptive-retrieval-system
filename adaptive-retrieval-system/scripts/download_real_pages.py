#!/usr/bin/env python3
"""
Download real document pages from DocVQA dataset.

This downloads a small sample of actual PDF pages that were used
in the ColPali paper benchmarks.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from PIL import Image
import numpy as np

def download_docvqa_sample(num_pages: int = 20, output_dir: str = "data/docvqa_sample"):
    """
    Download sample pages from DocVQA dataset.
    
    Args:
        num_pages: Number of pages to download
        output_dir: Directory to save pages
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_pages} pages from DocVQA dataset...")
    print("This may take a few minutes on first run (dataset will be cached)...")
    
    # Load DocVQA validation set (smaller than train)
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    
    pages = []
    page_metadata = []
    
    for i, sample in enumerate(dataset):
        if i >= num_pages:
            break
        
        # Get the image
        image = sample['image']
        
        # Convert to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Ensure RGB
        if len(image_array.shape) == 2:
            # Grayscale to RGB
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            # RGBA to RGB
            image_array = image_array[:, :, :3]
        
        pages.append(image_array)
        
        # Save metadata
        metadata = {
            "page_index": i,
            "doc_id": sample.get('doc_id', f'doc_{i}'),
            "question": sample.get('question', ''),
            "width": image_array.shape[1],
            "height": image_array.shape[0],
        }
        page_metadata.append(metadata)
        
        # Save image to disk
        img_path = output_path / f"page_{i:03d}.png"
        Image.fromarray(image_array).save(img_path)
        
        print(f"Downloaded page {i+1}/{num_pages}: {metadata['doc_id']} ({metadata['width']}x{metadata['height']})")
    
    print(f"\n✓ Downloaded {len(pages)} pages to {output_path}")
    print(f"  Total size: {sum(p.nbytes for p in pages) / 1024 / 1024:.1f} MB")
    
    return pages, page_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real document pages from DocVQA")
    parser.add_argument("--pages", type=int, default=20, help="Number of pages to download")
    parser.add_argument("--output-dir", type=str, default="data/docvqa_sample", help="Output directory")
    
    args = parser.parse_args()
    
    download_docvqa_sample(num_pages=args.pages, output_dir=args.output_dir)
