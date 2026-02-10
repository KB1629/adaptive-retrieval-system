#!/usr/bin/env python3
"""
Download real document pages from DocVQA dataset (used in ColPali paper).

This downloads a small subset of actual PDF pages for benchmarking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
import numpy as np
from PIL import Image

def download_docvqa_sample(num_pages: int = 20, output_dir: str = "data/docvqa_sample"):
    """
    Download sample pages from DocVQA dataset.
    
    DocVQA is one of the datasets used in the ColPali paper's ViDoRe benchmark.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_pages} pages from DocVQA dataset...")
    print("This is the same dataset used in the ColPali paper (ViDoRe benchmark)")
    
    # Load DocVQA validation set (smaller than train)
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    
    pages_saved = 0
    page_info = []
    
    for idx, sample in enumerate(dataset):
        if pages_saved >= num_pages:
            break
        
        try:
            # Get image
            image = sample['image']
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save image
            image_path = output_path / f"page_{pages_saved:03d}.png"
            image.save(image_path)
            
            # Save metadata
            page_info.append({
                "page_id": pages_saved,
                "original_id": sample.get('questionId', f'doc_{idx}'),
                "question": sample.get('question', ''),
                "width": image.width,
                "height": image.height,
                "path": str(image_path)
            })
            
            pages_saved += 1
            print(f"Saved page {pages_saved}/{num_pages}: {image.width}x{image.height}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save metadata
    import json
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(page_info, f, indent=2)
    
    print(f"\nDownloaded {pages_saved} pages to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return output_path, page_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DocVQA sample pages")
    parser.add_argument("--pages", type=int, default=20, help="Number of pages to download")
    parser.add_argument("--output-dir", type=str, default="data/docvqa_sample", help="Output directory")
    
    args = parser.parse_args()
    
    download_docvqa_sample(num_pages=args.pages, output_dir=args.output_dir)
