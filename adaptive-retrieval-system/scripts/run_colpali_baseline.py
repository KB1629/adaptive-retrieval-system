#!/usr/bin/env python3
"""
Pure ColPali Baseline Benchmark

Run ONLY ColPali (no adaptive routing) on the same DocVQA pages
to get a fair comparison on the same hardware (M1 Pro).

This gives us the true baseline to compare against.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.vision_path import VisionEmbeddingPath
from src.utils.hardware import detect_device, get_hardware_config
from src.models.data import Page


class ColPaliBaselineRunner:
    """Run pure ColPali on all pages (no routing)."""
    
    def __init__(self, output_dir: str = "outputs/benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": None,
                "hardware_info": None,
                "dataset": "DocVQA (ViDoRe benchmark)",
                "model": "Pure ColPali v1.2 (NO adaptive routing)",
            },
            "baseline": {
                "pages_processed": 0,
                "per_page_results": [],
                "latency_stats": {},
            },
        }
        
        print("Initializing Pure ColPali (no adaptive routing)...")
        self._init_components()
        
    def _init_components(self):
        """Initialize ColPali vision embedder."""
        device = detect_device()
        hw_config = get_hardware_config()
        
        self.results["metadata"]["device"] = device
        self.results["metadata"]["hardware_info"] = {
            "device_name": hw_config.device_name,
            "max_memory_gb": hw_config.max_memory_gb,
        }
        
        print(f"Device: {device} ({hw_config.device_name})")
        print("Loading ColPali model...")
        
        self.vision_path = VisionEmbeddingPath()
        
        print("ColPali loaded!")
        
    def load_real_pages(self, data_dir: str = "data/docvqa_sample") -> list[Page]:
        """Load real document pages from DocVQA dataset."""
        data_path = Path(data_dir)
        metadata_path = data_path / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        print(f"Loading pages from {data_path}...")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        pages = []
        # Get project root (parent directory of script's parent directory)
        project_root = Path(__file__).parent.parent.parent
        
        for info in metadata:
            image_path = Path(info['path'])
            # If path is relative, resolve from project root
            if not image_path.is_absolute():
                image_path = project_root / image_path
            
            if not image_path.exists():
                continue
            
            pil_image = Image.open(image_path).convert('RGB')
            image_array = np.array(pil_image)
            
            page = Page.from_array(
                image=image_array,
                page_number=info['page_id'] + 1,
                source_document=info['original_id']
            )
            pages.append(page)
        
        print(f"Loaded {len(pages)} pages")
        return pages
    
    def process_page(self, page: Page, page_idx: int) -> dict:
        """Process a single page with pure ColPali."""
        result = {
            "page_index": page_idx,
            "page_size": f"{page.width}x{page.height}",
            "source_doc": page.source_document,
            "total_latency_ms": 0,
        }
        
        start = time.perf_counter()
        
        try:
            embedding_result = self.vision_path.process_page(page.image)
            result["embedding_dim"] = embedding_result.vector.shape[0]
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
            result["embedding_dim"] = 0
            result["success"] = False
        
        end = time.perf_counter()
        result["total_latency_ms"] = (end - start) * 1000
        
        return result
    
    def run_benchmark(self, data_dir: str = "data/docvqa_sample"):
        """Run pure ColPali on all pages."""
        pages = self.load_real_pages(data_dir)
        num_pages = len(pages)
        
        print(f"\n{'='*60}")
        print(f"PURE COLPALI BASELINE - {num_pages} PAGES")
        print(f"No adaptive routing - ALL pages use ColPali")
        print(f"{'='*60}\n")
        
        print(f"Processing {num_pages} pages with ColPali (Batch Size: 4)...\n")
        
        batch_size = 4
        for i in range(0, num_pages, batch_size):
            batch_pages = pages[i:i + batch_size]
            batch_images = [p.image for p in batch_pages]
            
            print(f"Processing batch {i//batch_size + 1}/{(num_pages + batch_size - 1)//batch_size} (Pages {i+1}-{min(i+batch_size, num_pages)})...", end=" ", flush=True)
            
            try:
                # Process batch
                batch_results = self.vision_path.process_batch(batch_images)
                
                # Store results
                for j, result in enumerate(batch_results):
                    page_idx = i + j
                    page = batch_pages[j]
                    
                    res_dict = {
                        "page_index": page_idx,
                        "page_size": f"{page.width}x{page.height}",
                        "source_doc": page.source_document,
                        "total_latency_ms": result.processing_time_ms,  # Average per page in batch
                        "embedding_dim": result.vector.shape[0],
                        "success": True,
                        "embedding_error": None
                    }
                    self.results["baseline"]["per_page_results"].append(res_dict)
                
                avg_latency = batch_results[0].processing_time_ms if batch_results else 0
                print(f"✓ {avg_latency:.1f}ms/page")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                # Record failures
                for j in range(len(batch_pages)):
                    page_idx = i + j
                    page = batch_pages[j]
                    self.results["baseline"]["per_page_results"].append({
                        "page_index": page_idx,
                        "page_size": f"{page.width}x{page.height}",
                        "source_doc": page.source_document,
                        "total_latency_ms": 0,
                        "embedding_dim": 0,
                        "success": False,
                        "embedding_error": str(e)
                    })
        
        self.results["baseline"]["pages_processed"] = num_pages
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
    def _calculate_statistics(self):
        """Calculate latency statistics."""
        all_latencies = [r["total_latency_ms"] for r in self.results["baseline"]["per_page_results"]]
        
        # Exclude first page (cold start)
        if len(all_latencies) > 1:
            latencies = all_latencies[1:]
        else:
            latencies = all_latencies
        
        self.results["baseline"]["latency_stats"] = {
            "total": {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "std_ms": float(np.std(latencies)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
            },
            "cold_start": {
                "first_page_ms": float(all_latencies[0]) if all_latencies else 0,
            },
            "note": "Statistics exclude first page (cold start)",
        }
        
    def _save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = self.output_dir / f"colpali_baseline_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")
        
        # Save CSV
        csv_path = self.output_dir / f"colpali_baseline_{timestamp}.csv"
        with open(csv_path, "w") as f:
            f.write("page_index,source_doc,page_size,total_ms,success\n")
            for r in self.results["baseline"]["per_page_results"]:
                f.write(f"{r['page_index']},{r['source_doc']},{r['page_size']},")
                f.write(f"{r['total_latency_ms']:.2f},{r.get('success', False)}\n")
        print(f"CSV saved to: {csv_path}")
        
        # Save report
        report_path = self.output_dir / f"colpali_baseline_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(self._generate_report())
        print(f"Report saved to: {report_path}")
        
    def _generate_report(self) -> str:
        """Generate report."""
        stats = self.results["baseline"]["latency_stats"]["total"]
        cold_start = self.results["baseline"]["latency_stats"]["cold_start"]
        
        report = f"""
{'='*70}
PURE COLPALI BASELINE - BENCHMARK REPORT
{'='*70}

Generated: {self.results['metadata']['timestamp']}
Device: {self.results['metadata']['device']} ({self.results['metadata']['hardware_info']['device_name']})
Dataset: {self.results['metadata']['dataset']}
Model: {self.results['metadata']['model']}

{'='*70}
METHODOLOGY
{'='*70}

- Pure ColPali v1.2 (vidore/colpali-v1.2)
- NO adaptive routing - ALL pages processed with ColPali
- Hardware: {self.results['metadata']['hardware_info']['device_name']}
- Pages: {self.results['baseline']['pages_processed']} real DocVQA documents

{'='*70}
LATENCY MEASUREMENTS (Pure ColPali)
{'='*70}

Cold Start (First Page):
  First Page: {cold_start['first_page_ms']:.2f} ms (includes model loading)

Steady-State Latency (excluding cold start):
  Mean:   {stats['mean_ms']:.2f} ms
  Median: {stats['median_ms']:.2f} ms
  Std:    {stats['std_ms']:.2f} ms
  Min:    {stats['min_ms']:.2f} ms
  Max:    {stats['max_ms']:.2f} ms
  P95:    {stats['p95_ms']:.2f} ms

{'='*70}
SUMMARY
{'='*70}

Pure ColPali on M1 Pro: {stats['mean_ms']:.2f} ms/page (average)

This is the baseline to compare against our adaptive system.

{'='*70}
"""
        return report
        
    def _print_summary(self):
        """Print summary."""
        print(self._generate_report())


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pure ColPali baseline")
    parser.add_argument("--data-dir", type=str, default="data/docvqa_sample", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    runner = ColPaliBaselineRunner(output_dir=args.output_dir)
    runner.run_benchmark(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
