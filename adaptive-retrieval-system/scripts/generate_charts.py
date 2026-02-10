#!/usr/bin/env python3
"""
Generate visualization charts from benchmark results.

Creates:
1. Latency comparison bar chart (ColPali vs Adaptive)
2. Per-page latency line chart
3. Router classification pie chart
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Find the latest benchmark results
def find_latest_results(output_dir: str = "outputs/benchmark_results") -> Path:
    """Find the most recent benchmark results JSON file."""
    results_dir = Path(output_dir)
    json_files = sorted(results_dir.glob("benchmark_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No benchmark results found in {output_dir}")
    return json_files[-1]


def load_results(results_path: Path) -> dict:
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def create_latency_comparison_chart(results: dict, output_dir: Path):
    """Create bar chart comparing ColPali vs Adaptive system latency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colpali_latency = results["colpali_baseline"]["latency_ms_per_page"]
    adaptive_latency = results["comparison"]["our_mean_latency_ms"]
    speedup = results["comparison"]["speedup_factor"]
    
    systems = ["ColPali\n(Published Baseline)", "Adaptive System\n(Our Approach)"]
    latencies = [colpali_latency, adaptive_latency]
    colors = ["#e74c3c", "#27ae60"]
    
    bars = ax.bar(systems, latencies, color=colors, width=0.6, edgecolor="black")
    
    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.annotate(f'{latency:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel("Latency per Page (ms)", fontsize=12)
    ax.set_title(f"Latency Comparison: {speedup:.2f}x Speedup Achieved", fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(latencies) * 1.2)
    
    # Add target line
    target_latency = colpali_latency * 0.5
    ax.axhline(y=target_latency, color='orange', linestyle='--', linewidth=2, label=f'50% Target ({target_latency:.0f}ms)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = output_dir / "latency_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_per_page_latency_chart(results: dict, output_dir: Path):
    """Create line chart showing per-page latency."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    per_page = results["adaptive_system"]["per_page_results"]
    
    # Skip first page (cold start)
    pages = [r["page_index"] for r in per_page[1:]]
    latencies = [r["total_latency_ms"] for r in per_page[1:]]
    paths = [r["embedding_path"] for r in per_page[1:]]
    
    # Color by path
    colors = ["#27ae60" if p == "text" else "#3498db" for p in paths]
    
    ax.bar(pages, latencies, color=colors, edgecolor="black", alpha=0.8)
    
    # Add ColPali baseline
    ax.axhline(y=400, color='#e74c3c', linestyle='--', linewidth=2, label='ColPali Baseline (400ms)')
    
    # Add mean line
    mean_latency = np.mean(latencies)
    ax.axhline(y=mean_latency, color='#f39c12', linestyle='-', linewidth=2, label=f'Our Mean ({mean_latency:.1f}ms)')
    
    ax.set_xlabel("Page Index", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Per-Page Latency (excluding cold start)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', label='Text Path'),
        Patch(facecolor='#3498db', label='Vision Path'),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='upper right')
    
    plt.tight_layout()
    output_path = output_dir / "per_page_latency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_router_pie_chart(results: dict, output_dir: Path):
    """Create pie chart showing router classification distribution."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    router_stats = results["adaptive_system"]["router_stats"]
    
    labels = ["Text-Heavy\n(Fast Path)", "Visual-Critical\n(Vision Path)"]
    sizes = [router_stats["text_heavy"], router_stats["visual_critical"]]
    colors = ["#27ae60", "#3498db"]
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 12}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    ax.set_title("Router Classification Distribution", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "router_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_component_breakdown_chart(results: dict, output_dir: Path):
    """Create stacked bar chart showing component breakdown."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    latency_stats = results["adaptive_system"]["latency_stats"]
    
    components = ["Router", "Embedding"]
    times = [
        latency_stats["router"]["mean_ms"],
        latency_stats["embedding"]["mean_ms"]
    ]
    
    colors = ["#9b59b6", "#1abc9c"]
    
    bars = ax.barh(components, times, color=colors, edgecolor="black", height=0.5)
    
    # Add value labels
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.annotate(f'{time:.1f} ms',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12, fontweight='bold')
    
    total = sum(times)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_title(f"Component Breakdown (Total: {total:.1f}ms)", fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(times) * 1.3)
    
    plt.tight_layout()
    output_path = output_dir / "component_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all charts."""
    output_dir = Path("outputs/benchmark_results")
    
    print("Finding latest benchmark results...")
    results_path = find_latest_results()
    print(f"Using: {results_path}")
    
    results = load_results(results_path)
    
    print("\nGenerating charts...")
    create_latency_comparison_chart(results, output_dir)
    create_per_page_latency_chart(results, output_dir)
    create_router_pie_chart(results, output_dir)
    create_component_breakdown_chart(results, output_dir)
    
    print("\nAll charts generated successfully!")
    print(f"\nSummary:")
    print(f"  - Speedup: {results['comparison']['speedup_factor']:.2f}x")
    print(f"  - Latency Reduction: {results['comparison']['latency_reduction_percent']:.1f}%")
    print(f"  - Target Met: {'✅ YES' if results['comparison']['meets_target'] else '❌ NO'}")


if __name__ == "__main__":
    main()
