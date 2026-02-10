#!/usr/bin/env python3
"""
Generate visualization charts comparing Pure ColPali vs Adaptive System.

Creates:
1. Side-by-side latency comparison
2. Per-page latency comparison
3. Router classification breakdown
4. Speedup visualization
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(baseline_path: str, adaptive_path: str) -> tuple[dict, dict]:
    """Load both benchmark results."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(adaptive_path) as f:
        adaptive = json.load(f)
    return baseline, adaptive


def create_latency_comparison_chart(baseline: dict, adaptive: dict, output_dir: Path):
    """Create bar chart comparing Pure ColPali vs Adaptive system."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_mean = baseline["baseline"]["latency_stats"]["total"]["mean_ms"]
    adaptive_mean = adaptive["adaptive_system"]["latency_stats"]["total"]["mean_ms"]
    speedup = baseline_mean / adaptive_mean
    
    systems = ["Pure ColPali\n(No Routing)", "Adaptive System\n(With Routing)"]
    latencies = [baseline_mean, adaptive_mean]
    colors = ["#e74c3c", "#27ae60"]
    
    bars = ax.bar(systems, latencies, color=colors, width=0.6, edgecolor="black", linewidth=2)
    
    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.annotate(f'{latency:.0f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel("Mean Latency per Page (ms)", fontsize=12)
    ax.set_title(f"Latency Comparison on M1 Pro: {speedup:.2f}x Speedup", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(latencies) * 1.2)
    
    # Add speedup annotation
    ax.text(0.5, max(latencies) * 1.1, 
            f'Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)',
            ha='center', fontsize=12, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "comparison_latency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_per_page_comparison_chart(baseline: dict, adaptive: dict, output_dir: Path):
    """Create line chart comparing per-page latency."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Skip first page (cold start) for both
    baseline_pages = baseline["baseline"]["per_page_results"][1:]
    adaptive_pages = adaptive["adaptive_system"]["per_page_results"][1:]
    
    pages = [r["page_index"] for r in baseline_pages]
    baseline_latencies = [r["total_latency_ms"] for r in baseline_pages]
    adaptive_latencies = [r["total_latency_ms"] for r in adaptive_pages]
    
    # Plot both lines
    ax.plot(pages, baseline_latencies, 'o-', color='#e74c3c', linewidth=2, 
            markersize=6, label='Pure ColPali', alpha=0.8)
    ax.plot(pages, adaptive_latencies, 's-', color='#27ae60', linewidth=2, 
            markersize=6, label='Adaptive System', alpha=0.8)
    
    # Add mean lines
    baseline_mean = np.mean(baseline_latencies)
    adaptive_mean = np.mean(adaptive_latencies)
    ax.axhline(y=baseline_mean, color='#e74c3c', linestyle='--', linewidth=1.5, 
               alpha=0.5, label=f'Pure ColPali Mean ({baseline_mean:.0f}ms)')
    ax.axhline(y=adaptive_mean, color='#27ae60', linestyle='--', linewidth=1.5, 
               alpha=0.5, label=f'Adaptive Mean ({adaptive_mean:.0f}ms)')
    
    ax.set_xlabel("Page Index", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Per-Page Latency Comparison (excluding cold start)", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "comparison_per_page.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_router_breakdown_chart(adaptive: dict, output_dir: Path):
    """Create pie chart showing router classification."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    router_stats = adaptive["adaptive_system"]["router_stats"]
    
    labels = [
        f"Text-Heavy\n({router_stats['text_heavy']} pages)",
        f"Visual-Critical\n({router_stats['visual_critical']} pages)"
    ]
    sizes = [router_stats["text_heavy"], router_stats["visual_critical"]]
    colors = ["#27ae60", "#3498db"]
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.0f%%', shadow=True, startangle=90,
        textprops={'fontsize': 12}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title("Router Classification Distribution\n(DocVQA Dataset)", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "comparison_router_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_speedup_breakdown_chart(baseline: dict, adaptive: dict, output_dir: Path):
    """Create chart showing where speedup comes from."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baseline_mean = baseline["baseline"]["latency_stats"]["total"]["mean_ms"]
    adaptive_mean = adaptive["adaptive_system"]["latency_stats"]["total"]["mean_ms"]
    router_mean = adaptive["adaptive_system"]["latency_stats"]["router"]["mean_ms"]
    embedding_mean = adaptive["adaptive_system"]["latency_stats"]["embedding"]["mean_ms"]
    
    # Create stacked bar
    categories = ['Pure ColPali', 'Adaptive System']
    
    # Pure ColPali: all embedding
    colpali_embedding = [baseline_mean]
    colpali_router = [0]
    
    # Adaptive: router + embedding
    adaptive_router = [router_mean]
    adaptive_embedding = [embedding_mean]
    
    x = np.arange(len(categories))
    width = 0.5
    
    # Plot stacked bars
    p1 = ax.bar([0], colpali_embedding, width, label='Embedding', color='#e74c3c')
    p2 = ax.bar([1], adaptive_embedding, width, label='Embedding', color='#3498db')
    p3 = ax.bar([1], adaptive_router, width, bottom=adaptive_embedding, 
                label='Router', color='#9b59b6')
    
    # Add value labels
    ax.text(0, baseline_mean/2, f'{baseline_mean:.0f} ms', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(1, embedding_mean/2, f'{embedding_mean:.0f} ms', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(1, embedding_mean + router_mean/2, f'{router_mean:.0f} ms', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Add total labels on top
    ax.text(0, baseline_mean + 100, f'Total: {baseline_mean:.0f} ms', 
            ha='center', fontsize=11, fontweight='bold')
    ax.text(1, adaptive_mean + 100, f'Total: {adaptive_mean:.0f} ms', 
            ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Component Breakdown: Where Does Time Go?', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, max(baseline_mean, adaptive_mean) * 1.15)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Pure ColPali Embedding'),
        Patch(facecolor='#3498db', label='Adaptive Embedding'),
        Patch(facecolor='#9b59b6', label='Router Overhead')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_path = output_dir / "comparison_component_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_stats_chart(baseline: dict, adaptive: dict, output_dir: Path):
    """Create table showing summary statistics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    baseline_stats = baseline["baseline"]["latency_stats"]["total"]
    adaptive_stats = adaptive["adaptive_system"]["latency_stats"]["total"]
    
    speedup = baseline_stats["mean_ms"] / adaptive_stats["mean_ms"]
    
    table_data = [
        ["Metric", "Pure ColPali", "Adaptive System", "Improvement"],
        ["Mean Latency", f"{baseline_stats['mean_ms']:.1f} ms", 
         f"{adaptive_stats['mean_ms']:.1f} ms", 
         f"{speedup:.2f}x faster"],
        ["Median Latency", f"{baseline_stats['median_ms']:.1f} ms", 
         f"{adaptive_stats['median_ms']:.1f} ms", 
         f"{baseline_stats['median_ms']/adaptive_stats['median_ms']:.2f}x"],
        ["Std Deviation", f"{baseline_stats['std_ms']:.1f} ms", 
         f"{adaptive_stats['std_ms']:.1f} ms", ""],
        ["Min Latency", f"{baseline_stats['min_ms']:.1f} ms", 
         f"{adaptive_stats['min_ms']:.1f} ms", ""],
        ["Max Latency", f"{baseline_stats['max_ms']:.1f} ms", 
         f"{adaptive_stats['max_ms']:.1f} ms", ""],
        ["P95 Latency", f"{baseline_stats['p95_ms']:.1f} ms", 
         f"{adaptive_stats['p95_ms']:.1f} ms", ""],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Highlight improvement column
    for i in range(1, len(table_data)):
        if table_data[i][3]:
            table[(i, 3)].set_facecolor('#d5f4e6')
            table[(i, 3)].set_text_props(weight='bold')
    
    plt.title("Benchmark Statistics Summary\n(M1 Pro, 20 DocVQA Pages, Excluding Cold Start)", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / "comparison_summary_table.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all comparison charts."""
    baseline_path = "adaptive-retrieval-system/outputs/benchmark_results/colpali_baseline_20260204_144550.json"
    adaptive_path = "adaptive-retrieval-system/outputs/benchmark_results/real_benchmark_20260204_140646.json"
    output_dir = Path("adaptive-retrieval-system/outputs/benchmark_results")
    
    print("Loading benchmark results...")
    baseline, adaptive = load_results(baseline_path, adaptive_path)
    
    print("\nGenerating comparison charts...")
    create_latency_comparison_chart(baseline, adaptive, output_dir)
    create_per_page_comparison_chart(baseline, adaptive, output_dir)
    create_router_breakdown_chart(adaptive, output_dir)
    create_speedup_breakdown_chart(baseline, adaptive, output_dir)
    create_summary_stats_chart(baseline, adaptive, output_dir)
    
    print("\n✅ All comparison charts generated successfully!")
    
    # Print summary
    baseline_mean = baseline["baseline"]["latency_stats"]["total"]["mean_ms"]
    adaptive_mean = adaptive["adaptive_system"]["latency_stats"]["total"]["mean_ms"]
    speedup = baseline_mean / adaptive_mean
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Pure ColPali:     {baseline_mean:.2f} ms/page")
    print(f"Adaptive System:  {adaptive_mean:.2f} ms/page")
    print(f"Speedup:          {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
