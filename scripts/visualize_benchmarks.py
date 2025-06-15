#\!/usr/bin/env python3
"""
Benchmark visualization and analysis script for RAGnificent Rust components.

This script processes Criterion benchmark results and creates comprehensive
performance reports and visualizations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Plotting libraries not available. Install with: pip install matplotlib seaborn pandas numpy")

def load_benchmark_data(criterion_dir: Path) -> Dict[str, Any]:
    """Load all benchmark data from Criterion output directory."""
    benchmark_data = {}
    
    if not criterion_dir.exists():
        print(f"Criterion directory not found: {criterion_dir}")
        return {}
    
    # Traverse the criterion directory structure
    for group_dir in criterion_dir.iterdir():
        if group_dir.is_dir():
            group_name = group_dir.name
            benchmark_data[group_name] = {}
            
            for benchmark_dir in group_dir.iterdir():
                if benchmark_dir.is_dir():
                    benchmark_name = benchmark_dir.name
                    benchmark_data[group_name][benchmark_name] = {}
                    
                    for size_dir in benchmark_dir.iterdir():
                        if size_dir.is_dir() and (size_dir / "new" / "estimates.json").exists():
                            size_name = size_dir.name
                            estimates_file = size_dir / "new" / "estimates.json"
                            
                            try:
                                with open(estimates_file) as f:
                                    estimates = json.load(f)
                                    benchmark_data[group_name][benchmark_name][size_name] = estimates
                            except Exception as e:
                                print(f"Error loading {estimates_file}: {e}")
    
    return benchmark_data

def setup_dark_style():
    """Setup dark mode styling for plots."""
    if not PLOTTING_AVAILABLE:
        return
    
    # Set dark background style
    plt.style.use('dark_background')
    
    # Custom color palette - cyberpunk inspired
    colors = ['#ff006e', '#fb5607', '#ffbe0b', '#8338ec', '#3a86ff', '#06ffa5']
    sns.set_palette(colors)
    
    # Set global font and styling
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'monospace',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#444444',
        'axes.edgecolor': '#888888',
        'axes.linewidth': 1.5,
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
        'axes.labelcolor': '#ffffff',
        'text.color': '#ffffff',
        'figure.facecolor': '#0a0a0a',
        'axes.facecolor': '#1a1a1a',
        'savefig.facecolor': '#0a0a0a',
        'savefig.edgecolor': 'none',
    })

def create_performance_summary(data: Dict[str, Any]) -> List[Dict]:
    """Create a summary list of benchmark results."""
    rows = []
    
    for group_name, group_data in data.items():
        for benchmark_name, benchmark_data in group_data.items():
            for size_name, size_data in benchmark_data.items():
                if "mean" in size_data:
                    mean_time = size_data["mean"]["point_estimate"] / 1e6  # Convert to ms
                    std_dev = size_data["std_dev"]["point_estimate"] / 1e6  # Convert to ms
                    
                    rows.append({
                        "group": group_name,
                        "benchmark": benchmark_name,
                        "size": size_name,
                        "mean_time_ms": mean_time,
                        "std_dev_ms": std_dev,
                        "relative_std": std_dev / mean_time if mean_time > 0 else 0
                    })
    
    return rows

def create_sexy_visualizations(data: List[Dict], output_dir: Path):
    """Create dark mode, visually appealing performance charts."""
    if not PLOTTING_AVAILABLE or not data:
        return
    
    # Setup dark styling
    setup_dark_style()
    
    df = pd.DataFrame(data)
    
    # 1. Cyberpunk Performance Dashboard
    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    benchmarks = df['benchmark'].unique()
    sizes = df['size'].unique()
    
    # Create grouped bar chart with glow effect
    x = np.arange(len(benchmarks))
    width = 0.35
    
    for i, size in enumerate(sizes):
        size_data = df[df['size'] == size]
        times = [size_data[size_data['benchmark'] == b]['mean_time_ms'].iloc[0] 
                if len(size_data[size_data['benchmark'] == b]) > 0 else 0 
                for b in benchmarks]
        
        bars = ax1.bar(x + i*width, times, width, 
                      label=size.upper(), alpha=0.8, 
                      color=sns.color_palette()[i])
        
        # Add glow effect
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.add_patch(patches.Rectangle(
                    (bar.get_x() - 0.02, -0.001), 
                    bar.get_width() + 0.04, height + 0.002,
                    facecolor=bar.get_facecolor(), alpha=0.3, 
                    linewidth=0))
    
    ax1.set_xlabel('BENCHMARK OPERATIONS', fontweight='bold', fontsize=14)
    ax1.set_ylabel('EXECUTION TIME (ms)', fontweight='bold', fontsize=14)
    ax1.set_title('🚀 PERFORMANCE ANALYSIS - RAGnificent Rust Engine', 
                  fontsize=18, fontweight='bold', color='#ff006e', pad=20)
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([b.replace('_', ' ').title() for b in benchmarks], 
                       rotation=45, ha='right')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Performance heatmap
    ax2 = fig.add_subplot(gs[1, :2])
    pivot_data = df.pivot_table(index='benchmark', columns='size', 
                               values='mean_time_ms', fill_value=0)
    
    # Custom colormap for dark theme
    colors_heat = ['#1a1a1a', '#8338ec', '#ff006e', '#ffbe0b']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('dark_heat', colors_heat, N=n_bins)
    
    im = ax2.imshow(pivot_data.values, cmap=cmap, aspect='auto', alpha=0.9)
    ax2.set_xticks(range(len(pivot_data.columns)))
    ax2.set_yticks(range(len(pivot_data.index)))
    ax2.set_xticklabels(pivot_data.columns)
    ax2.set_yticklabels([b.replace('_', ' ').title() for b in pivot_data.index])
    ax2.set_title('⚡ PERFORMANCE HEATMAP', fontweight='bold', fontsize=16, color='#3a86ff')
    
    # Add value annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            ax2.text(j, i, f'{value:.3f}', ha='center', va='center',
                    color='white', fontweight='bold')
    
    # Speed distribution
    ax3 = fig.add_subplot(gs[0:2, 2])
    mean_times = df['mean_time_ms'].values
    n, bins, patches_hist = ax3.hist(mean_times, bins=10, alpha=0.8, 
                                    color='#06ffa5', edgecolor='#ffffff', linewidth=1.5)
    
    # Add gradient effect to histogram
    for i, p in enumerate(patches_hist):
        p.set_facecolor(plt.cm.plasma(i / len(patches_hist)))
    
    ax3.set_xlabel('EXECUTION TIME (ms)', fontweight='bold')
    ax3.set_ylabel('FREQUENCY', fontweight='bold')
    ax3.set_title('📊 SPEED DISTRIBUTION', fontweight='bold', color='#06ffa5')
    ax3.grid(True, alpha=0.3)
    
    # Performance metrics dashboard
    ax4 = fig.add_subplot(gs[2, :])
    metrics_text = f"""
🎯 PERFORMANCE METRICS DASHBOARD 🎯

⚡ FASTEST: {df.loc[df['mean_time_ms'].idxmin(), 'benchmark'].replace('_', ' ').title()} → {df['mean_time_ms'].min():.4f} ms
🐌 SLOWEST: {df.loc[df['mean_time_ms'].idxmax(), 'benchmark'].replace('_', ' ').title()} → {df['mean_time_ms'].max():.4f} ms
📈 AVERAGE: {df['mean_time_ms'].mean():.4f} ms
🎖️  MEDIAN: {df['mean_time_ms'].median():.4f} ms
🔥 COEFFICIENT OF VARIATION: {(df['std_dev_ms'].mean() / df['mean_time_ms'].mean() * 100):.1f}%
"""
    
    ax4.text(0.05, 0.5, metrics_text, transform=ax4.transAxes, 
            fontsize=14, fontfamily='monospace', fontweight='bold',
            verticalalignment='center', color='#ffffff',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#333333', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.savefig(output_dir / 'cyberpunk_performance_dashboard.png', 
                dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    # 2. SIMD Performance Analysis (if available)
    simd_data = [row for row in data if 'simd' in row['benchmark'].lower()]
    if simd_data:
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0a0a0a')
        
        simd_df = pd.DataFrame(simd_data)
        
        # Create radar chart style visualization
        benchmarks = simd_df['benchmark'].unique()
        angles = np.linspace(0, 2*np.pi, len(benchmarks), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for size in simd_df['size'].unique():
            size_data = simd_df[simd_df['size'] == size]
            values = []
            for bench in benchmarks:
                val = size_data[size_data['benchmark'] == bench]['mean_time_ms']
                values.append(val.iloc[0] if len(val) > 0 else 0)
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=3, label=size.upper(), alpha=0.8)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([b.replace('_', ' ').title() for b in benchmarks])
        ax.set_title('🔥 SIMD OPERATIONS RADAR', fontsize=18, fontweight='bold', 
                    color='#ff006e', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / 'simd_radar_chart.png', 
                    dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close()
    
    # 3. Performance Timeline/Scaling Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='#0a0a0a')
    
    # Top-left: Performance by size
    ax = axes[0, 0]
    for benchmark in df['benchmark'].unique():
        bench_data = df[df['benchmark'] == benchmark]
        ax.plot(bench_data['size'], bench_data['mean_time_ms'], 
               'o-', linewidth=3, markersize=8, 
               label=benchmark.replace('_', ' ').title(), alpha=0.8)
    ax.set_xlabel('INPUT SIZE', fontweight='bold')
    ax.set_ylabel('TIME (ms)', fontweight='bold')
    ax.set_title('📈 SCALING ANALYSIS', fontweight='bold', color='#3a86ff')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Top-right: Coefficient of Variation
    ax = axes[0, 1]
    cv_values = df['relative_std'] * 100
    bars = ax.bar(range(len(cv_values)), cv_values, color=sns.color_palette(), alpha=0.8)
    ax.set_xlabel('BENCHMARK INDEX', fontweight='bold')
    ax.set_ylabel('COEFFICIENT OF VARIATION (%)', fontweight='bold')
    ax.set_title('🎯 STABILITY ANALYSIS', fontweight='bold', color='#ffbe0b')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Performance comparison scatter
    ax = axes[1, 0]
    scatter = ax.scatter(df['mean_time_ms'], df['std_dev_ms'], 
                        c=range(len(df)), cmap='plasma', s=100, alpha=0.8)
    ax.set_xlabel('MEAN TIME (ms)', fontweight='bold')
    ax.set_ylabel('STD DEVIATION (ms)', fontweight='bold')
    ax.set_title('🎪 PERFORMANCE vs STABILITY', fontweight='bold', color='#06ffa5')
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Speed ranking
    ax = axes[1, 1]
    sorted_df = df.sort_values('mean_time_ms')
    y_pos = np.arange(len(sorted_df))
    bars = ax.barh(y_pos, sorted_df['mean_time_ms'], alpha=0.8)
    
    # Gradient color for bars
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.plasma(i / len(bars)))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['benchmark'].replace('_', ' ').title()} ({row['size']})" 
                       for _, row in sorted_df.iterrows()], fontsize=10)
    ax.set_xlabel('EXECUTION TIME (ms)', fontweight='bold')
    ax.set_title('🏆 SPEED LEADERBOARD', fontweight='bold', color='#ff006e')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'advanced_performance_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    print(f"🎨 Dark mode visualizations saved to: {output_dir}")
    print("   📊 cyberpunk_performance_dashboard.png")
    print("   🎯 advanced_performance_analysis.png")
    if simd_data:
        print("   🔥 simd_radar_chart.png")

def generate_performance_report(data: List[Dict], output_dir: Path):
    """Generate a comprehensive performance report."""
    report_file = output_dir / 'benchmark_report.md'
    
    if not data:
        with open(report_file, 'w') as f:
            f.write("# No benchmark data found\n")
        return
    
    # Sort by performance
    data_sorted = sorted(data, key=lambda x: x['mean_time_ms'])
    
    with open(report_file, 'w') as f:
        f.write("# RAGnificent Rust Benchmarks Performance Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("## Performance Summary\n\n")
        f.write(f"- Total benchmarks: {len(data)}\n")
        groups = list(set(row['group'] for row in data))
        sizes = list(set(row['size'] for row in data))
        f.write(f"- Benchmark groups: {', '.join(groups)}\n")
        f.write(f"- Test sizes: {', '.join(sizes)}\n\n")
        
        # Performance statistics
        f.write("### Performance Statistics\n\n")
        mean_times = [row['mean_time_ms'] for row in data]
        f.write(f"- Fastest operation: {data_sorted[0]['benchmark']} ({min(mean_times):.3f} ms)\n")
        f.write(f"- Slowest operation: {data_sorted[-1]['benchmark']} ({max(mean_times):.3f} ms)\n")
        f.write(f"- Average execution time: {sum(mean_times)/len(mean_times):.3f} ms\n")
        f.write(f"- Median execution time: {sorted(mean_times)[len(mean_times)//2]:.3f} ms\n\n")
        
        # Group-wise performance
        f.write("## Performance by Group\n\n")
        for group in groups:
            group_data = [row for row in data if row['group'] == group]
            group_times = [row['mean_time_ms'] for row in group_data]
            f.write(f"### {group}\n\n")
            f.write(f"- Operations: {len(group_data)}\n")
            f.write(f"- Average time: {sum(group_times)/len(group_times):.3f} ms\n")
            f.write(f"- Best performance: {min(group_times):.3f} ms\n")
            f.write(f"- Worst performance: {max(group_times):.3f} ms\n\n")
        
        # SIMD optimizations analysis
        simd_data = [row for row in data if 'simd' in row['benchmark'].lower()]
        if simd_data:
            f.write("## SIMD Optimizations\n\n")
            simd_times = [row['mean_time_ms'] for row in simd_data]
            f.write(f"- SIMD operations tested: {len(simd_data)}\n")
            f.write(f"- Average SIMD performance: {sum(simd_times)/len(simd_times):.3f} ms\n")
            f.write("\n### SIMD Operation Details\n\n")
            f.write(" < /dev/null |  Operation | Size | Mean Time (ms) | Std Dev (ms) |\n")
            f.write("|-----------|------|----------------|---------------|\n")
            for row in simd_data:
                f.write(f"| {row['benchmark']} | {row['size']} | "
                       f"{row['mean_time_ms']:.3f} | {row['std_dev_ms']:.3f} |\n")
            f.write("\n")
        
        # Detailed benchmark results
        f.write("## Detailed Results\n\n")
        f.write("| Group | Benchmark | Size | Mean Time (ms) | Std Dev (ms) | CV (%) |\n")
        f.write("|-------|-----------|------|----------------|--------------|--------|\n")
        
        for row in data_sorted:
            f.write(f"| {row['group']} | {row['benchmark']} | {row['size']} | "
                   f"{row['mean_time_ms']:.3f} | {row['std_dev_ms']:.3f} | "
                   f"{row['relative_std']*100:.1f} |\n")

def main():
    """Main function to process benchmarks and generate reports."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    criterion_dir = project_root / "target" / "criterion"
    output_dir = project_root / "benchmark_reports"
    
    print("RAGnificent Benchmark Analysis")
    print("=" * 50)
    print(f"Looking for benchmarks in: {criterion_dir}")
    
    # Load benchmark data
    data = load_benchmark_data(criterion_dir)
    
    if not data:
        print("No benchmark data found. Run 'cargo bench' first.")
        sys.exit(1)
    
    print(f"Found {len(data)} benchmark groups")
    
    # Create summary DataFrame
    summary_data = create_performance_summary(data)
    
    if not summary_data:
        print("No valid benchmark data to process.")
        sys.exit(1)
    
    print(f"Processing {len(summary_data)} benchmark results...")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Generate sexy visualizations
    if PLOTTING_AVAILABLE:
        create_sexy_visualizations(summary_data, output_dir)
    else:
        print("📊 Visualization dependencies not available - install with: pip install matplotlib seaborn pandas numpy")
    
    # Generate report
    generate_performance_report(summary_data, output_dir)
    print(f"Performance report saved to: {output_dir / 'benchmark_report.md'}")
    
    # Display quick summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total benchmarks: {len(summary_data)}")
    
    if summary_data:
        mean_times = [row['mean_time_ms'] for row in summary_data]
        fastest = min(summary_data, key=lambda x: x['mean_time_ms'])
        slowest = max(summary_data, key=lambda x: x['mean_time_ms'])
        
        print(f"Fastest: {fastest['benchmark']} ({fastest['mean_time_ms']:.3f} ms)")
        print(f"Slowest: {slowest['benchmark']} ({slowest['mean_time_ms']:.3f} ms)")
        print(f"Average: {sum(mean_times)/len(mean_times):.3f} ms")
        
        simd_data = [row for row in summary_data if 'simd' in row['benchmark'].lower()]
        if simd_data:
            simd_times = [row['mean_time_ms'] for row in simd_data]
            print(f"SIMD operations: {len(simd_data)} (avg: {sum(simd_times)/len(simd_times):.3f} ms)")

if __name__ == "__main__":
    main()
