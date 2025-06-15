#\!/usr/bin/env python3
"""
Benchmark tracking system for RAGnificent.
Tracks performance over time and maintains historical data.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

def save_benchmark_results(benchmark_data: List[Dict], results_dir: Path):
    """Save benchmark results with timestamp for historical tracking."""
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    
    # Save current results
    result_data = {
        'timestamp': timestamp,
        'date': datetime.now().isoformat(),
        'results': benchmark_data,
        'summary': {
            'total_benchmarks': len(benchmark_data),
            'fastest_time': min(r['mean_time_ms'] for r in benchmark_data),
            'slowest_time': max(r['mean_time_ms'] for r in benchmark_data),
            'average_time': sum(r['mean_time_ms'] for r in benchmark_data) / len(benchmark_data)
        }
    }
    
    with open(results_dir / filename, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"📊 Benchmark results saved: {filename}")

def load_historical_results(results_dir: Path) -> List[Dict]:
    """Load all historical benchmark results."""
    if not results_dir.exists():
        return []
    
    historical_data = []
    for file in results_dir.glob("benchmark_results_*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
                historical_data.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return sorted(historical_data, key=lambda x: x['timestamp'])

def create_performance_history_report(results_dir: Path, output_dir: Path):
    """Create a report showing performance trends over time."""
    historical_data = load_historical_results(results_dir)
    
    if not historical_data:
        print("No historical benchmark data found.")
        return
    
    report_file = output_dir / 'performance_history.md'
    
    with open(report_file, 'w') as f:
        f.write("# RAGnificent Performance History\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Trends\n\n")
        f.write(f"- Total benchmark runs: {len(historical_data)}\n")
        f.write(f"- First run: {historical_data[0]['date']}\n")
        f.write(f"- Latest run: {historical_data[-1]['date']}\n\n")
        
        f.write("## Historical Performance Data\n\n")
        f.write(" < /dev/null |  Date | Benchmarks | Fastest (ms) | Slowest (ms) | Average (ms) |\n")
        f.write("|------|------------|--------------|--------------|---------------|\n")
        
        for data in historical_data:
            summary = data['summary']
            date = datetime.fromisoformat(data['date']).strftime('%Y-%m-%d %H:%M')
            f.write(f"| {date} | {summary['total_benchmarks']} | "
                   f"{summary['fastest_time']:.3f} | {summary['slowest_time']:.3f} | "
                   f"{summary['average_time']:.3f} |\n")
        
        # Performance improvement analysis
        if len(historical_data) > 1:
            first = historical_data[0]['summary']
            latest = historical_data[-1]['summary']
            
            f.write("\n## Performance Analysis\n\n")
            fastest_improvement = ((first['fastest_time'] - latest['fastest_time']) / first['fastest_time']) * 100
            average_improvement = ((first['average_time'] - latest['average_time']) / first['average_time']) * 100
            
            f.write(f"- Fastest operation improvement: {fastest_improvement:.1f}%\n")
            f.write(f"- Average performance improvement: {average_improvement:.1f}%\n")
    
    print(f"📈 Performance history report saved: {report_file}")

def main():
    """Main function for benchmark tracking."""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "benchmark_history"
    output_dir = project_root / "benchmark_reports"
    
    # Import from visualize_benchmarks.py
    import sys
    sys.path.append(str(project_root / "scripts"))
    
    try:
        from visualize_benchmarks import load_benchmark_data, create_performance_summary
        
        # Load current benchmark data
        criterion_dir = project_root / "target" / "criterion"
        data = load_benchmark_data(criterion_dir)
        
        if data:
            summary_data = create_performance_summary(data)
            save_benchmark_results(summary_data, results_dir)
            create_performance_history_report(results_dir, output_dir)
        else:
            print("No benchmark data found to track.")
    
    except ImportError as e:
        print(f"Error importing benchmark functions: {e}")

if __name__ == "__main__":
    main()
