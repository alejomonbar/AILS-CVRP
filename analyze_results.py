#!/usr/bin/env python3
"""
Analyze and visualize AILS-II experimental results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

def load_run(run_dir):
    """Load all data from a single run"""
    iterations = pd.read_csv(f"{run_dir}/iterations.csv")
    summary = pd.read_csv(f"{run_dir}/summary.csv")
    
    with open(f"{run_dir}/config.txt", 'r') as f:
        config = f.read()
    
    return iterations, summary, config

def plot_convergence(iterations, run_id, output_dir):
    """Plot solution quality convergence over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gap vs Iteration
    ax1.plot(iterations['iteration'], iterations['gap_percent'], linewidth=1.5)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Gap (%)', fontsize=12)
    ax1.set_title(f'Convergence - {run_id}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Gap vs Time
    ax2.plot(iterations['time_sec'], iterations['gap_percent'], linewidth=1.5)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Gap (%)', fontsize=12)
    ax2.set_title(f'Gap vs Time - {run_id}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/convergence.png")
    plt.close()

def plot_parameters(iterations, run_id, output_dir):
    """Plot adaptive parameter evolution"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Omega evolution
    ax1.plot(iterations['iteration'], iterations['omega'], linewidth=1, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Omega (perturbation strength)')
    ax1.set_title('Omega Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Eta evolution
    ax2.plot(iterations['iteration'], iterations['eta'], linewidth=1, alpha=0.7, color='orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Eta (acceptance threshold)')
    ax2.set_title('Eta Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Distance LS
    ax3.plot(iterations['iteration'], iterations['distance_ls'], linewidth=1, alpha=0.7, color='green')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Distance after Local Search')
    ax3.set_title('Solution Distance')
    ax3.grid(True, alpha=0.3)
    
    # Number of routes
    ax4.plot(iterations['iteration'], iterations['num_routes'], linewidth=1, alpha=0.7, color='red')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Number of Routes')
    ax4.set_title('Routes Evolution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameters.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/parameters.png")
    plt.close()

def plot_operator_performance(iterations, run_id, output_dir):
    """Analyze perturbation operator and insertion heuristic performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Operator selection frequency
    pert_counts = iterations['perturbation_type'].value_counts()
    ax1.bar(pert_counts.index, pert_counts.values, alpha=0.7)
    ax1.set_xlabel('Perturbation Operator')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Operator Selection Frequency')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Insertion heuristic frequency
    ins_counts = iterations['insertion_heuristic'].value_counts()
    ax2.bar(ins_counts.index, ins_counts.values, alpha=0.7, color='orange')
    ax2.set_xlabel('Insertion Heuristic')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Insertion Heuristic Frequency')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/operators.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/operators.png")
    plt.close()

def plot_improvement_distribution(iterations, run_id, output_dir):
    """Plot distribution of improvements"""
    # Only consider improving iterations
    improving = iterations[iterations['improvement'] > 0]
    
    if len(improving) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of improvements
        ax1.hist(improving['improvement'], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Improvement')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Improvements ({len(improving)} improving iterations)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative improvement over time
        ax2.plot(improving['iteration'], improving['improvement'].cumsum(), linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative Improvement')
        ax2.set_title('Cumulative Solution Improvement')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/improvements.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/improvements.png")
        plt.close()

def generate_report(iterations, summary, config, output_dir):
    """Generate text report"""
    report = []
    report.append("=" * 60)
    report.append("AILS-II RUN ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY:")
    report.append(f"  Instance: {summary['instance'].iloc[0]}")
    report.append(f"  Seed: {summary['seed'].iloc[0]}")
    report.append(f"  Run ID: {summary['run_id'].iloc[0]}")
    report.append(f"  Total Iterations: {summary['total_iterations'].iloc[0]}")
    report.append(f"  Total Time: {summary['total_time'].iloc[0]:.2f} seconds")
    report.append("")
    
    # Performance metrics
    report.append("PERFORMANCE:")
    report.append(f"  Best Quality: {summary['best_quality'].iloc[0]:.2f}")
    report.append(f"  Best Gap: {summary['best_gap'].iloc[0]:.4f}%")
    report.append(f"  Final Quality: {summary['final_quality'].iloc[0]:.2f}")
    report.append(f"  Final Gap: {summary['final_gap'].iloc[0]:.4f}%")
    report.append(f"  Time to Best: {summary['time_to_best'].iloc[0]:.2f}s (iteration {summary['iter_to_best'].iloc[0]})")
    report.append("")
    
    # Iteration statistics
    report.append("ITERATION STATISTICS:")
    report.append(f"  Avg time per iteration: {iterations['time_sec'].diff().mean():.4f}s")
    report.append(f"  Improvements found: {(iterations['improvement'] > 0).sum()}")
    report.append(f"  Improvement rate: {100 * (iterations['improvement'] > 0).sum() / len(iterations):.2f}%")
    report.append("")
    
    # Operator usage
    report.append("OPERATOR USAGE:")
    for op, count in iterations['perturbation_type'].value_counts().items():
        pct = 100 * count / len(iterations)
        report.append(f"  {op}: {count} ({pct:.1f}%)")
    report.append("")
    
    # Insertion heuristic usage
    report.append("INSERTION HEURISTIC USAGE:")
    for ih, count in iterations['insertion_heuristic'].value_counts().items():
        pct = 100 * count / len(iterations)
        report.append(f"  {ih}: {count} ({pct:.1f}%)")
    report.append("")
    
    # Parameter ranges
    report.append("PARAMETER RANGES:")
    report.append(f"  Omega: [{iterations['omega'].min():.2f}, {iterations['omega'].max():.2f}] (avg: {iterations['omega'].mean():.2f})")
    report.append(f"  Eta: [{iterations['eta'].min():.4f}, {iterations['eta'].max():.4f}] (avg: {iterations['eta'].mean():.4f})")
    report.append(f"  Routes: [{iterations['num_routes'].min()}, {iterations['num_routes'].max()}] (avg: {iterations['num_routes'].mean():.1f})")
    report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    # Save to file
    with open(f"{output_dir}/report.txt", 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSaved: {output_dir}/report.txt")

def analyze_run(run_dir):
    """Analyze a single run"""
    print(f"\nAnalyzing: {run_dir}")
    
    # Load data
    iterations, summary, config = load_run(run_dir)
    run_id = summary['run_id'].iloc[0]
    
    # Generate visualizations
    plot_convergence(iterations, run_id, run_dir)
    plot_parameters(iterations, run_id, run_dir)
    plot_operator_performance(iterations, run_id, run_dir)
    plot_improvement_distribution(iterations, run_id, run_dir)
    
    # Generate report
    generate_report(iterations, summary, config, run_dir)
    
    return iterations, summary

def compare_runs(instance_dir):
    """Compare multiple runs for the same instance"""
    print(f"\nComparing runs in: {instance_dir}")
    
    run_dirs = [d for d in Path(instance_dir).iterdir() if d.is_dir()]
    
    if len(run_dirs) == 0:
        print("No runs found!")
        return
    
    all_summaries = []
    for run_dir in sorted(run_dirs):
        try:
            _, summary, _ = load_run(str(run_dir))
            all_summaries.append(summary)
        except:
            continue
    
    if len(all_summaries) == 0:
        print("No valid runs found!")
        return
    
    # Combine all summaries
    combined = pd.concat(all_summaries, ignore_index=True)
    
    # Statistical summary
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS RUNS")
    print("=" * 60)
    print(f"\nNumber of runs: {len(combined)}")
    print(f"\nBest Gap Statistics:")
    print(f"  Mean: {combined['best_gap'].mean():.4f}%")
    print(f"  Std:  {combined['best_gap'].std():.4f}%")
    print(f"  Min:  {combined['best_gap'].min():.4f}%")
    print(f"  Max:  {combined['best_gap'].max():.4f}%")
    
    # Save comparison
    combined.to_csv(f"{instance_dir}/comparison.csv", index=False)
    print(f"\nSaved comparison to: {instance_dir}/comparison.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <run_dir_or_instance_dir>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.exists(f"{path}/iterations.csv"):
        # Single run analysis
        analyze_run(path)
    else:
        # Multiple runs comparison
        compare_runs(path)
