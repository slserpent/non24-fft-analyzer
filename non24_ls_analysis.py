#!/usr/bin/env python3
"""
Sleep Cycle Analysis using Lomb-Scargle Periodogram

This script analyzes sleep cycle patterns using the Lomb-Scargle periodogram,
working directly with sleep timestamps rather than converting to binary data.

Input: CSV file with start and end times of sleep events
       - First column is always start time
       - Second column is always end time
       - If first row contains headers, it will be skipped
Output: Analysis results in selected formats (plot, CSV, and/or JSON)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import lombscargle, find_peaks
import argparse
import os
import csv
import json
import sys

# custom output class (monkey patch) for JSON so it doesn't print small floats in scientific notation
class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format_float_export(x))

json.encoder.c_make_encoder = None  # Force Python encoder over C encoder
json.encoder.float = RoundingFloat  # Use our custom float representation

# formats the floating point number to six numbers after the decimal point and then removes trailing zeros
def format_float_export(in_float):
    return format(in_float, '.6f').rstrip('0').rstrip('.')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sleep Cycle Analysis')
    parser.add_argument('input_file', nargs='?', help='Input CSV file with sleep data')
    parser.add_argument('--endpoint', choices=['mid', 'start', 'end'], default='mid',
                        help='Which position in sleep blocks to analyze (default: mid)')
    parser.add_argument('--min_peak_distance', type=float, default=0.5,
                        help='Minimum distance between peaks in hours (default: 0.5)')
    parser.add_argument('--range_start', type=str,
                        help='Start date for analysis (optional)')
    parser.add_argument('--range_end', type=str,
                        help='End date for analysis (optional)')
    parser.add_argument('--output_dir', default='lomb_scargle_analysis', 
                        help='Directory to save results')
    parser.add_argument('-p', '--plot', action='store_true', dest='save_plot',
                        help='Plot results on a PNG graph.')
    parser.add_argument('-c', '--csv', action='store_true',
                        help='Save results as CSV')
    parser.add_argument('-j', '--json', action='store_true',
                        help='Save results as JSON')
    parser.add_argument('-f', '--write_full_data', action='store_true',
                        help='Save full periodogram data. Requires CSV or JSON output enabled')
    
    args = parser.parse_args()
    
    # Validate min_peak_distance (0-12 hours)
    if args.min_peak_distance > 12 or args.min_peak_distance < 0:
        if args.min_peak_distance > 12:
            args.min_peak_distance = 12
        elif args.min_peak_distance < 0:
            args.min_peak_distance = 0
        print(f"Warning: Minimum peak distance was beyond limits. Set to limit of {args.min_peak_distance}.")
    
    if args.write_full_data and not (args.csv or args.json):
        parser.error("--write_full_data (-f) requires either --csv (-c) or --json (-j)")
    
    # Show help if no input file specified
    if not args.input_file:
        parser.print_help()
        exit(1)
    
    return args
    
def detect_header_row(file_path):
    """
    Detect if the first row is a header or data.
    """
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            row = next(csv.reader([first_line]))
            
            try:
                pd.to_datetime(row[0])
                pd.to_datetime(row[1])
                return False  # Don't skip - it's data
            except:
                return True  # Skip - it's a header
    except:
        return False

def load_sleep_data(file_path, range_start=None, range_end=None):
    """
    Load sleep data and calculate midpoints.
    
    Args:
        file_path: Path to CSV file with sleep data
        range_start: Optional start date for analysis
        range_end: Optional end date for analysis
    
    Returns:
        DataFrame with sleep data including calculated midpoints
    """
    print(f"Loading sleep data from {file_path}...")
    
    skip_header = detect_header_row(file_path)
    if skip_header:
        print("First row appears to be a header - skipping it")
    else:
        print("First row appears to be data - including it")
    
    df = pd.read_csv(file_path, header=None, skiprows=1 if skip_header else 0)
    
    if df.shape[1] < 2:
        raise ValueError("CSV file must have at least 2 columns: start time and end time")
    
    df = df.iloc[:, 0:2]
    df.columns = ['start', 'end']
    
    try:
        df['start'] = pd.to_datetime(df['start'], errors='coerce')
        df['end'] = pd.to_datetime(df['end'], errors='coerce')
    except Exception as e:
        print(f"Warning during datetime parsing: {e}")
        print("Attempting to continue with successfully parsed dates...")
    
    df = df.dropna()
    
    # Calculate midpoints
    df['mid'] = df['start'] + (df['end'] - df['start']) / 2
    
    # Parse and apply date range filtering if specified
    if range_start:
        try:
            range_start = pd.to_datetime(range_start)
        except:
            print(f"Error: Could not parse start date: {range_start}")
            exit(1)
        df = df[df['start'] >= range_start]
        
    if range_end:
        try:
            range_end = pd.to_datetime(range_end)
        except:
            print(f"Error: Could not parse end date: {range_end}")
            exit(1)
        df = df[df['end'] <= range_end]
    
    # Sort by midpoint time
    df = df.sort_values('mid').reset_index(drop=True)
    
    if len(df) == 0:
        print("Error: No sleep data found in specified date range")
        exit(1)
    
    date_range = f"from {df['start'].min()} to {df['end'].max()}"
    if range_start or range_end:
        date_range += " (filtered)"
    
    print(f"Loaded {len(df)} sleep events {date_range}")
    return df

def analyze_sleep_cycles(df, endpoint, min_peak_distance):
    """
    Analyze sleep cycles using Lomb-Scargle periodogram on specified timestamps.
    """
    print(f"Analyzing sleep cycles using {endpoint} times...")
    
    # Convert timestamps to hours since first event
    reference_time = df[endpoint].min()
    times_hours = [(t - reference_time).total_seconds() / 3600 for t in df[endpoint]]
    times_hours = np.array(times_hours)
    
    # Create signal array (1.0 for each event)
    signal = np.ones_like(times_hours)
    
    # Define frequencies to analyze (focusing on 12-48 hour range)
    min_period = 12  # hours
    max_period = 48  # hours
    frequencies = 2 * np.pi / np.linspace(min_period, max_period, 2000)
    
    print(f"Calculating Lomb-Scargle periodogram...")
    pgram = lombscargle(times_hours, signal, frequencies)
    
    # Convert back to periods in hours and normalize power
    periods = 2 * np.pi / frequencies
    pgram = pgram / np.sum(pgram) * 100  # Convert to percentage of total power
    
    # Sort by period for proper plotting
    sorted_indices = np.argsort(periods)
    periods = periods[sorted_indices]
    pgram = pgram[sorted_indices]
    
    # Convert min_peak_distance from hours to number of samples
    samples_per_hour = len(periods) / (periods[-1] - periods[0])
    min_samples = max(1, int(min_peak_distance * samples_per_hour))
    
    # Find peaks using scipy.signals
    peak_indices, _ = find_peaks(pgram, distance=min_samples, height=0.25*np.max(pgram))
    
    # Get top 5 peaks by height
    peak_heights = pgram[peak_indices]
    top_5_indices = np.argsort(peak_heights)[-5:][::-1]
    peak_indices = peak_indices[top_5_indices]
    
    # Calculate detailed stats for each peak
    peak_stats = [calculate_peak_stats(periods, pgram, idx) for idx in peak_indices]
    
    # Initialize harmonic fields in peak_stats
    for stats in peak_stats:
        stats['is_harmonic'] = False
        stats['harmonic_of'] = None
        stats['harmonic_fraction'] = None
    
    # Detect harmonic relationships
    harmonics_present = False
    num_peaks = len(peak_stats)
    
    for i in range(num_peaks):
        for j in range(num_peaks):
            if i == j:
                continue
                
            # Only label a peak as harmonic if its power is lower than the fundamental peak
            if peak_stats[i]['peak_power'] <= peak_stats[j]['peak_power']:
                ratio = peak_stats[i]['period'] / peak_stats[j]['period']
                
                # Check common harmonic relationships with 2% tolerance
                for fraction in [0.5, 0.33, 0.25, 2, 3, 4]:
                    if abs(ratio - fraction) < 0.02:  # 2% tolerance
                        peak_stats[i]['is_harmonic'] = True
                        peak_stats[i]['harmonic_of'] = peak_stats[j]['period']
                        peak_stats[i]['harmonic_fraction'] = fraction
                        harmonics_present = True
                        break
    
    return periods, pgram, peak_stats, harmonics_present
	
def calculate_peak_stats(periods, power, peak_idx):
    """
    Calculate detailed statistics for a peak including cluster power and FWHM.
    
    Args:
        periods: Array of all period values
        power: Array of all power values
        peak_idx: Index of the peak to analyze
    
    Returns:
        Dictionary containing peak statistics
    """
    peak_period = periods[peak_idx]
    peak_power = power[peak_idx]
    
    # Calculate noise floor
    noise_floor = np.mean(power[power < np.max(power) * 0.25])
    power_threshold = noise_floor + (np.max(power) * 0.05)  # 5% above noise floor
    min_samples = int(len(periods) * 0.005)  # 0.5% of total samples
    
    # Find cluster extents with lookahead
    def find_boundary(start_idx, step):
        """Find cluster boundary in one direction using lookahead."""
        idx = start_idx
        while 0 <= idx < len(power):
            if power[idx] < power_threshold:
                # Look ahead to check if power stays low
                lookahead_count = 0
                check_idx = idx
                while (0 <= check_idx < len(power) and 
                       check_idx * step <= idx * step + min_samples):
                    if power[check_idx] < power_threshold:
                        lookahead_count += 1
                    check_idx += step
                
                if lookahead_count >= min_samples:
                    # Found sustained drop - return last good index
                    return idx - step
            
            idx += step
        return idx - step  # Hit array boundary
    
    # Find left and right boundaries
    left = find_boundary(peak_idx, -1)
    right = find_boundary(peak_idx, 1)
    
    # Calculate cluster power
    cluster_power = np.sum(power[left:right+1])
    cluster_range = (periods[left], periods[right])
    
    # Calculate FWHM
    half_max = peak_power / 2
    left_width = peak_idx
    right_width = peak_idx
    
    while left_width > 0 and power[left_width] > half_max:
        left_width -= 1
    while right_width < len(power) - 1 and power[right_width] > half_max:
        right_width += 1
    
    fwhm = periods[right_width] - periods[left_width]
    
    # Calculate confidence (false alarm probability or "FAP" lol)
    N = len(periods)
    M = -6.362 + 1.193*N + 0.00098*N**2  # Horne & Baliunas approximation
    z = peak_power / noise_floor  # Compare to noise floor instead of max power
    fap = 1 - (1 - np.exp(-z))**M
    confidence = min((1 - fap) * 100, 99.99)  # Cap at 99.99%
    
    return {
        'period': peak_period,
        'peak_power': peak_power,
        'cluster_power': cluster_power,
        'cluster_range': cluster_range,
        'fwhm': fwhm,
        'confidence': confidence
    }

def plot_results(periods, power, peak_stats, output_dir, endpoint, harmonics_present):
    """Plot analysis results in the 12-48 hour range."""
    plt.figure(figsize=(12, 8))
    plt.plot(periods, power, linewidth=1.5)
    
    # Set fixed tick positions and labels
    hours_ticks = [12, 18, 20, 22, 24, 26, 28, 30, 36, 42, 48]
    plt.xticks(hours_ticks, [f"{h}h" for h in hours_ticks])
    
    # Color the 24h tick label red
    tick_labels = plt.gca().get_xticklabels()
    tick_positions = plt.gca().get_xticks()
    for i, pos in enumerate(tick_positions):
        if pos == 24:
            tick_labels[i].set_color('red')
    
    # Add markers for peaks
    for i, stats in enumerate(peak_stats):
        plt.plot(stats['period'], stats['peak_power'], 'ro', markersize=8)
        
        # Create offset for label to avoid overlap
        y_offset = 5 + (i % 3) * 10  # Cycle through 3 different heights
        x_offset = 5 if i % 2 == 0 else -30  # Alternate left and right
        
        # Add asterisk if peak is a harmonic
        harmonic_marker = "*" if stats['is_harmonic'] else ""
        
        plt.annotate(f"{stats['period']:.1f}h{harmonic_marker}",
                     xy=(stats['period'], stats['peak_power']),
                     xytext=(x_offset, y_offset),
                     textcoords='offset points',
                     fontsize=9,
                     arrowprops=dict(arrowstyle="->", color='black', lw=1))
    
    plt.xlabel('Period (hours)', fontsize=12)
    plt.ylabel('Power (% of total)', fontsize=12)
    plt.title(f'Lomb-Scargle Sleep Cycle Analysis ({endpoint})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create legend
    legend_handles = [
        plt.Line2D([0], [0], color='white', marker='o', markerfacecolor='red', 
                   markersize=8, label='Peak period')
    ]
    
    # Only add harmonic legend if harmonics are present
    if harmonics_present:
        legend_handles.append(
            plt.Line2D([0], [0], color='white', marker='o', markerfacecolor='red',
                      markersize=8, label='* Harmonic')
        )
    
    plt.legend(handles=legend_handles)
    
    plt.savefig(os.path.join(output_dir, 'sleep_cycle_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_results(peak_stats, harmonics_present):
    """Print analysis results to stdout."""
    print("\nDominant periods detected:")
    for i, stats in enumerate(peak_stats):
        print(f"{i+1}. Period: {stats['period']:.2f} hours")
        print(f"   Peak Power: {stats['peak_power']:.2f}% of total")
        print(f"   Cluster Power: {stats['cluster_power']:.2f}%")
        print(f"   Cluster Width: {stats['cluster_range'][1] - stats['cluster_range'][0]:.2f} hours "
              f"({stats['cluster_range'][0]:.1f}h - {stats['cluster_range'][1]:.1f}h)")
        print(f"   Full Width at Half Maximum: {stats['fwhm']:.2f} hours")
        print(f"   Confidence: {stats['confidence']:.2f}%")
        
        # Only display "Is Harmonic" if any harmonics were detected in the analysis
        if harmonics_present:
            print(f"   Is Harmonic: {'Yes' if stats['is_harmonic'] else 'No'}")
            
            # Only show "Harmonic of" for peaks that are actually harmonics
            if stats['is_harmonic']:
                print(f"   Harmonic of: {stats['harmonic_of']:.2f} hours ({stats['harmonic_fraction']:.2f}x)")

def save_results(peak_stats, periods, power, output_dir, save_csv=False, save_json=False, write_full_data=False):
    """Save analysis results in requested formats."""
    if save_csv:
        # Save peak statistics
        results_df = pd.DataFrame([{
            'period_hours': format_float_export(stats['period']),
            'peak_power_percent': format_float_export(stats['peak_power']),
            'cluster_power_percent': format_float_export(stats['cluster_power']),
            'cluster_start_hours': format_float_export(stats['cluster_range'][0]),
            'cluster_end_hours': format_float_export(stats['cluster_range'][1]),
            'fwhm_hours': format_float_export(stats['fwhm']),
            'confidence_percent': format_float_export(stats['confidence']),
            'is_harmonic': 1 if stats['is_harmonic'] else 0,
            'harmonic_of': format_float_export(stats['harmonic_of']) if stats['is_harmonic'] else '',
            'harmonic_fraction': format_float_export(stats['harmonic_fraction']) if stats['is_harmonic'] else ''
        } for stats in peak_stats])
        
        results_df.to_csv(os.path.join(output_dir, 'dominant_periods.csv'), index=False)
        
        if write_full_data:
            full_df = pd.DataFrame({
                'period_hours': [format(p, 'f') for p in periods],
                'power_percent': [format(p, 'f') for p in power]
            })
            full_df.to_csv(os.path.join(output_dir, 'full_periodogram.csv'), index=False)
    
    if save_json:
        # Save peak statistics
        results_json = {
            'dominant_periods': [{
                'period_hours': stats['period'],
                'peak_power_percent': stats['peak_power'],
                'cluster_power_percent': stats['cluster_power'],
                'cluster_range': {
                    'start_hours': stats['cluster_range'][0],
                    'end_hours': stats['cluster_range'][1]
                },
                'fwhm_hours': stats['fwhm'],
                'confidence_percent': stats['confidence'],
                'is_harmonic': stats['is_harmonic'],
                'harmonic_of': stats['harmonic_of'] if stats['is_harmonic'] else None,
                'harmonic_fraction': stats['harmonic_fraction'] if stats['is_harmonic'] else None
            } for stats in peak_stats]
        }
        
        with open(os.path.join(output_dir, 'dominant_periods.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        if write_full_data:
            full_json = {
                'periodogram': [{
                    'period_hours': period,
                    'power_percent': power_val
                } for period, power_val in zip(periods, power)]
            }
            
            with open(os.path.join(output_dir, 'full_periodogram.json'), 'w') as f:
                json.dump(full_json, f, indent=2)

def main():
    """Main function to run analysis."""
    if sys.version_info[0] < 3:
        print("Error: This script requires Python 3")
        exit(1)
    
    args = parse_arguments()
    
    try:
        # Load and analyze data
        df = load_sleep_data(args.input_file, args.range_start, args.range_end)
        periods, power, peak_stats, harmonics_present = analyze_sleep_cycles(
            df, args.endpoint, args.min_peak_distance)
        
        # Always print results to stdout
        print_results(peak_stats, harmonics_present)
        
        # Generate file outputs if requested
        will_write_files = args.save_plot or args.csv or args.json
        if will_write_files:
            os.makedirs(args.output_dir, exist_ok=True)
            
            if args.save_plot:
                plot_results(periods, power, peak_stats, 
                           args.output_dir, args.endpoint, harmonics_present)
            
            if args.csv or args.json:
                save_results(peak_stats, periods, power, args.output_dir,
                           save_csv=args.csv, save_json=args.json,
                           write_full_data=args.write_full_data)
            
            print(f"\nResults saved to {args.output_dir}/")
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())