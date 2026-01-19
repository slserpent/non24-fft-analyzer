#!/usr/bin/env python3
"""
Enhanced FFT Sleep Analysis

This script performs Fast Fourier Transform (FFT) analysis on sleep data
to detect periodic patterns in Non-24 Hour Sleep-Wake Disorder.

Input: CSV file with start and end times of sleep events
       - First column is always start time
       - Second column is always end time
       - If first row contains headers, it will be skipped
       - Otherwise, all rows are treated as data
Output: FFT analysis results and visualizations of periodicity in multiple formats
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
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

def format_float_export(in_float):
    """Format float to 6 decimal places and strip trailing zeros."""
    return format(in_float, '.6f').rstrip('0').rstrip('.')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced FFT Sleep Analysis')
    parser.add_argument('input_file', nargs='?', help='Input CSV file with sleep data')
    parser.add_argument('--sampling_interval', type=int, default=15,
                        help='Sampling interval for time series in minutes (default 15)')
    parser.add_argument('--min_peak_distance', type=float, default=0.5,
                        help='Minimum distance between peaks in hours (default: 0.5)')
    parser.add_argument('--range_start', type=str,
                        help='Start date for analysis (optional)')
    parser.add_argument('--range_end', type=str,
                        help='End date for analysis (optional)')
    parser.add_argument('--output_dir', default='fft_analysis', 
                        help='Directory to save results')
    parser.add_argument('-p', '--plot', action='store_true', dest='save_plot',
                        help='Plot results on a PNG graph.')
    parser.add_argument('-c', '--csv', action='store_true',
                        help='Save results as CSV')
    parser.add_argument('-j', '--json', action='store_true',
                        help='Save results as JSON')
    parser.add_argument('-f', '--write_full_data', action='store_true',
                        help='Save full FFT data. Requires CSV or JSON output enabled')
    
    args = parser.parse_args()
    
    # Validate sampling_interval (1-360 minutes)
    if args.sampling_interval > 360 or args.sampling_interval < 1:
        if args.sampling_interval > 360:
            args.sampling_interval = 360
        elif args.sampling_interval < 1:
            args.sampling_interval = 1
        print(f"Warning: Sampling interval was beyond limits. Set to limit of {args.sampling_interval}.")
    
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
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Boolean indicating if the first row should be skipped
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
    Load and preprocess sleep data from CSV file.
    Skips the first row only if it's detected as a header.
    
    Args:
        file_path: Path to CSV file with sleep times
        range_start: Optional start date for analysis
        range_end: Optional end date for analysis
        
    Returns:
        DataFrame with sleep data
    """
    print(f"Loading sleep data from {file_path}...")
    
    skip_header = detect_header_row(file_path)
    if skip_header:
        print("First row appears to be a header - skipping it")
    else:
        print("First row appears to be data - including it")
    
    df = pd.read_csv(file_path, header=None, skiprows=1 if skip_header else 0)
    
    if df.shape[1] >= 2:
        df = df.iloc[:, 0:2]
    else:
        raise ValueError("CSV file must have at least 2 columns: start time and end time")
    
    df.columns = ['start', 'end']
    
    try:
        df['start'] = pd.to_datetime(df['start'], errors='coerce')
        df['end'] = pd.to_datetime(df['end'], errors='coerce')
    except Exception as e:
        print(f"Warning during datetime parsing: {e}")
        print("Attempting to continue with successfully parsed dates...")
    
    df = df.dropna()
    
    # Apply date range filtering if specified
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
    
    df = df.sort_values('start').reset_index(drop=True)
    
    if len(df) == 0:
        print("Error: No sleep data found in specified date range")
        exit(1)
    
    date_range = f"from {df['start'].min()} to {df['end'].max()}"
    if range_start or range_end:
        date_range += " (filtered)"
    
    print(f"Loaded {len(df)} sleep events {date_range}")
    return df

def generate_binary_time_series(df, sampling_interval=15):
    """
    Generate binary time series where 1 = sleep and 0 = wake.
    
    Args:
        df: DataFrame with sleep data
        sampling_interval: Interval in minutes for time series
        
    Returns:
        Dictionary with time series data and metadata
    """
    print(f"Generating binary time series with {sampling_interval}-minute intervals...")
    
    start_date = df['start'].min()
    end_date = df['end'].max()
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    total_samples = total_minutes // sampling_interval + 1
    
    time_series = np.zeros(total_samples)
    timestamps = [start_date + timedelta(minutes=i*sampling_interval) for i in range(total_samples)]
    
    for _, event in df.iterrows():
        start_idx = int((event['start'] - start_date).total_seconds() / (60 * sampling_interval))
        end_idx = int((event['end'] - start_date).total_seconds() / (60 * sampling_interval))
        
        start_idx = max(0, start_idx)
        end_idx = min(total_samples - 1, end_idx)
        
        time_series[start_idx:end_idx+1] = 1
    
    sleep_samples = np.sum(time_series)
    wake_samples = total_samples - sleep_samples
    
    print(f"Time series spans from {start_date} to {end_date}")
    print(f"Total samples: {total_samples}")
    print(f"Sleep samples: {int(sleep_samples)} ({sleep_samples/total_samples*100:.2f}%)")
    print(f"Wake samples: {int(wake_samples)} ({wake_samples/total_samples*100:.2f}%)")
    
    return {
        'time_series': time_series,
        'timestamps': timestamps,
        'sampling_interval': sampling_interval,
        'start_date': start_date,
        'end_date': end_date,
        'total_samples': total_samples
    }

def perform_fft_analysis(time_series_data, min_peak_distance=0.5):
    """Perform Fast Fourier Transform analysis on sleep-wake time series."""
    print("Performing FFT analysis...")
    
    time_series = time_series_data['time_series']
    sampling_interval = time_series_data['sampling_interval']
    sampling_interval_hours = sampling_interval / 60
    
    detrended_series = signal.detrend(time_series)
    window = signal.windows.hann(len(detrended_series))
    windowed_series = detrended_series * window
    
    fft_result = fft(windowed_series)
    
    n = len(time_series)
	    # Perform FFT frequency analysis
    # Get all frequencies from the FFT
    all_fft_frequencies = fftfreq(n, sampling_interval_hours)
    
    # Get the positive frequencies only (first half of the array)
    # The second half contains redundant negative frequencies for real-valued signals
    positive_freq_count = n // 2  # Integer division to get half the length
    positive_frequencies = all_fft_frequencies[0:positive_freq_count]
    
    # Calculate the amplitude for each frequency
    all_fft_amplitudes = np.abs(fft_result)
    positive_amplitudes = all_fft_amplitudes[0:positive_freq_count] / n
    
    # Convert frequencies to periods, skipping the DC component (zero frequency)
    # DC component is at index 0, so we start from index 1
    nonzero_frequencies = positive_frequencies[1:positive_freq_count]
    periods = np.divide(1.0, nonzero_frequencies)  # Convert frequency to period
    amplitudes = positive_amplitudes[1:positive_freq_count]
    
    # Sort periods in ascending order and reorder amplitudes to match
    sort_indices = np.argsort(periods)
    periods = periods[sort_indices]
    amplitudes = amplitudes[sort_indices]
    
    # Filter for periods between 12-48 hours
    mask = (periods >= 12) & (periods <= 48)
    filtered_periods = periods[mask]
    filtered_amplitudes = amplitudes[mask]
    
    # Calculate spectral entropy
    prob = filtered_amplitudes / np.sum(filtered_amplitudes)
    entropy = -np.sum(prob * np.log2(prob + 1e-10))
    max_entropy = np.log2(len(filtered_amplitudes))
    spectral_entropy = entropy / max_entropy
    
    # Convert min_peak_distance from hours to number of samples
    samples_per_hour = len(filtered_periods) / (48 - 12)
    min_samples = max(1, int(min_peak_distance * samples_per_hour))
    
    # Find peaks
    peak_indices, _ = find_peaks(filtered_amplitudes, distance=min_samples, height=0.25*np.max(filtered_amplitudes))
    
    # Get top 5 peaks by height
    peak_heights = filtered_amplitudes[peak_indices]
    top_5_indices = np.argsort(peak_heights)[-5:][::-1]
    peak_indices = peak_indices[top_5_indices]
    
    # Calculate noise floor and threshold
    max_amp = np.max(filtered_amplitudes)
    noise_mask = filtered_amplitudes <= (max_amp * 0.25)
    noise_floor = np.mean(filtered_amplitudes[noise_mask])
    amplitude_threshold = noise_floor + (max_amp * 0.05)
    
    peak_stats = [calculate_peak_stats(filtered_periods, filtered_amplitudes, idx, noise_floor) 
                 for idx in peak_indices]
    
    # Detect harmonic relationships
    harmonics_present = False
    for i in range(len(peak_stats)):
        for j in range(len(peak_stats)):
            if i == j:
                continue
            
            if peak_stats[i]['amplitude'] <= peak_stats[j]['amplitude']:
                ratio = peak_stats[i]['period'] / peak_stats[j]['period']
                
                for fraction in [0.5, 0.33, 0.25, 2, 3, 4]:
                    if abs(ratio - fraction) < 0.02:
                        peak_stats[i]['is_harmonic'] = True
                        peak_stats[i]['harmonic_of'] = peak_stats[j]['period']
                        peak_stats[i]['harmonic_fraction'] = fraction
                        harmonics_present = True
                        break
    
    return {
        'periods': filtered_periods,
        'amplitudes': filtered_amplitudes,
        'peak_stats': peak_stats,
        'harmonics_present': harmonics_present,
        'noise_floor': noise_floor,
        'amplitude_threshold': amplitude_threshold,
        'spectral_entropy': spectral_entropy
    }

def calculate_peak_stats(periods, amplitudes, idx, noise_floor):
    """Calculate detailed statistics for a peak including FWHM, SNR, and spectral density."""
    peak_period = periods[idx]
    peak_amplitude = amplitudes[idx]
    max_amplitude = np.max(amplitudes)
    normalized_amplitude = peak_amplitude / max_amplitude * 100
    
    # Calculate Signal-to-Noise Ratio
    snr = peak_amplitude / noise_floor
    
    # Calculate FWHM (Full Width at Half Maximum)
    half_max = peak_amplitude / 2
    left_idx = idx
    right_idx = idx
    
    while left_idx > 0 and amplitudes[left_idx] > half_max:
        left_idx -= 1
    while right_idx < len(amplitudes) - 1 and amplitudes[right_idx] > half_max:
        right_idx += 1
    
    fwhm = periods[right_idx] - periods[left_idx]
    
    # Calculate Spectral Width and Density
    amplitude_threshold = noise_floor + (max_amplitude * 0.05)
    min_samples = int(len(periods) * 0.005)
    
    def find_boundary(start_idx, step):
        """Find cluster boundary in one direction using lookahead."""
        idx = start_idx
        while 0 <= idx < len(amplitudes):
            if amplitudes[idx] < amplitude_threshold:
                # Look ahead to check if power stays low
                lookahead_count = 0
                check_idx = idx
                while (0 <= check_idx < len(amplitudes) and 
                       check_idx * step <= idx * step + min_samples):
                    if amplitudes[check_idx] < amplitude_threshold:
                        lookahead_count += 1
                    check_idx += step
                
                if lookahead_count >= min_samples:
                    return idx - step
            idx += step
        return idx - step

    left_boundary = find_boundary(idx, -1)
    right_boundary = find_boundary(idx, 1)
    
    # Calculate spectral density
    spectral_amplitude = np.sum(amplitudes[left_boundary:right_boundary+1])
    total_amplitude = np.sum(amplitudes)
    spectral_density = spectral_amplitude / total_amplitude * 100
    
    # Calculate spectral width metrics
    spectral_width_range = (periods[left_boundary], periods[right_boundary])
    spectral_width = periods[right_boundary] - periods[left_boundary]
    
    return {
        'period': peak_period,
        'amplitude': peak_amplitude,
        'normalized_amplitude': normalized_amplitude,
        'fwhm': fwhm,
        'snr': snr,
        'spectral_density': spectral_density,
        'spectral_width': spectral_width,
        'spectral_width_range': spectral_width_range,
        'is_harmonic': False,
        'harmonic_of': None,
        'harmonic_fraction': None
    }

def print_results(fft_results):
    """Print analysis results to stdout."""
    print("\nDominant periods detected:")
    peak_stats = fft_results['peak_stats']
    harmonics_present = fft_results['harmonics_present']
    
    for i, stats in enumerate(peak_stats):
        print(f"{i+1}. Period: {stats['period']:.2f} hours")
        print(f"   Amplitude: {stats['amplitude']:.5f} ({stats['normalized_amplitude']:.2f}% of maximum)")
        print(f"   Spectral Density: {stats['spectral_density']:.2f}%")
        print(f"   Spectral Width: {stats['spectral_width']:.2f} hours "
              f"({stats['spectral_width_range'][0]:.1f}h - {stats['spectral_width_range'][1]:.1f}h)")
        print(f"   Full Width at Half Maximum: {stats['fwhm']:.2f} hours")
        print(f"   Signal-to-Noise Ratio: {stats['snr']:.2f}")
        
        # Only display "Is Harmonic" if any harmonics were detected in the analysis
        if harmonics_present:
            print(f"   Is Harmonic: {'Yes' if stats.get('is_harmonic', False) else 'No'}")
            
            # Only show "Harmonic of" for peaks that are actually harmonics
            if stats.get('is_harmonic', False):
                print(f"   Harmonic of: {stats['harmonic_of']:.2f} hours ({stats['harmonic_fraction']:.2f}x)")
    
    # Print global parameters
    print(f"\nNoise floor: {fft_results['noise_floor']:.5f}")
    print(f"Amplitude threshold: {fft_results['amplitude_threshold']:.5f}")
    print(f"Maximum amplitude: {max(stats['amplitude'] for stats in peak_stats):.5f}")
    print(f"Spectral entropy: {fft_results['spectral_entropy']:.4f} (0=structured, 1=random)")
    
def plot_fft_results(fft_results, output_dir):
    """
    Plot FFT analysis results focused on the 12-48 hour range.
    
    Args:
        fft_results: Dictionary with FFT results
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 8))
    
    # Get periods in the 12-48 hour range
    periods_hours = fft_results['periods']
    amplitudes = fft_results['amplitudes']
    
    # Filter to specified range
    range_start_hours = 12
    range_end_hours = 48
    mask = (periods_hours >= range_start_hours) & (periods_hours <= range_end_hours)
    filtered_periods = periods_hours[mask]
    filtered_amplitudes = amplitudes[mask]
    
    if len(filtered_periods) == 0:
        print("Warning: No periods found in the 12-48 hour range.")
        return
    
    # Plot periods vs. amplitudes
    plt.plot(filtered_periods, filtered_amplitudes, linewidth=1.5)
    
    # Set fixed tick positions and labels
    hours_ticks = [12, 18, 20, 22, 24, 26, 28, 30, 36, 42, 48]
    plt.xticks(hours_ticks, [f"{h}h" for h in hours_ticks])
    
    # Color the 24h tick label red
    tick_labels = plt.gca().get_xticklabels()
    tick_positions = plt.gca().get_xticks()
    for i, pos in enumerate(tick_positions):
        if pos == 24:
            tick_labels[i].set_color('red')
    
    # Add markers for dominant periods in this range
    peak_stats = fft_results['peak_stats']
    harmonics_present = fft_results['harmonics_present']
    
    for i, stats in enumerate(peak_stats):
        period_hours = stats['period']
        
        if 12 <= period_hours <= 48:
            amp = stats['amplitude']
            plt.plot(period_hours, amp, 'ro', markersize=8)
            
            # Add asterisk if this period is a harmonic
            harmonic_marker = "*" if stats['is_harmonic'] else ""
            
            # Create offset for label to avoid overlap
            y_offset = 5 + (i % 3) * 10  # Cycle through 3 different heights
            x_offset = 5 if i % 2 == 0 else -30  # Alternate left and right
            
            plt.annotate(f"{period_hours:.1f}h{harmonic_marker}",
                         xy=(period_hours, amp),
                         xytext=(x_offset, y_offset),
                         textcoords='offset points',
                         fontsize=9,
                         arrowprops=dict(arrowstyle="->", color='black', lw=1))
    
    plt.xlabel('Period (hours)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('FFT Sleep Cycle Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create legend
    legend_handles = [
        plt.Line2D([0], [0], color='white', marker='o', markerfacecolor='red', 
                   markersize=8, label='Dominant period')
    ]
    
    # Only add harmonic legend if harmonics are present
    if harmonics_present:
        legend_handles.append(
            plt.Line2D([0], [0], color='white', marker='o', markerfacecolor='red',
                      markersize=8, label='* Harmonic')
        )
    
    plt.legend(handles=legend_handles)
    
    plt.savefig(os.path.join(output_dir, 'fft_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def export_results_to_csv(fft_results, output_dir, write_full_data=False):
    """Export FFT analysis results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export peak statistics
    peak_stats = fft_results['peak_stats']
    results_df = pd.DataFrame([{
        'period_hours': format_float_export(stats['period']),
        'amplitude': format_float_export(stats['amplitude']),
        'normalized_amplitude_percent': format_float_export(stats['normalized_amplitude']),
        'spectral_density_percent': format_float_export(stats['spectral_density']),
        'spectral_width_hours': format_float_export(stats['spectral_width']),
        'spectral_width_start_hours': format_float_export(stats['spectral_width_range'][0]),
        'spectral_width_end_hours': format_float_export(stats['spectral_width_range'][1]),
        'fwhm_hours': format_float_export(stats['fwhm']),
        'snr': format_float_export(stats['snr']),
        'is_harmonic': 1 if stats['is_harmonic'] else 0,
        'harmonic_of_hours': format_float_export(stats['harmonic_of']) if stats['is_harmonic'] else '',
        'harmonic_fraction': format_float_export(stats['harmonic_fraction']) if stats['is_harmonic'] else ''
    } for stats in peak_stats])
    
    results_df.to_csv(os.path.join(output_dir, 'dominant_periods.csv'), index=False)
    
    # Export global parameters
    global_df = pd.DataFrame([{
        'noise_floor': format_float_export(fft_results['noise_floor']),
        'amplitude_threshold': format_float_export(fft_results['amplitude_threshold']),
        'max_amplitude': format_float_export(np.max(fft_results['amplitudes'])),
        'spectral_entropy': format_float_export(fft_results['spectral_entropy'])
    }])
    
    global_df.to_csv(os.path.join(output_dir, 'fft_parameters.csv'), index=False)
    
    # Export full FFT results if requested
    if write_full_data:
        full_df = pd.DataFrame({
            'period_hours': [format_float_export(p) for p in fft_results['periods']],
            'amplitude': [format_float_export(a) for a in fft_results['amplitudes']]
        }).sort_values('period_hours')  # Sort by period ascending
        
        full_df.to_csv(os.path.join(output_dir, 'full_fft_results.csv'), index=False)

def export_results_to_json(fft_results, output_dir, write_full_data=False):
    """Export FFT analysis results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare comprehensive results
    results = {
        'dominant_periods': [{
            'period_hours': stats['period'],
            'amplitude': stats['amplitude'],
            'normalized_amplitude_percent': stats['normalized_amplitude'],
            'spectral_density_percent': stats['spectral_density'],
            'spectral_width_hours': stats['spectral_width'],
            'spectral_width_range': {
                'start_hours': stats['spectral_width_range'][0],
                'end_hours': stats['spectral_width_range'][1]
            },
            'fwhm_hours': stats['fwhm'],
            'snr': stats['snr'],
            'is_harmonic': stats['is_harmonic'],
            'harmonic_of_hours': stats['harmonic_of'] if stats['is_harmonic'] else None,
            'harmonic_fraction': stats['harmonic_fraction'] if stats['is_harmonic'] else None
        } for stats in fft_results['peak_stats']],
        'analysis_parameters': {
            'noise_floor': fft_results['noise_floor'],
            'amplitude_threshold': fft_results['amplitude_threshold'],
            'spectral_entropy': fft_results['spectral_entropy']
        }
    }
    
    with open(os.path.join(output_dir, 'fft_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Export full FFT results if requested
    if write_full_data:
        full_results = [{
            'period_hours': period,
            'amplitude': amplitude
        } for period, amplitude in sorted(zip(fft_results['periods'], 
                                           fft_results['amplitudes']), 
                                       key=lambda x: x[0])]  # Sort by period ascending
        
        with open(os.path.join(output_dir, 'full_fft_results.json'), 'w') as f:
            json.dump(full_results, f, indent=2)

def main():
    """Main function to run analysis."""
    args = parse_arguments()
    
    try:
        # Load sleep data
        df = load_sleep_data(args.input_file, args.range_start, args.range_end)
        
        # Generate binary time series
        time_series_data = generate_binary_time_series(df, args.sampling_interval)
        
        # Run FFT analysis
        fft_results = perform_fft_analysis(time_series_data, args.min_peak_distance)
        
        # Print results to stdout
        print_results(fft_results)
        
        # Generate file outputs if requested
        will_write_files = args.save_plot or args.csv or args.json
        if will_write_files:
            os.makedirs(args.output_dir, exist_ok=True)
            
            if args.save_plot:
                plot_fft_results(fft_results, args.output_dir)
            
            if args.csv or args.json:
                if args.csv:
                    export_results_to_csv(fft_results, args.output_dir, args.write_full_data)
                if args.json:
                    export_results_to_json(fft_results, args.output_dir, args.write_full_data)
            
            print(f"\nResults saved to {args.output_dir}/")
    
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())