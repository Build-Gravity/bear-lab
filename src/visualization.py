from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import matplotlib.lines # For Line2D type hint
import os

if TYPE_CHECKING:
    from src.processing import ActivationKineticResults # type: ignore

# type: ignore
def plot_well_kinetics_with_fits(
    time_sec: pd.Series[float],
    fluorescence: pd.Series[float], # This will receive dF/F0 data when is_dFF0_data is True
    analysis_results: ActivationKineticResults,
    well_id: str,
    source_file: str,
    block_name: str,
    output_dir: str,
    min_points_per_phase: int = 2, # Default consistent with processing
    is_dFF0_data: bool = False # New parameter
) -> None:
    """
    Generates and saves a plot for a single well's kinetic data, showing raw data points,
    fitted linear slopes for increasing and decreasing phases, and the peak point.
    If is_dFF0_data is True, Y-axis is labeled as 'ΔF/F₀' and input 'fluorescence' is treated as dF/F0.

    Args:
        time_sec: Pandas Series of time data in seconds (float).
        fluorescence: Pandas Series of fluorescence data or dF/F0 data (float).
        analysis_results: Dictionary containing kinetic analysis results for the well.
                          Expected keys correspond to ActivationKineticResults from processing.py
        well_id: Identifier for the well (e.g., "A1").
        source_file: Name of the source XLSX file.
        block_name: Name of the block (e.g., "Activation").
        output_dir: Directory where the plot image will be saved.
        min_points_per_phase: Minimum number of points required for a phase to have a slope plotted.
        is_dFF0_data: If True, data in 'fluorescence' is treated as dF/F0 and Y-axis is labeled accordingly.
    """
    # Ensure data is clean for plotting (remove NaNs from aligned series)
    combined_df = pd.DataFrame({'time': time_sec, 'signal': fluorescence}).dropna() # type: ignore[call-arg, no-untyped-call]
    if combined_df.empty:
        # print(f"Plotting skipped for {well_id} in {source_file}/{block_name}: No valid data points after NaN removal.")
        return

    clean_time_sec: pd.Series[float] = combined_df['time'].reset_index(drop=True)
    clean_signal_values: pd.Series[float] = combined_df['signal'].reset_index(drop=True) # Renamed for clarity

    if clean_time_sec.empty: # Should be caught by combined_df.empty, but as a safeguard
        # print(f"Plotting skipped for {well_id} in {source_file}/{block_name}: Time data is empty after cleaning.")
        return

    # Use keys from ActivationKineticResults
    peak_idx: Optional[int] = analysis_results.get('peak_delta_f_f0_index')
    inc_slope: Optional[float] = analysis_results.get('increasing_slope_dFF0')
    dec_slope: Optional[float] = analysis_results.get('decreasing_slope_dFF0')

    plt.figure(figsize=(10, 6)) # type: ignore[no-untyped-call]
    plt.plot(clean_time_sec, clean_signal_values, 'o-', label="Raw Data", markersize=5, linewidth=1) # type: ignore[no-untyped-call]

    plot_title_str: str = f"Kinetics: {source_file} - {block_name} - Well {well_id}"
    _comment_val = analysis_results.get('comment')
    comment_text: str = _comment_val if _comment_val is not None else ''
    if comment_text:
        # Make title concise if comment is long, or wrap comment. For now, simple concatenation.
        plot_title_str += f"\nStatus: {comment_text}"
    
    plt.title(plot_title_str, fontsize=12) # type: ignore[no-untyped-call]
    plt.xlabel("Time (seconds)", fontsize=10) # type: ignore[no-untyped-call]
    
    y_label_str = "ΔF/F₀" if is_dFF0_data else "Fluorescence"
    plt.ylabel(y_label_str, fontsize=10) # type: ignore[no-untyped-call]
    plt.grid(True, linestyle='--', alpha=0.7) # type: ignore[no-untyped-call]

    legend_handles_list: list[matplotlib.lines.Line2D] = [] # Explicitly use matplotlib.lines.Line2D

    # Plot peak point if identified
    if peak_idx is not None and 0 <= peak_idx < len(clean_time_sec):
        peak_time_val: float = clean_time_sec.iloc[peak_idx]
        peak_signal_val: float = clean_signal_values.iloc[peak_idx] # Renamed
        peak_marker = plt.plot(peak_time_val, peak_signal_val, 'X', color='red', markersize=10, label=f"Peak (idx {peak_idx})") # type: ignore[no-untyped-call]
        if peak_marker: 
            legend_handles_list.append(peak_marker[0])


    # Plot increasing phase fit
    if inc_slope is not None and peak_idx is not None and 0 <= peak_idx < len(clean_time_sec) :
        time_inc_series: pd.Series[float] = clean_time_sec.iloc[0:peak_idx + 1]
        signal_inc_series: pd.Series[float] = clean_signal_values.iloc[0:peak_idx + 1] # Renamed
        
        if len(time_inc_series) >= min_points_per_phase and not time_inc_series.empty: 
            if not time_inc_series.empty:
                # Calculate intercept so the line passes through the peak point
                intercept_inc_val: float = signal_inc_series.iloc[-1] - inc_slope * time_inc_series.iloc[-1]
                fit_line_inc_series: pd.Series[float] = inc_slope * time_inc_series + intercept_inc_val
                inc_line = plt.plot(time_inc_series, fit_line_inc_series, '--', color='green', linewidth=2, label=f"Increasing Slope: {inc_slope:.2e}") # type: ignore[no-untyped-call]
                if inc_line:
                    legend_handles_list.append(inc_line[0])


    # Plot decreasing phase fit
    if dec_slope is not None and peak_idx is not None and 0 <= peak_idx < len(clean_time_sec):
        time_dec_series: pd.Series[float] = clean_time_sec.iloc[peak_idx:]
        signal_dec_series: pd.Series[float] = clean_signal_values.iloc[peak_idx:] # Renamed

        if len(time_dec_series) >= min_points_per_phase and not time_dec_series.empty:
            if not time_dec_series.empty:
                intercept_dec_val: float = signal_dec_series.iloc[0] - dec_slope * time_dec_series.iloc[0]
                fit_line_dec_series: pd.Series[float] = dec_slope * time_dec_series + intercept_dec_val
                dec_line = plt.plot(time_dec_series, fit_line_dec_series, '--', color='purple', linewidth=2, label=f"Decreasing Slope: {dec_slope:.2e}") # type: ignore[no-untyped-call]
                if dec_line:
                    legend_handles_list.append(dec_line[0])

    # Add raw data line to legend handles if not already there (it might be if no fits)
    raw_data_line_present_flag = any(
        h.get_label() == "Raw Data" for h in legend_handles_list if hasattr(h, 'get_label') and h.get_label() is not None
    )
    if not raw_data_line_present_flag:
        if plt.gca().get_lines(): # type: ignore[no-untyped-call]
            raw_line_item = plt.gca().get_lines()[0] # type: ignore[no-untyped-call]
            legend_handles_list.insert(0, raw_line_item)


    if legend_handles_list:
        sorted_handles_list: list[matplotlib.lines.Line2D] = []
        # Simplified sorting logic for legend
        order = ["Raw Data", "Peak", "Increasing Slope", "Decreasing Slope"]
        for label_prefix in order:
            for h in legend_handles_list:
                if str(h.get_label()).startswith(label_prefix) and h not in sorted_handles_list:
                    sorted_handles_list.append(h)
        
        # Add any remaining handles that didn't match the predefined order
        for h in legend_handles_list:
            if h not in sorted_handles_list:
                sorted_handles_list.append(h)

        if sorted_handles_list:
             plt.legend(handles=sorted_handles_list, fontsize=9) # type: ignore[no-untyped-call]
        elif legend_handles_list: # Fallback if sorting logic failed
            plt.legend(fontsize=9) # type: ignore[no-untyped-call]

    else: # If no fits and no peak, just the raw data was plotted
        plt.legend(fontsize=9) # type: ignore[no-untyped-call]


    # Create output directory if it doesn't exist
    # Sanitize block_name and source_file for path creation
    safe_block_name_str: str = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in block_name).rstrip()
    safe_source_file_str: str = os.path.splitext(source_file)[0] # Remove extension
    safe_source_file_str = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in safe_source_file_str).rstrip()


    # Construct path for block-specific subdirectory
    block_specific_dir_path: str = os.path.join(output_dir, safe_block_name_str)
    if not os.path.exists(block_specific_dir_path):
        try:
            os.makedirs(block_specific_dir_path)
        except OSError:
            # print(f"Error creating directory {block_specific_dir_path}: {e}. Saving to base output directory.")
            block_specific_dir_path = output_dir # Fallback to base output_dir

    # Construct filename
    plot_filename_str: str = f"{safe_source_file_str}_{safe_block_name_str}_Well_{well_id}.png"
    full_plot_path_str: str = os.path.join(block_specific_dir_path, plot_filename_str)

    try:
        plt.tight_layout(rect=(0, 0, 1, 0.96)) # type: ignore[no-untyped-call]
        plt.savefig(full_plot_path_str) # type: ignore[no-untyped-call]
        # print(f"Plot saved to {full_plot_path_str}")
    except Exception:
        # print(f"Error saving plot {full_plot_path_str}: {e}")
        pass # Continue if plotting fails for one well
    finally:
        plt.close() # type: ignore[no-untyped-call] # Close the figure to free memory 