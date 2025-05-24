from __future__ import annotations
from typing import List, Tuple, Optional, TypedDict, Dict, Any
import pandas as pd
from scipy.stats import linregress # type: ignore

from src.block_parser import parse_block, MetadataDict
from src.visualization import plot_well_kinetics_with_fits

# Type alias for a raw block tuple
RawBlock = Tuple[str, pd.DataFrame]

# Type alias for a successfully parsed block result
ParsedBlockResult = Tuple[MetadataDict, pd.DataFrame]

# Type alias for the result of a parsing attempt (may have failed)
ParsingAttemptResult = Tuple[MetadataDict, Optional[pd.DataFrame]]

# Type alias for F0 values map for a block
F0ValuesMap = Dict[str, float]

# Type alias for a paired block ready for dF/F0 analysis
PairedBlockForAnalysis = Tuple[
    ParsedBlockResult, # Background block (metadata, df)
    Tuple[MetadataDict, pd.DataFrame], # Activation block (metadata, df_with_dFF0)
    F0ValuesMap # The F0 values map associated with this pair for convenience
]

DEFAULT_MIN_POINTS_FOR_ANALYSIS: int = 4
DEFAULT_MIN_POINTS_PER_PHASE: int = 2
DEFAULT_F0_POINTS_TO_AVERAGE: int = 3 # New default for F0 calculation

# Updated keys for ΔF/F₀ analysis
ActivationKineticResults = TypedDict('ActivationKineticResults', {
    'increasing_slope_dFF0': Optional[float],
    'increasing_r_squared_dFF0': Optional[float],
    'decreasing_slope_dFF0': Optional[float],
    'decreasing_r_squared_dFF0': Optional[float],
    'peak_delta_f_f0_index': Optional[int],
    'comment': Optional[str]
})

def process_all_blocks(
    raw_blocks_with_sources: List[RawBlock],
    block_name_prefix: str = "plate:" # New parameter with default
) -> List[ParsingAttemptResult]:
    """
    Processes a list of raw blocks, applying the parse_block function to each.

    Returns a result for every raw block attempted. The DataFrame part
    of ParsingAttemptResult will be None if parsing failed (metadata will contain info).

    Args:
        raw_blocks_with_sources: A list of tuples, where each tuple contains:
            - source_filename (str): The original filename.
            - raw_block_df (pd.DataFrame): The raw DataFrame for a block.
        block_name_prefix: The string prefix to identify the block name metadata row.

    Returns:
        A list of tuples (ParsingAttemptResult), where each tuple contains:
            - metadata_dict (MetadataDict): The metadata extracted from the block.
            - cleaned_dataframe (Optional[pd.DataFrame]): The cleaned data table, or None.
    """
    all_parsing_attempts: List[ParsingAttemptResult] = []

    for source_filename, raw_block_df in raw_blocks_with_sources:
        # Pass the block_name_prefix to parse_block
        metadata, cleaned_df = parse_block(source_filename, raw_block_df, block_name_keyword_prefix=block_name_prefix)
        
        # Append result for every block, cleaned_df will be None if parsing failed
        all_parsing_attempts.append((metadata, cleaned_df))

    return all_parsing_attempts


def select_all_activation_blocks(
    parsed_blocks: List[ParsedBlockResult],
    activation_keyword: str = "activation"
) -> List[ParsedBlockResult]:
    """
    Selects all blocks from a list of parsed blocks whose metadata's 'block_name'
    contains the activation_keyword (case-insensitive) and have a valid DataFrame.

    Args:
        parsed_blocks: A list of successfully parsed blocks, where each item is a tuple
                       (metadata_dict, cleaned_dataframe).
        activation_keyword: The keyword to search for in the 'block_name' metadata field.
                            Defaults to "activation".

    Returns:
        A list of (metadata_dict, cleaned_dataframe) tuples for all matching activation blocks.
        Returns an empty list if no matching blocks are found.
    """
    activation_keyword_lower = activation_keyword.lower()
    selected_blocks: List[ParsedBlockResult] = []
    for metadata, df in parsed_blocks: # df is pd.DataFrame here due to ParsedBlockResult type
        block_name = metadata.get("block_name")
        # Ensure df is not None (guaranteed by type) and not empty.
        # Linter fix: removed "df is not None" as it's always true here.
        if isinstance(block_name, str) and activation_keyword_lower in block_name.lower() and not df.empty:
            selected_blocks.append((metadata, df))
    
    if not selected_blocks:
        # print(f"Warning: No blocks found with keyword '{activation_keyword}' in block_name.") # Optional logging
        pass
    return selected_blocks

def prepare_time_column(df: pd.DataFrame, time_column_name: str = "Time") -> pd.DataFrame:
    """
    Converts a time column (e.g., "HH:MM:SS" strings) in a DataFrame to total seconds.

    A new column 'Time_sec' is added to the DataFrame.
    Rows where the original time string cannot be parsed will have np.nan in 'Time_sec'.
    The function does not drop rows with NaT or NaN time values; this can be handled
    by downstream processes if necessary.

    Args:
        df: The input pandas DataFrame. It is modified in place if a 'Time' column exists
            and can be processed, otherwise a copy might be returned or an error logged.
            To be safe, it's better to assume it returns a modified DataFrame.
        time_column_name: The name of the column containing time strings. Defaults to "Time".

    Returns:
        The DataFrame with an added 'Time_sec' column. If the specified time_column_name
        does not exist, or if all time values fail to parse (though this is unlikely
        if at least one is valid), the original DataFrame (or a copy) is returned without
        the 'Time_sec' column, and a warning might be implied or logged.
    """
    if time_column_name not in df.columns:
        # print(f"Warning: Time column '{time_column_name}' not found in DataFrame. Skipping time preparation.")
        return df.copy() # Return a copy to avoid surprises with original df modification status

    # Create a copy to avoid SettingWithCopyWarning and ensure modification safety
    df_processed = df.copy()

    # Convert to timedelta, coercing errors to NaT (Not a Time)
    time_deltas = pd.to_timedelta(df_processed[time_column_name], errors='coerce') # type: ignore[call-arg]

    # Convert timedelta to total seconds. NaT will become np.nan.
    df_processed['Time_sec'] = time_deltas.dt.total_seconds() # type: ignore[attr-defined]

    # Optional: Log how many NaT values were encountered if any
    # nat_count = df_processed['Time_sec'].isna().sum()
    # if nat_count > 0:
    #     print(f"Warning: Encountered {nat_count} unparseable time entries in column '{time_column_name}'.")

    return df_processed 

def ensure_numeric_well_data(df: pd.DataFrame, non_well_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Converts all columns in a DataFrame, except for specified non-well columns,
    to a numeric type using pd.to_numeric with errors='coerce'.

    This is intended to prepare fluorescence data (or other similar data in well columns)
    for numerical analysis. Non-numeric values will be converted to np.nan.

    Args:
        df: The input pandas DataFrame. It is modified in place if columns are converted,
            otherwise a copy might be returned. To be safe, it's better to assume
            it returns a modified DataFrame.
        non_well_columns: A list of column names that should NOT be converted to numeric.
                          Defaults to ['Time', 'Time_sec'] if None.

    Returns:
        The DataFrame with relevant columns converted to numeric types.
    """
    if df.empty:
        return df.copy()

    df_processed = df.copy()

    # Determine columns to skip numeric conversion
    actual_non_well_columns: List[str]
    if non_well_columns is None: # Default behavior: preserve Time and Time_sec
        actual_non_well_columns = ['Time', 'Time_sec']
    elif not non_well_columns: # Empty list provided: convert all columns
        actual_non_well_columns = []
    else: # Specific list provided: preserve Time, Time_sec, and the custom ones
        actual_non_well_columns = ['Time', 'Time_sec']
        for custom_col in non_well_columns:
            if custom_col not in actual_non_well_columns:
                actual_non_well_columns.append(custom_col)

    for col in df_processed.columns:
        if col not in actual_non_well_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # type: ignore[call-arg]
            # Ensure the dtype is float if there were NaNs introduced, or if original was int.
            # pd.to_numeric with errors='coerce' on a purely int column might keep it int.
            # For consistency in fluorescence data, float is usually preferred.
            if pd.api.types.is_numeric_dtype(df_processed[col]): # type: ignore[call-arg]
                 # If it's int or int-like after to_numeric, and we want float (e.g. to hold NaNs)
                 # This check ensures we don't try to convert a column that became all NaNs (object dtype)
                 # or if it was already float.
                if not pd.api.types.is_float_dtype(df_processed[col]): # type: ignore[call-arg]
                    try:
                        df_processed[col] = df_processed[col].astype(float)
                    except ValueError:
                        # This can happen if a column becomes all NaNs and its dtype is object
                        # In such cases, astype(float) might fail. But to_numeric already made it NaN.
                        pass # Values are already NaN, dtype might be object if all were bad

    return df_processed 

# Renamed from calculate_single_well_kinetics, removed expected_shape
def calculate_activation_kinetics(
    time_sec: pd.Series[float],
    delta_f_over_f0: pd.Series[float], # Changed from fluorescence
    min_points_for_analysis: int = DEFAULT_MIN_POINTS_FOR_ANALYSIS,
    min_points_per_phase: int = DEFAULT_MIN_POINTS_PER_PHASE
) -> ActivationKineticResults: # Updated return type
    """
    Calculates biphasic kinetic parameters for a single well, using dF/F0 data,
    assuming an "activation" profile (increase then decrease, peak is max dF/F0).

    Args:
        time_sec: Pandas Series of time data in seconds (float).
        delta_f_over_f0: Pandas Series of dF/F0 data (float).
        min_points_for_analysis: Minimum number of valid (non-NaN) data points
                                 required for the entire well analysis.
        min_points_per_phase: Minimum number of data points required for each phase
                              to calculate a slope.

    Returns:
        An ActivationKineticResults dictionary containing slopes, R-squared values, 
        peak dF/F0 index, and a comment if analysis could not be fully performed.
    """
    # Updated dictionary keys are already in ActivationKineticResults
    results: ActivationKineticResults = {
        'increasing_slope_dFF0': None,
        'increasing_r_squared_dFF0': None,
        'decreasing_slope_dFF0': None,
        'decreasing_r_squared_dFF0': None,
        'peak_delta_f_f0_index': None,
        'comment': None
    }

    if not isinstance(time_sec, pd.Series) or not isinstance(delta_f_over_f0, pd.Series): # type: ignore[redundant-expr]
        results['comment'] = "Error: time_sec and delta_f_over_f0 must be pandas Series."
        return results
        
    combined_df = pd.DataFrame({'time': time_sec, 'dFF0': delta_f_over_f0}).dropna() # type: ignore[call-arg]
    
    if len(combined_df) < 1:
        results['comment'] = "No valid (non-NaN) data points after alignment."
        return results

    clean_time_sec: pd.Series[float] = combined_df['time'].reset_index(drop=True)
    clean_delta_f_over_f0: pd.Series[float] = combined_df['dFF0'].reset_index(drop=True)

    num_valid_points = len(clean_time_sec)

    if num_valid_points < min_points_for_analysis:
        results['comment'] = f"Insufficient data points ({num_valid_points}) for biphasic analysis (min: {min_points_for_analysis})."
        return results

    # --- Turning Point Detection: Always use idxmax() for activation kinetics (max dF/F0) ---
    peak_idx: int = clean_delta_f_over_f0.idxmax() # type: ignore 
    results['peak_delta_f_f0_index'] = peak_idx

    # --- Calculate Slopes using Linear Regression ---
    comments: List[str] = []

    # Increasing Phase
    time_increasing: pd.Series[float] = clean_time_sec.iloc[0:peak_idx + 1]
    dFF0_increasing: pd.Series[float] = clean_delta_f_over_f0.iloc[0:peak_idx + 1]

    if len(time_increasing) >= min_points_per_phase:
        valid_indices_increasing = time_increasing.notna() & dFF0_increasing.notna()
        time_increasing_clean = time_increasing[valid_indices_increasing]
        dFF0_increasing_clean = dFF0_increasing[valid_indices_increasing]
        
        if len(time_increasing_clean) >= min_points_per_phase:
            if len(time_increasing_clean.unique()) > 1: # type: ignore[no-untyped-call]
                regression_results_increasing = linregress(time_increasing_clean, dFF0_increasing_clean)
                results['increasing_slope_dFF0'] = float(regression_results_increasing.slope) # type: ignore[attr-defined]
                results['increasing_r_squared_dFF0'] = float(regression_results_increasing.rvalue**2) # type: ignore[attr-defined]
            else:
                results['increasing_slope_dFF0'] = 0.0
                results['increasing_r_squared_dFF0'] = 0.0 
                comments.append(f"Increasing phase: All time points are identical ({len(time_increasing_clean)} points), slope undefined.")
        else:
            comments.append(f"Increasing phase: Insufficient valid data points ({len(time_increasing_clean)}) after NaN removal (min: {min_points_per_phase}).")
    else:
        comments.append(f"Increasing phase: Insufficient data points ({len(time_increasing)}) for slope calculation (min: {min_points_per_phase}).")

    # Decreasing Phase
    time_decreasing: pd.Series[float] = clean_time_sec.iloc[peak_idx:]
    dFF0_decreasing: pd.Series[float] = clean_delta_f_over_f0.iloc[peak_idx:]

    if len(time_decreasing) >= min_points_per_phase:
        valid_indices_decreasing = time_decreasing.notna() & dFF0_decreasing.notna()
        time_decreasing_clean = time_decreasing[valid_indices_decreasing]
        dFF0_decreasing_clean = dFF0_decreasing[valid_indices_decreasing]

        if len(time_decreasing_clean) >= min_points_per_phase:
            if len(time_decreasing_clean.unique()) > 1: # type: ignore[no-untyped-call]
                regression_results_decreasing = linregress(time_decreasing_clean, dFF0_decreasing_clean)
                results['decreasing_slope_dFF0'] = float(regression_results_decreasing.slope) # type: ignore[attr-defined]
                results['decreasing_r_squared_dFF0'] = float(regression_results_decreasing.rvalue**2) # type: ignore[attr-defined]
            else:
                results['decreasing_slope_dFF0'] = 0.0
                results['decreasing_r_squared_dFF0'] = 0.0
                comments.append(f"Decreasing phase: All time points are identical ({len(time_decreasing_clean)} points), slope undefined.")
        else:
            comments.append(f"Decreasing phase: Insufficient valid data points ({len(time_decreasing_clean)}) after NaN removal (min: {min_points_per_phase}).")
    else:
        comments.append(f"Decreasing phase: Insufficient data points ({len(time_decreasing)}) for slope calculation (min: {min_points_per_phase}).")
    
    # Consolidate comments
    existing_comment = results['comment']
    if existing_comment:
        if comments:
            results['comment'] = existing_comment + " | " + " | ".join(comments)
    elif comments:
        results['comment'] = " | ".join(comments)

    # Add specific comment for monotonic/single phase if applicable and no other primary error comment exists
    if not existing_comment: # Only add this if there wasn't a more severe initial comment
        is_monotonic_or_one_phase = False
        phase_specific_monotonic_comments: List[str] = []

        # Peak at start implies primarily decreasing trend for an inverted-U shape context
        if peak_idx == 0 and results['decreasing_slope_dFF0'] is not None and results['increasing_slope_dFF0'] is None:
            is_monotonic_or_one_phase = True
            if not any("Increasing phase" in c for c in comments if c): # Check if comments is not empty before iterating
                 phase_specific_monotonic_comments.append("Increasing phase too short (data starts at peak).")
        # Peak at end implies primarily increasing trend
        elif peak_idx == num_valid_points - 1 and results['increasing_slope_dFF0'] is not None and results['decreasing_slope_dFF0'] is None:
            is_monotonic_or_one_phase = True
            if not any("Decreasing phase" in c for c in comments if c):
                 phase_specific_monotonic_comments.append("Decreasing phase too short (data ends at peak).")
        
        if is_monotonic_or_one_phase:
            current_comments_str = results['comment']
            monotonic_label = "Monotonic or single effective phase."
            if phase_specific_monotonic_comments:
                monotonic_label += " (" + ", ".join(phase_specific_monotonic_comments) + ")"
            
            if current_comments_str:
                if monotonic_label not in current_comments_str: 
                    results['comment'] = current_comments_str + " | " + monotonic_label
            else:
                results['comment'] = monotonic_label
        elif not results['increasing_slope_dFF0'] and not results['decreasing_slope_dFF0'] and not results['comment'] and not comments:
             results['comment'] = "Both phases too short or invalid for slope calculation."
        elif not results['comment'] and not comments and results['increasing_slope_dFF0'] is not None and results['decreasing_slope_dFF0'] is not None:
            results['comment'] = "Biphasic analysis successful."


    return results

def identify_well_columns(df: pd.DataFrame, non_well_cols: Optional[List[str]] = None) -> List[str]:
    """
    Identifies columns in a DataFrame that are likely to be well data columns.
    It excludes known non-data columns and columns that are entirely NaN after numeric conversion attempt.

    Args:
        df: The input pandas DataFrame.
        non_well_cols: A list of column names that are known not to be well data.
                       Defaults include 'Time', 'Time_sec', 'Temperature(¡C)'.

    Returns:
        A list of string column names identified as well data.
    """
    if df.empty:
        return []

    # Define default non-well columns (case-insensitive considerations might be needed if headers vary greatly)
    default_non_well_list = ['Time', 'Time_sec', 'Temperature(¡C)', 'Temperature (°C)', 'Temperature(°C)', 'Plate', 'Sample']
    
    current_non_well_set = set(s.lower() for s in default_non_well_list) # Use a set for efficient lookup
    if non_well_cols:
        current_non_well_set.update(s.lower() for s in non_well_cols)

    potential_well_cols: List[str] = []
    for col_name_obj in df.columns:
        col_name = str(col_name_obj) # Ensure col_name is a string
        if col_name.lower() not in current_non_well_set:
            # Further check: try to convert to numeric and see if it's all NaN
            # This helps filter out columns that might look like wells but are text/empty
            try:
                numeric_series = pd.to_numeric(df[col_name_obj], errors='coerce') # type: ignore[call-arg]
                if not numeric_series.isnull().all(): # type: ignore[no-untyped-call]
                    potential_well_cols.append(col_name)
            except Exception:
                # If to_numeric fails for some reason, or it's not a data column, skip
                pass 
                
    return potential_well_cols

def calculate_f0_for_block(
    background_df: pd.DataFrame,
    f0_points_to_average: int = DEFAULT_F0_POINTS_TO_AVERAGE,
    non_well_columns: Optional[List[str]] = None
) -> F0ValuesMap:
    """
    Calculates F0 (baseline fluorescence) for each well in a background block DataFrame.

    Args:
        background_df: DataFrame containing the background block data.
                       Assumes well data is numeric and 'Time'/'Time_sec' might exist.
        f0_points_to_average: Number of last data points to average for F0.
        non_well_columns: Optional list of columns to exclude from F0 calculation.

    Returns:
        A dictionary mapping well IDs (str) to their calculated F0 values (float).
        Wells with no valid data or too few points will have F0 as float('nan').
    """
    f0_values: F0ValuesMap = {}
    if background_df.empty:
        return f0_values

    # Ensure well data is numeric before processing, applying to a copy
    # The original ensure_numeric_well_data can be used if it's confirmed non_well_columns handling is appropriate.
    # For safety, let's assume background_df might need this step specifically for its context.
    temp_background_df = ensure_numeric_well_data(background_df.copy(), non_well_columns=non_well_columns)

    well_ids = identify_well_columns(temp_background_df, non_well_cols=non_well_columns)

    for well_id in well_ids:
        if well_id not in temp_background_df.columns:
            continue

        # Explicitly work with a Series of floats
        current_well_series: pd.Series[float] = pd.to_numeric(temp_background_df[well_id], errors='coerce').dropna().astype(float) # type: ignore[no-untyped-call]
        
        if current_well_series.empty:
            f0_values[well_id] = float('nan')
            continue

        f0: float
        if len(current_well_series) >= f0_points_to_average:
            f0 = current_well_series.iloc[-f0_points_to_average:].mean() # type: ignore[no-untyped-call]
        elif len(current_well_series) > 0: # type: ignore[arg-type] # Average all available points if fewer than N
            f0 = current_well_series.mean() # type: ignore[no-untyped-call]
        else: # No data points
            f0 = float('nan')
        
        f0_values[well_id] = float(f0) if pd.notna(f0) else float('nan') # type: ignore[call-arg]

    return f0_values

def prepare_paired_blocks_for_analysis(
    parsed_blocks_results: List[ParsingAttemptResult],
    background_keyword: str = "background",
    activation_keyword: str = "activation",
    f0_points_to_average: int = DEFAULT_F0_POINTS_TO_AVERAGE,
    non_well_columns_for_f0: Optional[List[str]] = None,
    non_well_columns_for_activation: Optional[List[str]] = None,
    time_column_name: str = "Time"
) -> List[PairedBlockForAnalysis]:
    """
    Identifies background and activation block pairs, calculates F0, 
    and then calculates dF/F0 for the activation block data.

    Args:
        parsed_blocks_results: List of (metadata, Optional[df]) from process_all_blocks.
        background_keyword: Keyword to identify background blocks by name.
        activation_keyword: Keyword to identify activation blocks by name.
        f0_points_to_average: Number of points for F0 calculation.
        non_well_columns_for_f0: Columns to ignore during F0 calculation in background block.
        non_well_columns_for_activation: Columns to ignore during dF/F0 calculation in activation block.
        time_column_name: Name of the time column in activation blocks.

    Returns:
        A list of PairedBlockForAnalysis tuples. Each tuple contains:
        - (background_metadata, background_df)
        - (activation_metadata, activation_df_with_dFF0) # activation_df has dF/F0 values
        - F0ValuesMap for the pair
    """
    paired_blocks: List[PairedBlockForAnalysis] = []    
    valid_parsed_blocks: List[ParsedBlockResult] = []
    for meta, df_opt in parsed_blocks_results:
        if df_opt is not None and not df_opt.empty:
            valid_parsed_blocks.append((meta, df_opt))

    # Group blocks by source filename to facilitate pairing
    blocks_by_file: Dict[str, List[ParsedBlockResult]] = {}
    for meta, df in valid_parsed_blocks:
        source_file = meta.get("source_filename", "UnknownFile")
        if source_file not in blocks_by_file:
            blocks_by_file[source_file] = []
        blocks_by_file[source_file].append((meta, df))

    bg_kw_lower = background_keyword.lower()
    act_kw_lower = activation_keyword.lower()

    for source_file, blocks_in_file in blocks_by_file.items():
        # Assume blocks are in order; iterate to find pairs
        for i in range(len(blocks_in_file) - 1):
            block1_meta, block1_df = blocks_in_file[i]
            block2_meta, block2_df = blocks_in_file[i+1]

            block1_name = str(block1_meta.get("block_name", "")).lower()
            block2_name = str(block2_meta.get("block_name", "")).lower()

            # Check for background -> activation pair
            if bg_kw_lower in block1_name and act_kw_lower in block2_name:
                background_block_parsed: ParsedBlockResult = (block1_meta, block1_df)
                activation_block_parsed_orig: ParsedBlockResult = (block2_meta, block2_df)

                # 1. Calculate F0 for the background block
                f0_map = calculate_f0_for_block(
                    background_df=block1_df.copy(), # Use a copy for safety
                    f0_points_to_average=f0_points_to_average,
                    non_well_columns=non_well_columns_for_f0
                )

                if not f0_map: # Skip if no F0 values could be calculated
                    # print(f"Skipping pair for {source_file} background {block1_name} - no F0 values.")
                    continue

                # 2. Prepare activation block: numeric wells and time column
                act_df_prepared = activation_block_parsed_orig[1].copy()
                act_df_prepared = ensure_numeric_well_data(act_df_prepared, non_well_columns=non_well_columns_for_activation)
                act_df_prepared = prepare_time_column(act_df_prepared, time_column_name=time_column_name)

                # 3. Calculate dF/F0 for the activation block
                well_ids_activation = identify_well_columns(act_df_prepared, non_well_cols=non_well_columns_for_activation)
                
                act_df_with_dFF0 = act_df_prepared.copy()

                for well_id in well_ids_activation:
                    if well_id not in act_df_with_dFF0.columns: # Should not happen
                        continue
                    
                    f0_well = f0_map.get(well_id)

                    if f0_well is not None and f0_well != 0 and pd.notna(f0_well): # type: ignore[call-arg]
                        fluorescence_series: pd.Series[float] = act_df_with_dFF0[well_id].astype(float)
                        dff0_series: pd.Series[float] = (fluorescence_series - f0_well) / f0_well
                        act_df_with_dFF0[well_id] = dff0_series
                    else:
                        # If F0 is missing, NaN, or zero, set dF/F0 to NaN for the whole series
                        act_df_with_dFF0[well_id] = float('nan') 
                
                final_activation_block_tuple = (activation_block_parsed_orig[0], act_df_with_dFF0)

                paired_blocks.append((
                    background_block_parsed,
                    final_activation_block_tuple,
                    f0_map
                ))
    return paired_blocks

def analyze_block_kinetics(
    paired_block_for_analysis: PairedBlockForAnalysis,
    min_points_for_analysis: int = DEFAULT_MIN_POINTS_FOR_ANALYSIS,
    min_points_per_phase: int = DEFAULT_MIN_POINTS_PER_PHASE,
    custom_non_well_columns: Optional[List[str]] = None, # Will be used for identify_well_columns on dFF0 df
    output_plot_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyzes all wells in a single prepared activation block (which already contains dF/F0 data)
    for activation kinetic parameters. Assumes an "increase then decrease" (peak) pattern.

    Args:
        paired_block_for_analysis: A PairedBlockForAnalysis tuple containing:
            - ((background_meta, background_df), 
            -  (activation_meta, activation_df_with_dFF0), 
            -  f0_map)
        min_points_for_analysis: Min data points for a well to be analyzed.
        min_points_per_phase: Min data points per phase for slope calculation.
        custom_non_well_columns: Optional list of columns to exclude from well identification 
                                 in the activation_df_with_dFF0.
        output_plot_dir: Optional path to a directory where plots for each well will be saved.
                         If None, plotting is skipped.

    Returns:
        A pandas DataFrame containing kinetic results for all analyzable wells in the block.
        Columns include WellID, F0_Value, kinetic parameters (Increasing_Slope_dFF0, etc.), 
        comments, and source/block metadata.
        Returns an empty DataFrame with correct columns if no wells are analyzed.
    """
    background_block_parsed, activation_block_processed, f0_map = paired_block_for_analysis
    bg_metadata, _ = background_block_parsed
    act_metadata, act_df_with_dFF0 = activation_block_processed
    
    all_results_list: List[Dict[str, Any]] = []

    # Updated output columns to include F0_Value and reflect dFF0
    output_columns = [
        'Source_File', 'Block_Name_Background', 'Block_Name_Activation', 'WellID', 'F0_Value',
        'Increasing_Slope_dFF0', 'Increasing_R_Squared_dFF0', 
        'Decreasing_Slope_dFF0', 'Decreasing_R_Squared_dFF0',
        'Peak_DeltaF_F0_Index', 'Comment'
    ]

    if act_df_with_dFF0.empty or 'Time_sec' not in act_df_with_dFF0.columns:
        # print(f"Warning: Skipping activation block from {act_metadata.get('source_filename')} - empty or no 'Time_sec' column.")
        return pd.DataFrame(columns=output_columns)

    # Identify well columns from the activation DataFrame (which contains dFF0 data)
    well_ids = identify_well_columns(act_df_with_dFF0, non_well_cols=custom_non_well_columns)

    if not well_ids:
        # print(f"Warning: No well columns identified in activation block from {act_metadata.get('source_filename')}.")
        return pd.DataFrame(columns=output_columns)

    for well_id in well_ids:
        if well_id not in act_df_with_dFF0.columns or act_df_with_dFF0[well_id].isnull().all(): # type: ignore[no-untyped-call]
            continue

        time_data: pd.Series[float] = act_df_with_dFF0['Time_sec'] 
        # This data is already dF/F0
        dFF0_data_intermediate: pd.Series = act_df_with_dFF0[well_id] # type: ignore[assignment]
        dFF0_data_clean: pd.Series[float] = dFF0_data_intermediate.astype(float) # type: ignore[assignment]

        # Combine dFF0 data with time and drop rows where either is NaN
        combined_well_data = pd.DataFrame({'time': time_data, 'dFF0': dFF0_data_clean}).dropna() # type: ignore[call-arg]
        if combined_well_data.empty:
            continue
        
        time_data_clean_for_kinetics: pd.Series[float] = combined_well_data['time'].reset_index(drop=True).astype(float)
        dFF0_data_clean_for_kinetics: pd.Series[float] = combined_well_data['dFF0'].reset_index(drop=True).astype(float)
        
        f0_value_for_well = f0_map.get(well_id) # Get F0 for this well

        kinetic_params: ActivationKineticResults = calculate_activation_kinetics(
            time_sec=time_data_clean_for_kinetics,
            delta_f_over_f0=dFF0_data_clean_for_kinetics, 
            min_points_for_analysis=min_points_for_analysis,
            min_points_per_phase=min_points_per_phase
        )

        result_row: Dict[str, Any] = {
            'Source_File': act_metadata.get('source_filename', 'N/A'),
            'Block_Name_Background': bg_metadata.get('block_name', 'N/A'),
            'Block_Name_Activation': act_metadata.get('block_name', 'N/A'),
            'WellID': well_id,
            'F0_Value': f0_value_for_well if f0_value_for_well is not None else float('nan'),
            'Increasing_Slope_dFF0': kinetic_params['increasing_slope_dFF0'],
            'Increasing_R_Squared_dFF0': kinetic_params['increasing_r_squared_dFF0'],
            'Decreasing_Slope_dFF0': kinetic_params['decreasing_slope_dFF0'],
            'Decreasing_R_Squared_dFF0': kinetic_params['decreasing_r_squared_dFF0'],
            'Peak_DeltaF_F0_Index': kinetic_params['peak_delta_f_f0_index'],
            'Comment': kinetic_params['comment']
        }
        all_results_list.append(result_row)

        # Call plotting function if output_plot_dir is provided
        # plot_well_kinetics_with_fits will need to be updated to handle dFF0 data appropriately
        if output_plot_dir:
            source_file_for_plot = act_metadata.get('source_filename', 'UnknownFile')
            block_name_for_plot = act_metadata.get('block_name', 'UnknownBlock')
            
            plot_well_kinetics_with_fits(
                time_sec=time_data_clean_for_kinetics, 
                fluorescence=dFF0_data_clean_for_kinetics, # Passing dFF0 data here
                analysis_results=kinetic_params, 
                well_id=well_id,
                source_file=source_file_for_plot,
                block_name=block_name_for_plot,
                output_dir=output_plot_dir,
                min_points_per_phase=min_points_per_phase,
                is_dFF0_data=True # Add a flag to indicate dFF0 data
            )

    if not all_results_list:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame(all_results_list)

def save_results_to_csv(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the results DataFrame to a CSV file.

    Args:
        results_df: Pandas DataFrame containing the analysis results.
        output_path: The full path (including filename.csv) where the CSV will be saved.

    Raises:
        IOError: If there is an issue writing the file.
    """
    try:
        results_df.to_csv(output_path, index=False)
        print(f"Results successfully saved to {output_path}")
    except IOError as e:
        print(f"Error saving results to {output_path}: {e}")
        # Depending on desired behavior, you might want to re-raise or handle differently
        raise # Re-raise the exception to make the caller aware 