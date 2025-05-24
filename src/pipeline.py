from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.file_ingestion import get_all_raw_blocks_with_source
from src.processing import (
    process_all_blocks,
    select_all_activation_blocks,
    prepare_time_column,
    ensure_numeric_well_data,
    calculate_activation_kinetics,
    identify_well_columns,
    RawBlock,
    ParsedBlockResult,
    ParsingAttemptResult,
    MetadataDict,
    ActivationKineticResults
)
from src.visualization import plot_well_kinetics_with_fits

# Default values for analysis parameters, can be overridden via function arguments
DEFAULT_MIN_POINTS_FOR_ANALYSIS: int = 4
DEFAULT_MIN_POINTS_PER_PHASE: int = 2


def _log_block_parsing_summary(parsed_blocks_list: List[ParsedBlockResult]) -> None:
    """
    Logs a summary of parsed block names and any parsing warnings.
    Helper function for execute_full_analysis.
    """
    print("\n--- Unique Block Names Found ---")
    unique_block_display_names: set[str] = set()
    all_metadata_warnings: Dict[str, List[str]] = {} # identifier -> warnings

    for i, (metadata, _) in enumerate(parsed_blocks_list):
        block_name: Optional[str] = metadata.get('block_name')
        warnings: Optional[List[str]] = metadata.get('parsing_warnings')
        
        # Create a display name and an identifier for warnings
        identifier: str
        display_name: str
        source_file: str = metadata.get('source_filename', 'unknown_file')

        if block_name:
            display_name = f"{block_name} (from {source_file})"
            identifier = display_name # For named blocks, display name is unique enough
        else:
            display_name = f"Unnamed Block #{i+1} (from {source_file})"
            identifier = display_name # This should be unique for unnamed blocks too

        unique_block_display_names.add(display_name)

        if warnings:
            all_metadata_warnings[identifier] = warnings
            
    if unique_block_display_names:
        for name in sorted(list(unique_block_display_names)):
            print(f"- {name}")
            if name in all_metadata_warnings: # Check if this display name has warnings
                print("  Parsing Warnings:")
                for warning in all_metadata_warnings[name]:
                    print(f"    * {warning}")
    else:
        print("No block names found in parsed metadata (or no blocks parsed at all).")
    print("-----------------------------")


def _log_failed_parsing_attempts(parsing_attempts: List[ParsingAttemptResult]) -> None:
    """
    Logs blocks that could not be parsed into a usable DataFrame.
    Helper function for execute_full_analysis.
    """
    failed_blocks_info: List[str] = []
    for i, (metadata, df) in enumerate(parsing_attempts):
        if df is None or df.empty:
            source_file: str = metadata.get('source_filename', 'Unknown file')
            block_name: Optional[str] = metadata.get('block_name')
            comment: Optional[str] = metadata.get('parsing_comment', 'No specific comment.')
            error: Optional[str] = metadata.get('parsing_error')
            shape_info: Optional[Tuple[int, int]] = metadata.get('failed_raw_block_shape')
            preview_info: Optional[str] = metadata.get('failed_raw_block_preview')

            identifier: str
            if block_name:
                identifier = f"Block '{block_name}' (from {source_file})"
            else:
                # Attempt to give a more stable identifier if raw block info was available
                # For now, use an index-based one as fallback
                identifier = f"Unnamed/Raw Block approx. index {i+1} (from {source_file})"
            
            reason: str = comment if comment else "Could not be parsed into a data table."
            if error:
                reason += f" Critical Error: {error}"
            
            details: str = f"- {identifier}: {reason}"
            if shape_info:
                details += f"\n    Raw Shape: {shape_info}"
            if preview_info:
                details += f"\n    Raw Preview (first 3 rows):\n{preview_info}\n"
            
            failed_blocks_info.append(details)

    if failed_blocks_info:
        print("\n--- Blocks That Failed Parsing ---")
        for info in failed_blocks_info:
            print(info)
        print("--------------------------------")
    else:
        print("\n--- Blocks That Failed Parsing ---")
        print("All raw blocks were successfully parsed into data tables.")
        print("--------------------------------")


def execute_full_analysis(
    input_dir: str,
    activation_keyword: str,
    block_name_prefix: str,
    output_plots_dir: Optional[str],
    min_points_for_analysis: int = DEFAULT_MIN_POINTS_FOR_ANALYSIS,
    min_points_per_phase: int = DEFAULT_MIN_POINTS_PER_PHASE,
    expected_kinetic_shape: str = 'u-shape' # Defaulting to 'u-shape'
) -> Optional[pd.DataFrame]:
    """
    Executes the full CFTR analysis pipeline.
    1. Ingests data from XLSX files.
    2. Processes all raw blocks.
    3. Logs a summary of parsed blocks.
    4. Selects the 'activation' block.
    5. Prepares its data (time column, numeric conversion).
    6. Identifies well columns for analysis.
    7. Calculates biphasic kinetics for each well.
    8. Compiles and returns results.
    """
    print(f"Starting analysis for data in: {input_dir}")

    # 1. Ingest data
    master_raw_blocks: List[RawBlock] = get_all_raw_blocks_with_source(input_dir)
    if not master_raw_blocks:
        print("No raw blocks found in any XLSX files. Exiting pipeline.")
        return None
    print(f"Found {len(master_raw_blocks)} raw blocks from {len(set(item[0] for item in master_raw_blocks))} files.")

    # 2. Process all blocks
    all_parsing_results: List[ParsingAttemptResult] = process_all_blocks(master_raw_blocks, block_name_prefix=block_name_prefix)
    
    # Separate successfully parsed blocks from failed attempts
    parsed_blocks_list: List[ParsedBlockResult] = []
    for metadata, df in all_parsing_results:
        if df is not None and not df.empty:
            parsed_blocks_list.append((metadata, df))
            
    if not parsed_blocks_list: # If NO blocks were successfully parsed
        print("No blocks were successfully parsed into usable data. Exiting pipeline.")
        _log_failed_parsing_attempts(all_parsing_results) # Log failures before exiting
        return None
        
    print(f"Successfully parsed {len(parsed_blocks_list)} blocks with data out of {len(all_parsing_results)} raw blocks attempted.")

    # 3. Log block parsing summary (for successfully parsed blocks)
    _log_block_parsing_summary(parsed_blocks_list)
    # Log blocks that failed parsing
    _log_failed_parsing_attempts(all_parsing_results)

    # 4. Select the activation block(s) from successfully parsed blocks
    all_activation_blocks: List[ParsedBlockResult] = select_all_activation_blocks(parsed_blocks_list, activation_keyword)
    
    if not all_activation_blocks:
        print(f"No blocks containing the keyword '{activation_keyword}' found. Exiting pipeline.")
        return None
    
    print(f"Found {len(all_activation_blocks)} activation block(s). Processing all of them.")

    master_well_results: List[Dict[str, Any]] = [] # Initialize a list to store results from ALL blocks

    for activation_block_data in all_activation_blocks:
        activation_metadata: MetadataDict = activation_block_data[0]
        activation_df: pd.DataFrame = activation_block_data[1]
        
        current_source_filename: str = activation_metadata.get('source_filename', 'Unknown File')
        current_block_name: str = activation_metadata.get('block_name', 'Unknown Block')

        print(f"Processing activation block: '{current_block_name}' from '{current_source_filename}'")

        # 5. Prepare the activation DataFrame for analysis
        prepared_df: pd.DataFrame = activation_df.copy() 
        prepared_df = prepare_time_column(prepared_df)
        if 'Time_sec' not in prepared_df.columns:
            print(f"Error: 'Time_sec' column not created for block '{current_block_name}' from '{current_source_filename}'. Skipping this block.")
            continue # Skip to the next activation block
        prepared_df = ensure_numeric_well_data(prepared_df)

        # 6. Identify well columns
        well_columns: List[str] = identify_well_columns(prepared_df)
        if not well_columns:
            print(f"No well columns identified for kinetic analysis in block '{current_block_name}' from '{current_source_filename}'. Skipping this block.")
            continue # Skip to the next activation block
        print(f"  Identified {len(well_columns)} well columns for analysis in this block: {', '.join(well_columns[:5])}{'...' if len(well_columns) > 5 else ''}")

        # 7. Calculate kinetics for each well in the current block
        current_block_well_results: List[Dict[str, Any]] = []
        for well_id in well_columns:
            time_series: pd.Series[float] = prepared_df['Time_sec']
            fluor_series: pd.Series[float] = prepared_df[well_id]

            kinetics_results: ActivationKineticResults = calculate_activation_kinetics(
                time_sec=time_series,
                delta_f_over_f0=fluor_series,
                min_points_for_analysis=min_points_for_analysis,
                min_points_per_phase=min_points_per_phase
            )
            
            full_kinetics_result: Dict[str, Any] = dict(kinetics_results) 
            full_kinetics_result['well_id'] = well_id
            # Use metadata from the current block in the loop
            full_kinetics_result['source_filename'] = current_source_filename
            full_kinetics_result['block_name'] = current_block_name
            current_block_well_results.append(full_kinetics_result)

            if output_plots_dir:
                plot_well_kinetics_with_fits(
                    time_sec=time_series, 
                    fluorescence=fluor_series,
                    analysis_results=kinetics_results,
                    well_id=well_id,
                    source_file=current_source_filename, # Use current source file
                    block_name=current_block_name,     # Use current block name
                    output_dir=output_plots_dir,
                    min_points_per_phase=min_points_per_phase,
                    is_dFF0_data=True # Indicate that fluorescence data is dF/F0
                )
        
        master_well_results.extend(current_block_well_results) # Add results from this block to the master list

    # 8. Compile results into a DataFrame
    if not master_well_results: # Check if the master list is empty
        print("No kinetic results generated for any well across all activation blocks. Exiting pipeline.")
        return None
        
    results_df: pd.DataFrame = pd.DataFrame(master_well_results)
    
    # Reorder columns for clarity
    cols_order: List[str] = [
        'source_filename', 'block_name', 'well_id', 
        'increasing_slope', 'increasing_r_squared',
        'decreasing_slope', 'decreasing_r_squared',
        'peak_fluorescence_index', 'comment'
    ]
    # Ensure all columns in cols_order exist in results_df, add missing ones as None
    for col in cols_order:
        if col not in results_df.columns:
            results_df[col] = pd.NA # Use pd.NA for missing values, more consistent
            
    results_df = results_df[cols_order]

    print("\n--- Pipeline Execution Summary ---")
    print(f"Total raw blocks ingested: {len(master_raw_blocks)}")
    print(f"Successfully parsed blocks: {len(parsed_blocks_list)}")
    if all_activation_blocks:
        print(f"Activation blocks found and processed: {len(all_activation_blocks)}")
    else: # Should not happen if we exited earlier, but as a safeguard
        print("Activation blocks found and processed: 0")

    if results_df.empty:
        print("No data in final results DataFrame.")
    else:
        print(f"Final results DataFrame contains {len(results_df)} rows.")
        # print(results_df.head()) # Optional: print head for quick check
    print("------------------------------------")
    
    return results_df 