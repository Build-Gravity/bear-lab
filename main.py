from __future__ import annotations
import os
import datetime
from typing import List, Optional, Tuple
import pandas as pd

from src.file_ingestion import get_all_raw_blocks_with_source
from src.processing import (
    process_all_blocks,
    prepare_paired_blocks_for_analysis,
    analyze_block_kinetics,
    save_results_to_csv,
    ParsingAttemptResult,
    PairedBlockForAnalysis,
    DEFAULT_F0_POINTS_TO_AVERAGE,
    DEFAULT_MIN_POINTS_FOR_ANALYSIS,
    DEFAULT_MIN_POINTS_PER_PHASE
)

# --- Configuration Constants ---
INPUT_DIR: str = "input_data"
OUTPUT_CSV_DIR: str = "output_data"
OUTPUT_PLOT_DIR: str = "output_plots"

# Keywords for block identification
BACKGROUND_KEYWORD: str = "background"
ACTIVATION_KEYWORD: str = "activation"

# Parameters for F0 and kinetic analysis (can be adjusted)
F0_POINTS_TO_AVERAGE: int = DEFAULT_F0_POINTS_TO_AVERAGE
MIN_POINTS_FOR_KINETIC_ANALYSIS: int = DEFAULT_MIN_POINTS_FOR_ANALYSIS
MIN_POINTS_PER_KINETIC_PHASE: int = DEFAULT_MIN_POINTS_PER_PHASE

# Columns to consistently ignore during well identification in various stages
# These are added to internal defaults like 'Time', 'Time_sec'
# Add columns like 'Temperature(¡C)' if they appear and are not wells
NON_WELL_COLUMNS_CONFIG: Optional[List[str]] = ['Temperature(¡C)', 'Temperature (°C)']

# Define type alias consistent with file_ingestion.py's output
RawBlockWithSource = Tuple[str, pd.DataFrame]


def ensure_dir_exists(dir_path: str) -> None:
    """Checks if a directory exists, and creates it if it doesn't."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")
            raise # Re-raise the error to halt execution if critical dirs can't be made

def main() -> None:
    """
    Main function to orchestrate the CFTR assay data analysis workflow.
    """
    print("Starting CFTR data analysis workflow...")

    # Ensure output directories exist
    ensure_dir_exists(OUTPUT_CSV_DIR)
    ensure_dir_exists(OUTPUT_PLOT_DIR)

    # --- Part 1: File Ingestion ---
    print(f"Step 1: Identifying XLSX files in '{INPUT_DIR}'...")
    try:
        # xlsx_files: List[str] = find_xlsx_files(INPUT_DIR) # find_xlsx_files is called within get_all_raw_blocks_with_source
        # if not xlsx_files:
        #     print(f"No XLSX files found in '{INPUT_DIR}'. Exiting.")
        #     return
        # print(f"Found {len(xlsx_files)} XLSX files to process.")

        # get_all_raw_blocks_with_source finds files and segments them.
        raw_blocks_with_sources: List[RawBlockWithSource] = get_all_raw_blocks_with_source(INPUT_DIR)
        if not raw_blocks_with_sources:
            print(f"No raw blocks could be segmented from files in '{INPUT_DIR}'. Exiting.")
            return
        print(f"Successfully segmented {len(raw_blocks_with_sources)} raw blocks from all files.")

    except Exception as e:
        print(f"Error during file ingestion (Step 1): {e}")
        return

    # --- Part 2 & 3: Block Parsing and Preparation for dF/F0 ---
    print("Step 2&3: Parsing blocks and preparing for dF/F0 analysis...")
    try:
        # process_all_blocks expects List[Tuple[str, pd.DataFrame]] which matches RawBlockWithSource
        all_parsing_attempts: List[ParsingAttemptResult] = process_all_blocks(raw_blocks_with_sources)
        
        # prepare_paired_blocks_for_analysis filters for successfully parsed blocks internally
        paired_blocks_for_dFF0: List[PairedBlockForAnalysis] = prepare_paired_blocks_for_analysis(
            parsed_blocks_results=all_parsing_attempts,
            background_keyword=BACKGROUND_KEYWORD,
            activation_keyword=ACTIVATION_KEYWORD,
            f0_points_to_average=F0_POINTS_TO_AVERAGE,
            non_well_columns_for_f0=NON_WELL_COLUMNS_CONFIG,
            non_well_columns_for_activation=NON_WELL_COLUMNS_CONFIG,
            time_column_name="Time" # Default, can be configured if needed
        )

        if not paired_blocks_for_dFF0:
            print("No background/activation block pairs suitable for dF/F0 analysis were found. Exiting.")
            return
        print(f"Prepared {len(paired_blocks_for_dFF0)} paired blocks for kinetic analysis (dF/F0 calculated).")

    except Exception as e:
        print(f"Error during block parsing or dF/F0 preparation (Step 2&3): {e}")
        return

    # --- Part 4: Kinetic Analysis of Activation Blocks (using dF/F0) ---
    print("Step 4: Performing kinetic analysis on activation blocks...")
    all_results_dfs: List[pd.DataFrame] = []
    try:
        for paired_block_item in paired_blocks_for_dFF0:
            # analyze_block_kinetics now takes PairedBlockForAnalysis
            # and its DataFrame for activation block already has dF/F0
            results_df_single_block = analyze_block_kinetics(
                paired_block_for_analysis=paired_block_item,
                min_points_for_analysis=MIN_POINTS_FOR_KINETIC_ANALYSIS,
                min_points_per_phase=MIN_POINTS_PER_KINETIC_PHASE,
                custom_non_well_columns=NON_WELL_COLUMNS_CONFIG,
                output_plot_dir=OUTPUT_PLOT_DIR # Pass plot directory
            )
            if not results_df_single_block.empty:
                all_results_dfs.append(results_df_single_block)
        
        if not all_results_dfs:
            print("No kinetic results generated from any activation block. Check data and pairing.")
            # No CSV to save, but workflow completed up to this point.
        else:
            print(f"Kinetic analysis completed for {len(all_results_dfs)} activation block(s).")
    
    except Exception as e:
        print(f"Error during kinetic analysis (Step 4): {e}")
        # Optionally, one might still try to save partial results if any were collected
        # For now, we exit if analysis fails.
        return

    # --- Part 5: Output Results ---
    if not all_results_dfs:
        print("No results to save. Workflow finished.")
        return

    print("Step 5: Compiling and saving results...")
    try:
        final_results_df = pd.concat(all_results_dfs, ignore_index=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_filename = f"cftr_activation_dFF0_slopes_{timestamp}.csv"
        output_csv_path = os.path.join(OUTPUT_CSV_DIR, output_csv_filename)
        
        save_results_to_csv(final_results_df, output_csv_path)
        # save_results_to_csv already prints success or error

    except Exception as e:
        print(f"Error during results compilation or saving (Step 5): {e}")
        return

    print("CFTR data analysis workflow completed successfully!")

if __name__ == "__main__":
    main() 