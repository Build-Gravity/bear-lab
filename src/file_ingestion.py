import os
import glob
from typing import List, Tuple
import pandas as pd

def find_xlsx_files(directory_path: str) -> List[str]:
    """
    Finds all .xlsx files in the specified directory (non-recursive).

    Args:
        directory_path: The path to the directory to search.

    Returns:
        A list of absolute paths to .xlsx files. 
        Returns an empty list if the directory doesn't exist or no .xlsx files are found.
    """
    if not os.path.isdir(directory_path):
        # print(f"Warning: Directory not found: {directory_path}") # Optional warning
        return []
    
    # Construct the search pattern for .xlsx files in the given directory
    # os.path.join ensures the path is constructed correctly for the OS
    # glob.glob performs the file search
    search_pattern: str = os.path.join(directory_path, "*.xlsx")
    # On Windows, glob can be case-insensitive. We need to filter for exact '.xlsx'.
    # A more robust way for all OS, though glob("*.xlsx") is often sufficient on Unix-like for case-sensitivity.
    all_matching_glob: List[str] = glob.glob(search_pattern)
    
    # Filter for exact lowercase ".xlsx" extension
    xlsx_files: List[str] = [f for f in all_matching_glob if f.endswith(".xlsx")]
    
    # Convert to absolute paths
    return [os.path.abspath(f) for f in xlsx_files]

def extract_raw_blocks_from_file(file_path: str) -> List[pd.DataFrame]:
    """
    Reads an XLSX file and segments its first sheet into raw experimental blocks.

    Blocks are assumed to be separated by a row where the first cell contains '~End'.

    Args:
        file_path: The absolute path to the .xlsx file.

    Returns:
        A list of pandas DataFrames, where each DataFrame represents a raw block.
        Returns an empty list if the file cannot be read, the sheet is empty,
        or no blocks are found.
    """
    try:
        df_sheet: pd.DataFrame = pd.read_excel(file_path, header=None, sheet_name=0) # type: ignore
    except FileNotFoundError:
        # print(f"Error: File not found: {file_path}") # Consider logging
        return []
    except ValueError as _e: # Renamed e to _e
        # print(f"Error: Could not read sheet in {file_path}. Might be empty or not an Excel file. {_e}")
        return []
    except Exception as _e: # Renamed e to _e
        # print(f"Error reading {file_path}: {_e}") # Consider logging
        return []

    if df_sheet.empty:
        return []

    raw_blocks: List[pd.DataFrame] = []
    current_block_start_index: int = 0
    # Ensure the first column exists before trying to access iloc[:, 0]
    if df_sheet.shape[1] == 0: # No columns
        return []

    for i, row in df_sheet.iterrows(): # type: ignore
        # Check if the first cell of the row contains '~End'
        # Ensure the cell value is a string before checking
        first_cell_value: any = row.iloc[0] # type: ignore
        if isinstance(first_cell_value, str) and first_cell_value.strip() == "~End":
            # i from iterrows is the index, which can be used directly for slicing
            if i > current_block_start_index: # type: ignore[operator]
                raw_blocks.append(df_sheet.iloc[current_block_start_index:i].reset_index(drop=True))
            current_block_start_index = i + 1 # type: ignore[operator]
    
    # Add the last block if it doesn't end with '~End' or if it's the only block
    if current_block_start_index < len(df_sheet):
        potential_last_block: pd.DataFrame = df_sheet.iloc[current_block_start_index:].reset_index(drop=True)
        # Heuristic: Check if this potential last block is just a single metadata row
        if not (
            len(potential_last_block) == 1 and 
            potential_last_block.shape[1] > 0 and # Ensure there is at least one column
            isinstance(potential_last_block.iloc[0, 0], str) and \
            "Original Filename:" in potential_last_block.iloc[0, 0] # type: ignore[operator]
        ):
            raw_blocks.append(potential_last_block)
        # else: This block looks like a trailing metadata row, so we skip it.
    
    return raw_blocks

def get_all_raw_blocks_with_source(directory_path: str) -> List[Tuple[str, pd.DataFrame]]:
    """
    Finds all .xlsx files in a directory, extracts raw blocks from each, 
    and returns a list associating each block with its source filename.

    Args:
        directory_path: The path to the directory containing .xlsx files.

    Returns:
        A list of tuples: (source_filename, raw_block_dataframe).
        source_filename is the basename of the file.
    """
    xlsx_file_paths: List[str] = find_xlsx_files(directory_path)
    all_blocks_with_source: List[Tuple[str, pd.DataFrame]] = []

    for file_path in xlsx_file_paths:
        raw_blocks: List[pd.DataFrame] = extract_raw_blocks_from_file(file_path)
        file_name: str = os.path.basename(file_path)
        for block_df in raw_blocks:
            all_blocks_with_source.append((file_name, block_df))
            
    return all_blocks_with_source 