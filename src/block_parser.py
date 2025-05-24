import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

# Define a type alias for the metadata dictionary for clarity
MetadataDict = Dict[str, Any]

def parse_block(
    source_filename: str, 
    raw_block_df: pd.DataFrame,
    block_name_keyword_prefix: str = "plate:" # New parameter
) -> Tuple[MetadataDict, Optional[pd.DataFrame]]:
    """
    Parses a single raw block DataFrame to extract metadata and a cleaned data table.

    Args:
        source_filename: The original filename from which the block was extracted.
        raw_block_df: The DataFrame representing a single raw block, including
                      potential metadata lines and the data table.
        block_name_keyword_prefix: The lowercase string prefix to identify the row
                                     containing the block name (e.g., "plate:").

    Returns:
        A tuple containing:
            - metadata_dict: A dictionary of extracted metadata.
            - cleaned_tabular_df: A pandas DataFrame of the cleaned tabular data.
                                  Returns None if the main data table cannot be identified
                                  or is empty.
    """
    metadata: MetadataDict = {"source_filename": source_filename, "parsing_warnings": []}
    cleaned_df: Optional[pd.DataFrame] = None
    block_name_found: bool = False # Flag to track if block_name was identified

    if raw_block_df.empty:
        metadata["parsing_comment"] = "Raw block DataFrame was empty."
        metadata["failed_raw_block_shape"] = raw_block_df.shape # Should be (0,0) or similar
        metadata["failed_raw_block_preview"] = "Raw block was empty."
        # If parsing_warnings is empty, pop it.
        if not metadata.get("parsing_warnings"): # This will be true if it's an empty list
            metadata.pop("parsing_warnings", None)
        return metadata, None

    # --- Metadata Extraction (Step 2.2) ---
    possible_metadata_rows: int = min(5, len(raw_block_df)) # Check up to first 5 rows
    data_table_start_row_idx: Optional[int] = None # Keep track for potential future use

    # Ensure the prefix is lowercase for case-insensitive comparison
    normalized_block_name_prefix = block_name_keyword_prefix.lower()

    for i in range(possible_metadata_rows):
        # Use pd.isna() for robust missing value check in pandas/numpy context
        if pd.isna(raw_block_df.iloc[i, 0]):
            metadata["parsing_warnings"].append(f"Row {i}, Cell 0: Is empty, skipped for metadata.")
            continue # Skip empty cells

        first_cell_value: str = str(raw_block_df.iloc[i, 0]).strip()

        if first_cell_value.lower().startswith(normalized_block_name_prefix):
            name_part_after_colon = first_cell_value.split(":", 1)[-1].strip()
            
            if name_part_after_colon: # Name is in the same cell after the colon
                metadata["block_name"] = name_part_after_colon
                block_name_found = True
            elif raw_block_df.shape[1] > 1: # Check if there's a second cell in the row
                # Try to get block name from the next cell in the same row
                second_cell_value_obj = raw_block_df.iloc[i, 1]
                if pd.notna(second_cell_value_obj):
                    second_cell_value = str(second_cell_value_obj).strip()
                    if second_cell_value: # Ensure it's not empty
                        metadata["block_name"] = second_cell_value
                        block_name_found = True
                        metadata["parsing_warnings"].append(f"Row {i}: Used Cell 1 ('{second_cell_value}') for block name after Cell 0 prefix '{normalized_block_name_prefix}' had empty name part.")
                    else:
                        metadata["parsing_warnings"].append(f"Row {i}, Cell 0: Matched prefix '{normalized_block_name_prefix}'. Cell 1 also empty/invalid for block name.")
                else:
                    metadata["parsing_warnings"].append(f"Row {i}, Cell 0: Matched prefix '{normalized_block_name_prefix}'. Cell 1 is NaN/empty.")
            else: # No second cell to check
                metadata["parsing_warnings"].append(f"Row {i}, Cell 0: Matched prefix '{normalized_block_name_prefix}' but name part empty and no second cell in row.")
        elif first_cell_value.lower().startswith("measurements:"):
            try:
                metadata["measurements"] = int(first_cell_value.split(":", 1)[-1].strip())
            except ValueError:
                metadata["parsing_warnings"].append(f"Could not parse measurements from: {first_cell_value}")
        else:
            # Only add to warnings if block_name hasn't been found yet and this row was checked for it
            if not block_name_found:
                 metadata["parsing_warnings"].append(f"Row {i}, Cell 0: Value '{first_cell_value}' did not match prefix '{normalized_block_name_prefix}' or other known metadata prefixes.")
        # Add more metadata parsing rules here if needed

    # --- Data Table Parsing (Step 2.3) ---
    header_row_idx: Optional[int] = None
    for i in range(len(raw_block_df)):
        # raw_block_df.iloc[i].values can contain mixed types, str(x) is a general approach
        row_values: List[str] = [str(x).lower() for x in raw_block_df.iloc[i].values if pd.notna(x)] # type: ignore
        if "time" in row_values: 
            header_row_idx = i
            data_table_start_row_idx = i + 1 
            break
    
    if header_row_idx is None:
        metadata["parsing_comment"] = "Could not find data table header (e.g., 'Time' row)."
        if not block_name_found: # Check if block_name was not found during metadata scan
            metadata["parsing_error"] = "Essential metadata (block_name) and data table header not found."
            # Add collected warnings about why block_name might be missing
            metadata["parsing_comment"] = metadata.get("parsing_comment", "") + \
                                           " Potential reasons for missing block_name can be found in 'parsing_warnings'."
        # Add details about the raw block that failed to parse fully
        metadata["failed_raw_block_shape"] = raw_block_df.shape
        try:
            # Get first 3 rows as a string representation for preview
            metadata["failed_raw_block_preview"] = raw_block_df.head(3).to_string() # type: ignore
        except Exception:
            metadata["failed_raw_block_preview"] = "Error converting raw block head to string."
        return metadata, None

    if data_table_start_row_idx is not None and data_table_start_row_idx < len(raw_block_df):
        current_cleaned_df = raw_block_df.iloc[data_table_start_row_idx:].copy()
        current_cleaned_df.columns = raw_block_df.iloc[header_row_idx].values # type: ignore
        current_cleaned_df.reset_index(drop=True, inplace=True)
        
        current_cleaned_df.dropna(axis=1, how='all', inplace=True) # type: ignore
        current_cleaned_df.dropna(axis=0, how='all', inplace=True) # type: ignore

        if current_cleaned_df.empty:
            metadata["parsing_comment"] = metadata.get("parsing_comment", "") + " Extracted data table is empty."
            cleaned_df = None 
        else:
            cleaned_df = current_cleaned_df
    else:
        if "parsing_comment" not in metadata: 
             metadata["parsing_comment"] = "Data table start row not found or no data after header."
        # cleaned_df remains None if no valid data table segment is found

    # If no warnings were added, remove the empty list
    if not metadata.get("parsing_warnings"):
        metadata.pop("parsing_warnings", None)
    # If block_name was found, remove the specific warnings about failing to find it.
    elif block_name_found:
        # Filter out warnings that were specifically about not finding the block name prefix
        # This is a bit more complex to do selectively, so for now, if block_name is found,
        # we assume the user is satisfied with that part.
        # A more sophisticated approach might tag warnings by type.
        # For simplicity now: if block_name is found, let's assume parsing_warnings related to its absence are less critical.
        # However, other warnings (like for measurements) should persist.
        # This needs careful thought; let's keep all warnings for now and let user inspect.
        pass

    # If cleaned_df is still None here, it means parsing failed to produce a data table
    if cleaned_df is None and "parsing_comment" not in metadata and "parsing_error" not in metadata:
        metadata["parsing_comment"] = "Failed to extract a clean data table for unknown reasons."

    return metadata, cleaned_df 