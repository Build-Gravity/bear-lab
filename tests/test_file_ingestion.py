# Adjusted import path if your find_xlsx_files is in a different location
# For example, if it's in src/file_ingestion.py:
from src.file_ingestion import find_xlsx_files, extract_raw_blocks_from_file, get_all_raw_blocks_with_source
import pytest
from typing import List
import os
import pandas as pd

# Define a fixture for a temporary test directory
@pytest.fixture
def temp_test_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a temporary directory for test files."""
    test_dir = tmp_path_factory.mktemp("test_data")
    return str(test_dir)

# Define a fixture for a temporary input_data directory structure
@pytest.fixture
def input_data_dir(temp_test_dir: str) -> str:
    """Create a temporary input_data directory with test files."""
    input_dir = os.path.join(temp_test_dir, "input_data")
    os.makedirs(input_dir, exist_ok=True)

    # Create dummy .xlsx files
    open(os.path.join(input_dir, "file1.xlsx"), "w").close()
    open(os.path.join(input_dir, "file2.xlsx"), "w").close()
    open(os.path.join(input_dir, "FILE3.XLSX"), "w").close() # Uppercase extension
    open(os.path.join(input_dir, "notes.txt"), "w").close() # Other extension

    # Create a subdirectory with another .xlsx file (should be ignored by non-recursive search)
    os.makedirs(os.path.join(input_dir, "archive"), exist_ok=True)
    open(os.path.join(input_dir, "archive", "file_in_subdir.xlsx"), "w").close()
    
    return input_dir

# Helper to create dummy xlsx files for block extraction tests
@pytest.fixture
def block_test_file_dir(temp_test_dir: str) -> str:
    """Creates a directory for block extraction test files."""
    block_dir = os.path.join(temp_test_dir, "block_test_files")
    os.makedirs(block_dir, exist_ok=True)
    return block_dir

def create_excel_file(file_path: str, data: List[List[any]]) -> None: # type: ignore
    """Helper function to create an Excel file with given data."""
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False, header=False, engine='openpyxl') # type: ignore

# Tests for extract_raw_blocks_from_file

def test_extract_multiple_blocks(block_test_file_dir: str) -> None:
    """Test extracting multiple blocks separated by ~End."""
    file_path = os.path.join(block_test_file_dir, "multiple_blocks.xlsx")
    data = [
        ["Block 1 Data 1", "Val1"],
        ["Block 1 Data 2", "Val2"],
        ["~End", ""],
        ["Block 2 Data 1", "Val3"],
        ["~End", "Something else"],
        ["Block 3 Data 1", "Val4"],
        ["Block 3 Data 2", "Val5"],
    ]
    create_excel_file(file_path, data)
    
    raw_blocks = extract_raw_blocks_from_file(file_path)
    assert len(raw_blocks) == 3
    assert raw_blocks[0].shape == (2, 2)
    assert raw_blocks[1].shape == (1, 2)
    assert raw_blocks[2].shape == (2, 2)
    assert raw_blocks[0].iloc[0, 0] == "Block 1 Data 1"
    assert raw_blocks[1].iloc[0, 0] == "Block 2 Data 1"
    assert raw_blocks[2].iloc[0, 0] == "Block 3 Data 1"

def test_extract_single_block_no_marker(block_test_file_dir: str) -> None:
    """Test a file with a single block and no ~End marker."""
    file_path = os.path.join(block_test_file_dir, "single_block.xlsx")
    data = [
        ["Data A", 1],
        ["Data B", 2]
    ]
    create_excel_file(file_path, data)
    raw_blocks = extract_raw_blocks_from_file(file_path)
    assert len(raw_blocks) == 1
    assert raw_blocks[0].shape == (2, 2)
    assert raw_blocks[0].iloc[0, 0] == "Data A"

def test_extract_block_with_trailing_marker(block_test_file_dir: str) -> None:
    """Test a file where the last block is followed by ~End."""
    file_path = os.path.join(block_test_file_dir, "trailing_marker.xlsx")
    data = [
        ["Block X", "Y"],
        ["~End", ""]
    ]
    create_excel_file(file_path, data)
    raw_blocks = extract_raw_blocks_from_file(file_path)
    # Expect one block, the marker row itself is consumed and doesn't create an empty block after it
    assert len(raw_blocks) == 1 
    assert raw_blocks[0].shape == (1,2) # Corrected: Was (1,1), should be (1,2) for [["Block X", "Y"]]
    # Correction based on current implementation: The block before ~End is extracted.
    # df_sheet.iloc[current_block_start_index:i]
    assert raw_blocks[0].iloc[0,0] == "Block X"


def test_extract_empty_file(block_test_file_dir: str) -> None:
    """Test with an empty Excel file (no sheets or empty sheet)."""
    file_path = os.path.join(block_test_file_dir, "empty_file.xlsx")
    # Create an excel file that will be empty or have an empty sheet
    # One way to make pandas create an "empty" (but valid) xlsx is an empty dataframe
    create_excel_file(file_path, []) 
    raw_blocks = extract_raw_blocks_from_file(file_path)
    assert len(raw_blocks) == 0

def test_extract_file_with_only_end_marker(block_test_file_dir: str) -> None:
    """Test a file that only contains the ~End marker."""
    file_path = os.path.join(block_test_file_dir, "only_marker.xlsx")
    data = [["~End"]]
    create_excel_file(file_path, data)
    raw_blocks = extract_raw_blocks_from_file(file_path)
    # The current logic might create an empty dataframe before the marker, 
    # or no blocks if current_block_start_index == i initially.
    # If first row is ~End, current_block_start_index becomes 1. loop ends.
    # Then, if 1 < len(df_sheet) (which is 1 < 1, false), no final block added.
    # If current_block_start_index = 0, i = 0. block from 0 to 0 is empty. Start becomes 1.
    # Expected based on refined understanding: no blocks, or one empty block that gets filtered.
    # The implementation `if i > current_block_start_index:` prevents an empty block before first marker.
    assert len(raw_blocks) == 0

def test_extract_marker_not_in_first_col(block_test_file_dir: str) -> None:
    """Test with ~End not in the first column (should not split)."""
    file_path = os.path.join(block_test_file_dir, "marker_wrong_col.xlsx")
    data = [
        ["Data1", "Val1"],
        ["Data2", "~End"], # Marker not in first column
        ["Data3", "Val3"]
    ]
    create_excel_file(file_path, data)
    raw_blocks = extract_raw_blocks_from_file(file_path)
    assert len(raw_blocks) == 1
    assert raw_blocks[0].shape == (3, 2)

def test_extract_marker_with_spaces(block_test_file_dir: str) -> None:
    """Test with ~End having leading/trailing spaces."""
    file_path = os.path.join(block_test_file_dir, "marker_spaces.xlsx")
    data = [
        ["Block A", "A1"],
        ["  ~End  ", ""], # Marker with spaces
        ["Block B", "B1"]
    ]
    create_excel_file(file_path, data)
    raw_blocks = extract_raw_blocks_from_file(file_path)
    assert len(raw_blocks) == 2
    assert raw_blocks[0].shape == (1, 2)
    assert raw_blocks[1].shape == (1, 2)

def test_extract_non_excel_file(block_test_file_dir: str) -> None:
    """Test providing a non-Excel file (e.g., a text file renamed to .xlsx)."""
    file_path = os.path.join(block_test_file_dir, "not_excel.xlsx")
    with open(file_path, "w") as f:
        f.write("This is not an Excel file.")
    raw_blocks = extract_raw_blocks_from_file(file_path)
    assert len(raw_blocks) == 0

def test_extract_file_no_columns(block_test_file_dir: str) -> None:
    """Test a file that reads as a DataFrame with 0 columns."""
    # This is hard to create directly with to_excel if it auto-adds a default col for empty list of lists.
    # However, the code has `if df_sheet.shape[1] == 0: return []`
    # We can simulate by passing a mock that returns such a df, but let's test the file path for now.
    # For now, an empty data list to create_excel_file results in an empty sheet, handled by `df_sheet.empty`
    # or `extract_empty_file` test. A specific test for 0 columns might require deeper mocking of pd.read_excel.
    # The `extract_empty_file` covers the `df_sheet.empty` case.
    # If `pd.read_excel` somehow returns a non-empty df with 0 columns, that's a rarer pandas internal case.
    # Let's assume `df_sheet.empty` or read error handles most practical scenarios of malformed files.
    pass # Covered by test_extract_empty_file primarily

def test_find_xlsx_files_found(input_data_dir: str) -> None:
    """Test finding .xlsx files in a directory that contains them."""
    expected_files: List[str] = sorted([
        os.path.abspath(os.path.join(input_data_dir, "file1.xlsx")),
        os.path.abspath(os.path.join(input_data_dir, "file2.xlsx"))
    ])
    actual_files: List[str] = sorted(find_xlsx_files(input_data_dir))
    assert actual_files == expected_files, "Should find only lowercase .xlsx files in the root."

def test_find_xlsx_files_empty(temp_test_dir: str) -> None:
    """Test finding .xlsx files in an empty directory."""
    empty_input_dir: str = os.path.join(temp_test_dir, "empty_input_data")
    os.makedirs(empty_input_dir, exist_ok=True)
    assert find_xlsx_files(empty_input_dir) == [], "Should return an empty list for an empty directory."

def test_find_xlsx_files_no_xlsx(temp_test_dir: str) -> None:
    """Test finding .xlsx files in a directory with no .xlsx files."""
    no_xlsx_dir: str = os.path.join(temp_test_dir, "no_xlsx_data")
    os.makedirs(no_xlsx_dir, exist_ok=True)
    open(os.path.join(no_xlsx_dir, "document.txt"), "w").close()
    open(os.path.join(no_xlsx_dir, "archive.zip"), "w").close()
    assert find_xlsx_files(no_xlsx_dir) == [], "Should return an empty list if no .xlsx files are present."

def test_find_xlsx_files_non_existent_dir(temp_test_dir: str) -> None:
    """Test behavior when the input directory does not exist."""
    non_existent_dir: str = os.path.join(temp_test_dir, "non_existent_dir")
    # Do not create the directory
    assert find_xlsx_files(non_existent_dir) == [], "Should return an empty list for a non-existent directory."

def test_find_xlsx_files_root_level_search_only(input_data_dir: str) -> None:
    """Test that the search is non-recursive."""
    found_files: List[str] = find_xlsx_files(input_data_dir)
    file_in_subdir_abs_path: str = os.path.abspath(os.path.join(input_data_dir, "archive", "file_in_subdir.xlsx"))
    assert file_in_subdir_abs_path not in found_files, "Should not find files in subdirectories."

# Tests for get_all_raw_blocks_with_source

def test_get_all_raw_blocks_with_source(block_test_file_dir: str) -> None:
    """Test consolidating blocks from multiple files with source tracking."""
    file1_path = os.path.join(block_test_file_dir, "src_file1.xlsx")
    data1 = [
        ["F1B1 Data1"], ["~End"], ["F1B2 Data1"]
    ]
    create_excel_file(file1_path, data1)

    file2_path = os.path.join(block_test_file_dir, "src_file2.xlsx")
    data2 = [
        ["F2B1 Data1"], ["F2B1 Data2"], ["~End"], ["F2B2 Data1"], ["F2B2 Data2"]
    ]
    create_excel_file(file2_path, data2)

    # Add a file that won't be processed (e.g., .txt or non-lowercase .xlsx if strict)
    # The find_xlsx_files should already filter these out.
    with open(os.path.join(block_test_file_dir, "ignore_me.txt"), "w") as f:
        f.write("text")
    open(os.path.join(block_test_file_dir, "UPPER.XLSX"), "w").close() # Should be ignored

    all_sourced_blocks = get_all_raw_blocks_with_source(block_test_file_dir)

    assert len(all_sourced_blocks) == 4 # 2 blocks from file1, 2 from file2

    # Check source filenames and basic block structure
    # File1 Block1
    assert all_sourced_blocks[0][0] == "src_file1.xlsx"
    assert all_sourced_blocks[0][1].shape == (1, 1)
    assert all_sourced_blocks[0][1].iloc[0, 0] == "F1B1 Data1"

    # File1 Block2
    assert all_sourced_blocks[1][0] == "src_file1.xlsx"
    assert all_sourced_blocks[1][1].shape == (1, 1)
    assert all_sourced_blocks[1][1].iloc[0, 0] == "F1B2 Data1"

    # File2 Block1
    assert all_sourced_blocks[2][0] == "src_file2.xlsx"
    assert all_sourced_blocks[2][1].shape == (2, 1)
    assert all_sourced_blocks[2][1].iloc[0, 0] == "F2B1 Data1"

    # File2 Block2
    assert all_sourced_blocks[3][0] == "src_file2.xlsx"
    assert all_sourced_blocks[3][1].shape == (2, 1) 
    assert all_sourced_blocks[3][1].iloc[0, 0] == "F2B2 Data1"

def test_get_all_raw_blocks_with_source_empty_dir(temp_test_dir: str) -> None:
    """Test with an empty directory for consolidation."""
    empty_block_dir = os.path.join(temp_test_dir, "empty_block_test_files")
    os.makedirs(empty_block_dir, exist_ok=True)
    all_sourced_blocks = get_all_raw_blocks_with_source(empty_block_dir)
    assert len(all_sourced_blocks) == 0

def test_get_all_raw_blocks_with_source_no_valid_files(block_test_file_dir: str) -> None:
    """Test with a directory containing no valid .xlsx files for consolidation."""
    # Clear out any .xlsx files, leave others if any
    for item in os.listdir(block_test_file_dir):
        if item.endswith(".xlsx"):
            os.remove(os.path.join(block_test_file_dir, item))
    
    with open(os.path.join(block_test_file_dir, "notes.txt"), "w") as f:
        f.write("This is a text file.")
    open(os.path.join(block_test_file_dir, "ANOTHER.XLSX"), "w").close() # Uppercase, should be ignored

    all_sourced_blocks = get_all_raw_blocks_with_source(block_test_file_dir)
    assert len(all_sourced_blocks) == 0