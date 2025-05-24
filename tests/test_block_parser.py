import pandas as pd
import pytest
from typing import Dict, Any, Optional, Tuple

from src.block_parser import parse_block, MetadataDict

# Define a type alias for the expected return type of parse_block for clarity in tests
ExpectedParseResult = Tuple[Optional[MetadataDict], Optional[pd.DataFrame]]

def test_parse_block_returns_correct_types() -> None:
    """
    Tests that parse_block returns a tuple containing a metadata dictionary (or None)
    and a pandas DataFrame (or None).
    This is the basic structural test based on plan.md Step 2.1.
    """
    source_filename: str = "test_sample.xlsx"
    # Create a minimal, representative raw_block_df as described in plan.md
    # "Plate: Name" for potential metadata, "Time" and "0" for potential data table.
    raw_block_data: Dict[str, Any] = {
        'col1': ["Plate: Test Block", "Time", "00:00:00", "1.0"],
        'col2': ["Measurement: XYZ", "A1", "100", "200"]
    }
    raw_block_df: pd.DataFrame = pd.DataFrame(raw_block_data)

    result: ExpectedParseResult = parse_block(source_filename, raw_block_df)

    assert isinstance(result, tuple), "parse_block should return a tuple."
    assert len(result) == 2, "The returned tuple should have two elements."

    metadata, data_df = result

    # Check metadata type: should be a dict or None
    assert metadata is None or isinstance(metadata, dict), \
        "First element of the tuple (metadata) should be a dictionary or None."

    # Check data_df type: should be a pandas DataFrame or None
    assert data_df is None or isinstance(data_df, pd.DataFrame), \
        "Second element of the tuple (data_df) should be a pandas DataFrame or None."

    # Updated assertions for test_parse_block_returns_correct_types
    # This test now checks against the new parsing logic, not the old placeholder.
    if not raw_block_df.empty:
        assert metadata is not None
        assert data_df is not None # Expect a parsed DataFrame
        
        # Expected metadata from raw_block_data
        assert metadata.get("block_name") == "Test Block" 
        # "Measurement: XYZ" is not parsed by current logic, so "measurements" key shouldn't exist
        # or if it allows "measurement:", it should warn or parse if possible.
        # Current logic: startswith("measurements:") - this won't match "Measurement: XYZ"
        assert "measurements" not in metadata 
        # A parsing warning might occur if "Measurement: XYZ" was attempted and failed.
        # If "Measurement: XYZ" does not start with "measurements:" (plural), it's ignored.
        # Let's assume for this specific test raw_block_data, "Measurement: XYZ" is ignored for simplicity of this test case.


        # Expected data_df from raw_block_data
        # Header is ["Time", "A1"]
        # Data is ["00:00:00", "100"] and ["1.0", "200"]
        expected_data_content = {
            "Time": ["00:00:00", "1.0"],
            "A1": ["100", "200"] # Values are strings as per raw_block_data and current parsing
        }
        expected_parsed_df = pd.DataFrame(expected_data_content)
        
        assert data_df.reset_index(drop=True).equals(expected_parsed_df.reset_index(drop=True)), f"Parsed data_df does not match expected.\nExpected:\n{expected_parsed_df}\nGot:\n{data_df}" # type: ignore
    else: # This case (raw_block_df.empty) is better covered by test_parse_block_with_empty_raw_df
          # but keeping basic check here.
        assert data_df is None, "If raw_block_df is empty, cleaned_df should be None."
        # metadata is guaranteed by parse_block's behavior (see test_parse_block_with_empty_raw_df)
        # to be a dict when raw_block_df is empty.
        assert "parsing_comment" in metadata, \
            "Metadata should have a 'parsing_comment' if raw block was empty."


def test_parse_block_with_empty_raw_df() -> None:
    """
    Tests how parse_block handles an empty raw_block_df.
    """
    source_filename: str = "empty_block_source.xlsx"
    empty_raw_df: pd.DataFrame = pd.DataFrame()

    metadata, data_df = parse_block(source_filename, empty_raw_df)

    assert isinstance(metadata, dict), "Metadata should still be a dict even for empty raw df."
    assert "source_filename" in metadata, "Metadata should contain 'source_filename'."
    assert metadata["source_filename"] == source_filename
    assert "parsing_comment" in metadata, "Metadata should have a parsing comment for empty raw df."
    assert metadata["parsing_comment"] == "Raw block DataFrame was empty." # Corrected expected string
    assert data_df is None, "Cleaned DataFrame should be None for an empty raw df."


@pytest.fixture
def sample_raw_block_df_valid() -> pd.DataFrame:
    """Provides a valid sample raw_block_df for testing.
    Contains metadata and a data table.
    """
    data = {
        0: ["Plate: Activation Block", "Measurements: 120", None, "Time", "00:00:00", "00:00:15"],
        1: [None, None, None, "Temp", 37.0, 37.1],
        2: [None, None, None, "A1", 100.5, 101.0],
        3: [None, None, None, "A2", 200.1, 201.5],
        4: [None, None, None, None, None, None] # Empty column to be dropped
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_raw_block_df_no_time_header() -> pd.DataFrame:
    """Provides a raw_block_df with metadata but no 'Time' header.
    """
    data = {
        0: ["Plate: No Time Test", "Measurements: 60", None, "Header1", "Data1"],
        1: [None, None, None, "Header2", "Data2"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_raw_block_df_header_no_data() -> pd.DataFrame:
    """Provides a raw_block_df with a 'Time' header but no subsequent data rows.
    The parser should find the header, but the slice for data rows will be empty.
    """
    data = {
        0: ["Plate: HeaderOnlyTest", "Measurements: 10", "Time"],
        1: [None, None, "A1"],
        2: [None, None, "A2"]
        # No further rows, so iloc[header_idx+1:] will be empty.
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_block_df_malformed_meta() -> pd.DataFrame:
    """Provides a raw_block_df with malformed metadata.
    Specifically, a "Measurements:" line where the value is not an integer.
    """
    data = {
        0: ["Plate NoColon", "Measurements: NotANumber", "Time", "00:00:00"], # Changed to test ValueError path
        1: [None, None, "A1", 150.0]
    }
    return pd.DataFrame(data)

def test_parse_block_successful_extraction(sample_raw_block_df_valid: pd.DataFrame) -> None:
    """
    Tests successful extraction of metadata and data table from a valid block.
    Corresponds to plan.md Step 2.2 and 2.3.
    """
    source_filename: str = "valid_block.xlsx"
    metadata, data_df = parse_block(source_filename, sample_raw_block_df_valid)

    assert metadata is not None, "Metadata should be extracted."
    assert data_df is not None, "Data table should be extracted."

    # Test metadata content (Step 2.2)
    assert metadata.get("source_filename") == source_filename
    assert metadata.get("block_name") == "Activation Block"
    assert metadata.get("measurements") == 120
    # Expect a warning for the third row (index 2) where the first cell is None.
    assert "parsing_warnings" in metadata
    assert len(metadata["parsing_warnings"]) == 1
    assert "Row 2, Cell 0: Is empty, skipped for metadata." in metadata["parsing_warnings"]
    assert "parsing_comment" not in metadata # For a successful parse, no comment is expected.

    # Test data_df content (Step 2.3)
    assert not data_df.empty, "Cleaned DataFrame should not be empty."
    expected_columns = ["Time", "Temp", "A1", "A2"] # Last column is all NaN and should be dropped
    assert list(data_df.columns) == expected_columns
    assert len(data_df) == 2 # Two data rows
    assert data_df.iloc[0]["Time"] == "00:00:00"
    assert data_df.iloc[1]["A1"] == 101.0

def test_parse_block_no_time_header(sample_raw_block_df_no_time_header: pd.DataFrame) -> None:
    """
    Tests parsing when metadata is present but no 'Time' header is found.
    """
    source_filename: str = "no_time_header.xlsx"
    metadata, data_df = parse_block(source_filename, sample_raw_block_df_no_time_header)

    assert metadata is not None, "Metadata should still be extracted."
    assert data_df is None, "Data table should be None if no Time header."

    assert metadata.get("block_name") == "No Time Test"
    assert metadata.get("measurements") == 60
    assert "parsing_comment" in metadata
    assert metadata["parsing_comment"] == "Could not find data table header (e.g., 'Time' row)."


def test_parse_block_header_no_data(sample_raw_block_df_header_no_data: pd.DataFrame) -> None:
    """
    Tests parsing when a 'Time' header is found but no data rows follow.
    """
    source_filename: str = "header_no_data.xlsx"
    raw_block_df_header_only = sample_raw_block_df_header_no_data

    metadata, data_df = parse_block(source_filename, raw_block_df_header_only)

    assert metadata is not None, "Metadata should be extracted."
    assert data_df is None, "Data table should be None if header exists but no data rows lead to an empty table."
    
    assert metadata.get("block_name") == "HeaderOnlyTest"
    assert metadata.get("measurements") == 10 # As per fixture data
    assert "parsing_comment" in metadata
    # Corrected expected comment for this scenario based on parse_block logic
    assert metadata.get("parsing_comment") == "Data table start row not found or no data after header.", \
        f"Unexpected parsing comment: {metadata.get('parsing_comment')}"


def test_parse_block_malformed_metadata(sample_raw_block_df_malformed_meta: pd.DataFrame) -> None:
    """
    Tests parsing with malformed or missing standard metadata entries.
    """
    source_filename: str = "malformed_meta.xlsx"
    metadata, data_df = parse_block(source_filename, sample_raw_block_df_malformed_meta)

    assert metadata is not None
    assert data_df is not None # Data table should still parse if header is findable

    # Check what metadata was found or not found
    assert "block_name" not in metadata # "Plate NoColon" is not parsed by current logic for "plate:"
    
    # For "Measurements: NotANumber"
    assert "measurements" not in metadata # It failed to parse as int
    assert "parsing_warnings" in metadata

    # Based on current parse_block logic for sample_raw_block_df_malformed_meta:
    # raw_block_df has 2 rows. metadata search is min(5, len(df)) = 2 rows.
    # Row 0:
    #   cell 0: "Plate NoColon" -> warning: "Row 0, Cell 0: Value 'Plate NoColon' did not match prefix 'plate:' or other known metadata prefixes."
    #   (block_name_found is False)
    #   cell 0: "Plate NoColon" (re-check, not startswith "measurements:")
    # Row 1:
    #   cell 0: "Measurements: NotANumber" -> warning: "Could not parse measurements from: Measurements: NotANumber"
    #   (block_name_found is False)
    #   cell 0: "Measurements: NotANumber" (startswith "measurements:") -> try int conversion -> ValueError.
    # The original test fixture for sample_raw_block_df_malformed_meta was:
    # data = {
    #     0: ["Plate NoColon", "Measurements: NotANumber", "Time", "00:00:00"],
    #     1: [None, None, "A1", 150.0]
    # }
    # So, raw_block_df.iloc[1,0] is None.
    # Iteration i=1 (second row):
    #   pd.isna(raw_block_df.iloc[1,0]) is True.
    #   Warning: "Row 1, Cell 0: Is empty, skipped for metadata."
    #
    # So, expected warnings:
    expected_warnings = [
        "Row 0, Cell 0: Value 'Plate NoColon' did not match prefix 'plate:' or other known metadata prefixes.",
        "Could not parse measurements from: Measurements: NotANumber",
        "Row 2, Cell 0: Value 'Time' did not match prefix 'plate:' or other known metadata prefixes.",
        "Row 3, Cell 0: Value '00:00:00' did not match prefix 'plate:' or other known metadata prefixes."
    ]
    assert len(metadata["parsing_warnings"]) == len(expected_warnings)
    for warning in expected_warnings:
        assert warning in metadata["parsing_warnings"]

    # Data table should still be fine if "Time" header is present
    # The "Time" header is at raw_block_df.iloc[0,2] ("Time") in this fixture.
    # The parse_block logic searches for "time" in any cell of a row.
    # It should find "Time" at (0,2). Header_row_idx = 0. Data_table_start_row_idx = 1.
    # Cleaned_df should take columns from row 0: ["Plate NoColon", "Measurements: NotANumber", "Time", "00:00:00"]
    # Data from row 1: [None, None, "A1", 150.0]
    assert data_df is not None
    assert list(data_df.columns) == ["Time", "A1"]
    assert len(data_df) == 1
    assert data_df.iloc[0,0] == "00:00:00"
    assert data_df.iloc[0,1] == 150.0


def test_parse_block_no_metadata_but_data_table_present() -> None:
    """
    Tests a scenario where standard metadata like 'Plate:' is missing,
    but a data table with a 'Time' header is present.
    """
    source_filename: str = "data_only_block.xlsx"
    raw_data = {
        0: ["Some other info", "Another line", "Time", "00:00:10", "00:00:20"],
        1: ["Value1", "Value2", "WellA", 10, 20],
        2: ["V3", "V4", "WellB", 15, 25]
    }
    raw_block_df_data_only = pd.DataFrame(raw_data)

    metadata, data_df = parse_block(source_filename, raw_block_df_data_only)

    assert metadata is not None, "Metadata dict should always be returned."
    assert data_df is not None, "Data table should be extracted if 'Time' header is found."

    assert "block_name" not in metadata
    assert "measurements" not in metadata
    # When data table is found but block_name is not, there's no "parsing_comment" directly from the "Time" header check.
    # The "parsing_comment" for "Could not find data table header" is only set if header_row_idx remains None.
    # If header_row_idx IS found, but block_name was NOT, 'parsing_warnings' will contain messages about
    # failing to find 'plate:' prefix.
    # There should be no "parsing_comment" in this specific success case (data found, some metadata potentially missing).
    # However, parse_block adds a comment about "Potential reasons for missing block_name" if 'Time' row is not found *and* block_name is not found.
    # Let's re-evaluate the expected state of 'parsing_comment' and 'parsing_warnings'.

    # For raw_block_df_data_only:
    # metadata search (first 5 rows or len(df)):
    # Row 0: "Some other info" -> warning "...did not match prefix 'plate:'..."
    # Row 1: "Value1" -> warning "...did not match prefix 'plate:'..."
    # Row 2: "Time" -> warning "...did not match prefix 'plate:'..."
    # Row 3: "00:00:10" -> warning "...did not match prefix 'plate:'..."
    # Row 4: "00:00:20" -> warning "...did not match prefix 'plate:'..."
    # No block_name found.
    # "Time" header is found at row index 2.
    # cleaned_df is created.
    # 'parsing_warnings' should exist and contain the messages about not finding 'plate:'.
    # 'parsing_comment' should NOT exist because a data table was found.
    # The logic "if not metadata.get("parsing_warnings"):" will pop it if empty. It's not empty here.

    assert "parsing_warnings" in metadata
    assert len(metadata["parsing_warnings"]) > 0 # Expect warnings about missing plate prefix
    # Example check for one such warning
    assert "Row 0, Cell 0: Value 'Some other info' did not match prefix 'plate:' or other known metadata prefixes." in metadata["parsing_warnings"]
    assert "parsing_comment" not in metadata # data_df was successfully parsed

    assert list(data_df.columns) == ["Time", "WellA", "WellB"] # type: ignore
    assert len(data_df) == 2
    assert data_df.iloc[0]["Time"] == "00:00:10"
    assert data_df.iloc[1]["WellA"] == 20 # Assuming correct data extraction


def test_parse_block_completely_unparseable_block() -> None:
    """
    Tests a block with no recognizable metadata keywords and no 'Time' header.
    """
    source_filename: str = "unparseable_block.xlsx"
    raw_data = {
        0: ["Random Data 1", "Info X", "Value A"],
        1: ["Random Data 2", "Info Y", "Value B"],
        2: ["Random Data 3", "Info Z", "Value C"]
    }
    raw_block_df_unparseable = pd.DataFrame(raw_data)

    metadata, data_df = parse_block(source_filename, raw_block_df_unparseable)

    assert metadata is not None, "Metadata dict should still be returned."
    assert data_df is None, "Data table should be None as no 'Time' header."

    assert "block_name" not in metadata
    assert "parsing_comment" in metadata
    # Updated expected comment to align with current parsing logic
    assert metadata["parsing_comment"] == "Could not find data table header (e.g., 'Time' row). Potential reasons for missing block_name can be found in 'parsing_warnings'."
    assert "parsing_error" in metadata # Because block_name also not found
    assert metadata["parsing_error"] == "Essential metadata (block_name) and data table header not found."

def test_parse_block_custom_prefix() -> None:
    """
    Tests a scenario where standard metadata like 'Plate:' is missing,
    but a data table with a 'Time' header is present.
    """
    source_filename: str = "data_only_block.xlsx"
    raw_data = {
        0: ["CustomPrefix: Custom Prefix Block", "Another line", "Time", "00:00:10", "00:00:20"],
        1: ["Value1", "Value2", "WellA", 10, 20],
        2: ["V3", "V4", "WellB", 15, 25]
    }
    raw_block_df_data_only = pd.DataFrame(raw_data)

    metadata, data_df = parse_block(source_filename, raw_block_df_data_only, block_name_keyword_prefix="customprefix:")

    assert metadata is not None, "Metadata dict should always be returned."
    assert data_df is not None, "Data table should be extracted if 'Time' header is found."

    assert metadata.get("block_name") == "Custom Prefix Block"
    assert "parsing_warnings" not in metadata # No warnings if prefix matches and name found
    assert "parsing_comment" not in metadata

    # Test data_df content
    assert list(data_df.columns) == ["Time", "WellA", "WellB"] # type: ignore
    assert len(data_df) == 2
    assert data_df.iloc[0]["Time"] == "00:00:10"
    assert data_df.iloc[1]["WellA"] == 20 # Assuming correct data extraction

# Test to ensure 'parsing_warnings' is popped if empty after successful block_name found
def test_parsing_warnings_popped_if_empty_after_block_name_found() -> None:
    source_filename: str = "empty_block_source.xlsx"
    empty_raw_df: pd.DataFrame = pd.DataFrame()

    metadata, data_df = parse_block(source_filename, empty_raw_df)

    assert isinstance(metadata, dict), "Metadata should still be a dict even for empty raw df."
    assert "source_filename" in metadata, "Metadata should contain 'source_filename'."
    assert metadata["source_filename"] == source_filename
    assert "parsing_comment" in metadata, "Metadata should have a parsing comment for empty raw df."
    assert metadata["parsing_comment"] == "Raw block DataFrame was empty."
    assert data_df is None, "Cleaned DataFrame should be None for an empty raw df."

    # If raw_block_df is empty, parsing_warnings list is initialized but nothing is added.
    # So, it should be popped.
    assert "parsing_warnings" not in metadata, "parsing_warnings should be popped if empty."