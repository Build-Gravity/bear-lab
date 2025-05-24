# CFTR Channel Activity Assay Analysis

This project provides a Python script for analyzing multi-block experimental data from CFTR channel activity assays.
The primary goal is to parse XLSX files, identify and focus on "activation" blocks, calculate ΔF/F₀ values using preceding "background" blocks, detect biphasic changes
in ΔF/F₀ (an increasing phase and a decreasing phase), calculate the slopes of these two phases for each well, and visualize the results.

## Project Structure (Conceptual)

*   `input_data/`: Directory where users place their input `.xlsx` files.
*   `output_data/`: Directory where CSV results are saved.
*   `output_plots/`: Directory where generated plots for each well are saved.
*   `src/`: Contains the source Python modules:
    *   `file_ingestion.py`: Handles finding and segmenting XLSX files into raw blocks.
    *   `block_parser.py`: Parses individual raw blocks into metadata and cleaned data tables.
    *   `processing.py`: Contains functions for processing blocks, selecting activation blocks, data preparation, kinetic calculations (`calculate_activation_kinetics`, `analyze_block_kinetics`), and saving results.
    *   `visualization.py`: Contains functions for plotting kinetic data (`plot_well_kinetics_with_fits`).
*   `tests/`: Contains unit tests for the functions in `src/`.
*   `main.py` (or a similar script, TBD): Orchestrates the overall workflow from file input to result output and plotting.
*   `requirements.txt`: Lists project dependencies.
*   `README.md`: This file.
*   `STYLE_GUIDE.md`: Outlines coding conventions (see [Style Guide and Coding Conventions](STYLE_GUIDE.md)).
*   `plan.md`: The development plan document.

## Getting Started (for Non-Technical Users)

This guide will help you run the CFTR channel activity assay analysis.

### 1. Prerequisites

*   **Python:** You need Python installed on your computer. If you don't have it, you can download it from [python.org](https://www.python.org/downloads/). Version 3.8 or newer is recommended. During installation, make sure to check the box that says "Add Python to PATH" or "Add Python.exe to PATH".

### 2. Setting Up the Project

1.  **Download the Code:**
    *   If you received the code as a ZIP file, find the file and unzip it to a folder on your computer (e.g., in your `Documents` or `Downloads` folder).
    *   If the code is hosted online (like on GitHub), you might need to use a "Download ZIP" option or use a program like GitHub Desktop to get the files.

2.  **Open the Terminal (Command Prompt):**
    *   This is a program that lets you type commands.
    *   **Windows:** Search for "Command Prompt" or "PowerShell" in the Start Menu and open it.
    *   **macOS:** Open "Finder", go to "Applications" -> "Utilities", and open "Terminal".
    *   **Linux:** You likely know how to open a terminal (e.g., Ctrl+Alt+T).

3.  **Navigate to the Project Folder:**
    *   In the terminal you just opened, you need to tell it to go into the folder where you put the project code.
    *   Use the `cd` command (which stands for "change directory").
    *   For example, if you unzipped the project into `C:\Users\YourName\Documents\bear-lab`, you would type:
        `cd C:\Users\YourName\Documents\bear-lab`
        (Replace `YourName` and the path with your actual folder location).
    *   *Tip: You can often drag the folder from your file explorer directly into the terminal window, and it will paste the path for you.*

4.  **Create a Virtual Environment (Good Practice):**
    *   This step creates an isolated space for this project's software needs, so it doesn't interfere with other Python programs on your computer.
    *   In the terminal (make sure you are in the project folder), type this command and press Enter:
        `python -m venv venv`
    *   This creates a folder named `venv` inside your project directory.

5.  **Activate the Virtual Environment:**
    *   You need to "turn on" this isolated space.
    *   **Windows (in Command Prompt):** `.\venv\Scripts\activate`
    *   **Windows (in PowerShell):** `.\venv\Scripts\Activate.ps1` (If you get an error about execution policies, you might need to run PowerShell as Administrator or run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` first, then try again).
    *   **macOS/Linux:** `source venv/bin/activate`
    *   You should see `(venv)` at the beginning of your terminal prompt, indicating it's active. If you close the terminal, you'll need to reactivate it next time.

6.  **Install Required Software Packages:**
    *   The project uses some specific Python tools. This command installs them into your virtual environment.
    *   Make sure `(venv)` is visible in your prompt, then type this and press Enter:
        `pip install -r requirements.txt`
    *   Wait for it to download and install everything. You might see a lot of text scrolling by.

### 3. Preparing Your Data Files

1.  **Locate/Create the `input_data` Folder:**
    *   Inside your main project folder (e.g., `bear-lab`), look for a folder named `input_data`.
    *   If it's not there, create it.

2.  **Add Your Excel Files:**
    *   Copy your experiment's Excel files (they must end with `.xlsx`) into this `input_data` folder.
    *   The analysis script will only look at the **first sheet** in each Excel file.

3.  **Format Your Excel Files for Analysis:**
    *   The script needs to know where one set of experimental readings (a "block") ends and another begins.
    *   In your Excel sheet, insert a new row between different experimental blocks.
    *   In the very first cell of this new row (column A), type exactly `~End` (it's case-sensitive).
    *   The script uses these `~End` markers to separate the data correctly.

### 4. Running the Analysis

1.  **Make Sure Your Virtual Environment is Active:**
    *   If you don't see `(venv)` at the start of your terminal prompt, go back to step 2.5 and activate it.
    *   Make sure you are still in the project's main folder in the terminal (use `cd` if needed).

2.  **Run the Main Script:**
    *   Type the following command and press Enter:
        `python main.py`
    *   The script will start processing your files. You'll see messages about its progress.

### 5. Getting Your Results

*   Once the script finishes (you'll see a "workflow completed successfully!" message), it will have created (or updated) two folders inside your project directory:
    *   `output_data/`: This folder will contain CSV files (spreadsheet-like files) with the calculated numerical results (like slopes). The filename will include the date and time of the analysis.
    *   `output_plots/`: This folder will contain image files (plots) for each well analyzed, showing the kinetic data and fits.

### Troubleshooting Tips

*   **"Command not found" or "'python' is not recognized..."**:
    *   Python might not be installed correctly, or not added to your system's PATH. Revisit Step 1 (Prerequisites).
    *   If you created a virtual environment (`venv`), make sure it's activated (Step 2.5).
*   **Permission Errors (especially with `Activate.ps1` on PowerShell):**
    *   You might need to run PowerShell as an administrator.
    *   Or, try the command: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then try activating again.
*   **No `input_data` folder:** Make sure you created it in the correct location (directly inside the main project folder).
*   **Script runs but no output / errors related to files:**
    *   Double-check that your `.xlsx` files are in the `input_data` folder.
    *   Ensure your Excel files have the `~End` markers correctly placed on the first sheet to separate blocks.
    *   Make sure the file extension is lowercase `.xlsx`.

## Setup (Developer)

1.  Ensure you have Python 3.8+ installed.
2.  Clone this repository (if applicable).
3.  Navigate to the project directory.
4.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
5.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ``` 

## Data Input and Raw Block Segmentation

This section describes how the script ingests data from your `.xlsx` files and performs the initial segmentation into raw experimental blocks.

### File Location and Identification

*   **Input Directory:** All input `.xlsx` files must be placed in a directory named `input_data` located at the root of the project. If this directory does not exist, you will need to create it.
*   **File Scanning:** The script scans this `input_data/` directory (non-recursively) for files with a lowercase `.xlsx` extension (e.g., `my_experiment.xlsx`). Other files (e.g., `.XLSX`, `.txt`) or `.xlsx` files located in subdirectories of `input_data/` will be ignored by the current file identification logic.

### Supported XLSX Structure for Raw Block Segmentation

The script makes the following assumptions when reading and segmenting your Excel files into raw blocks:

*   **First Sheet Only:** Data is read exclusively from the *first sheet* of each successfully identified `.xlsx` file.
*   **Block Separator:** Experimental blocks within a single sheet are expected to be separated by a specific marker:
    *   A row where the *first cell* (column A) contains the exact string `~End`. The comparison is case-sensitive, but leading/trailing whitespace around `~End` in the cell is automatically stripped before comparison.
    *   These `~End` separator rows are used to identify the end of a block and are *not* included in the data content of any block.
*   **Data Integrity and Error Handling:** 
    *   If an `.xlsx` file cannot be read by the underlying library (e.g., it's corrupted, password-protected, or not a valid Excel Open XML format), it will typically be skipped. The script aims to continue processing other valid files.
    *   Sheets that are empty or cannot be segmented according to the `~End` logic might result in no blocks being extracted from that particular file.

### Output of this Stage

After this initial ingestion and segmentation phase, the script will have a collection of "raw blocks." Each raw block is represented as a Pandas DataFrame containing all rows and columns belonging to that block as they appeared in the original Excel sheet (between `~End` markers, or from the start of the sheet to the first `~End`, or from the last `~End` to the end of the sheet). Each raw block is also associated with its original source filename (e.g., `my_experiment.xlsx`). These raw blocks are then passed to the next stage for more detailed parsing.

*(Note: The detailed parsing of metadata lines (like "Plate:", "Measurements:") and the structure of the data table within each raw block is handled in a subsequent step, described under "Part 2: Individual Block Parsing".)* 

## Individual Block Parsing (`parse_block`)

Once raw blocks are segmented from the XLSX files (as described in "Data Input and Raw Block Segmentation"), each individual `raw_block_dataframe` (along with its `source_filename`) is processed by the `parse_block` function. This function is typically located in `src/block_parser.py`.

**Objective:** To transform each raw, unstructured block DataFrame into more organized information: a dictionary of metadata and a cleaned, tabular DataFrame representing the main experimental data.

### Input to `parse_block`

*   `source_filename (str)`: The name of the original XLSX file from which the block originated.
*   `raw_block_df (pd.DataFrame)`: The Pandas DataFrame representing the single raw block, which may contain a mix of metadata lines and tabular data.
*   `block_name_keyword_prefix (str, optional)`: A string that indicates the start of a line containing the block's name. Defaults to "plate:" (case-insensitive). For example, if a line starts with "Plate: Activation Assay", "Activation Assay" would be extracted as the block name.

### Output of `parse_block`

The function returns a tuple: `(metadata_dict, cleaned_tabular_df)`

1.  **`metadata_dict (dict)`:**
    *   A Python dictionary containing extracted metadata. Key entries typically include:
        *   `"source_filename"`: The original filename.
        *   `"block_name"`: The name of the block (e.g., "Background", "Activation"), extracted using the `block_name_keyword_prefix`.
        *   `"measurements"`: The number of measurements, if a line like "Measurements: 120" is found.
        *   `"parsing_warnings" (list, optional)`: A list of strings detailing any non-critical issues encountered during parsing of this block (e.g., a metadata line was found but its value couldn't be interpreted, or an expected metadata line was missing).
        *   `"parsing_comment" (str, optional)`: A general comment about the parsing status, especially if the data table could not be extracted (e.g., "Could not find data table header (e.g., 'Time' row).").
        *   `"parsing_error" (str, optional)`: A comment indicating more critical parsing failures, for example, if neither essential metadata (like `block_name`) nor the data table header could be found.
    *   If metadata like `block_name` or `measurements` cannot be found or parsed, the corresponding keys might be absent from the dictionary, or their absence will be noted in `parsing_warnings` or `parsing_comment`.

2.  **`cleaned_tabular_df (Optional[pd.DataFrame])`:**
    *   A Pandas DataFrame representing the main kinetic data table from the block. This DataFrame is cleaned by:
        *   Identifying a header row (typically by looking for a cell containing "Time", case-insensitive).
        *   Using this row to define the column names.
        *   Extracting all subsequent rows as data.
        *   Dropping any columns or rows that are entirely empty (all NaN values).
    *   If a data table header cannot be reliably identified, or if no data rows exist after the header, or if the raw block itself was empty, this part of the output will be `None`. The `metadata_dict` will usually contain a `parsing_comment` explaining why.

This `parse_block` function is crucial for structuring the varied content of each experimental block into a consistent format that can be used for subsequent data preparation and analysis steps. 

## Data Preparation for Kinetic Analysis

After individual raw blocks have been parsed into metadata and cleaned DataFrames (see "Individual Block Parsing (`parse_block`)"), the script performs several preparation steps, particularly focusing on blocks identified as "activation" blocks, to get them ready for kinetic analysis.

### 1. Processing All Parsed Blocks

*   The `process_all_blocks` function (typically in `src/processing.py`) iterates through all the `(source_filename, raw_block_dataframe)` tuples obtained from the initial file ingestion (Part 1).
*   It applies the `parse_block` function to each raw block.
*   **Output:** This step produces a list of `(metadata_dict, cleaned_dataframe)` tuples. Importantly, only blocks for which `parse_block` successfully returned a `cleaned_dataframe` (i.e., the DataFrame is not `None` and not empty) are kept for further processing. Blocks that could not be meaningfully parsed into a data table are filtered out at this stage.

### 2. Identifying and Pairing "Background" and "Activation" Blocks

*   From the list of successfully parsed blocks, the script identifies and pairs "background" and "activation" blocks. This pairing is crucial for calculating F₀ values which are then used for the ΔF/F₀ calculation in the activation block.
*   **Selection and Pairing Criteria:**
    *   It checks the `"block_name"` entry in each block's `metadata_dict`.
    *   It looks for specific keywords (case-insensitive): e.g., "background" for background blocks and "activation" for activation blocks. These keywords can be configurable.
    *   **Pairing Logic:** A common assumption is that a background block immediately precedes its corresponding activation block within the same source XLSX file. The script will attempt to pair them based on this sequential order and matching `source_filename`.
    *   Only pairs where both the background and activation blocks have valid, non-empty `cleaned_dataframe` are typically considered for further analysis.
*   **Output:** This results in a list of paired blocks, e.g., `paired_blocks_for_analysis`, where each item might be a tuple like `((background_metadata, background_df), (activation_metadata, activation_df))`. Unpaired blocks or pairs with missing data are typically logged and excluded from ΔF/F₀ analysis.

### 3. Calculating F₀ from "Background" Blocks

*   For each successfully paired `((background_metadata, background_df), (activation_metadata, activation_df))`, the F₀ (baseline fluorescence) is calculated for each well using the data from the `background_df`.
*   **Method for F₀ Calculation (per well):**
    1.  The fluorescence data for the specific well in `background_df` must be numeric (this is typically ensured by `ensure_numeric_well_data` applied to all parsed blocks).
    2.  F₀ is calculated by averaging the fluorescence readings for that well. A common and often preferred method is to average the *last N* (e.g., N=3, this value should be configurable) valid numeric data points for that well in the `background_df`. This uses the most immediate pre-stimulus baseline.
    3.  If a well has fewer than N valid points, all available valid points are averaged.
    4.  If a well has no valid numeric fluorescence data in the background block, its F₀ will be `NaN` (Not a Number). This well cannot be used for subsequent ΔF/F₀ calculations in the corresponding activation block.
*   **Storage:** The calculated F₀ values (one for each well in the background block) are stored, often in a way that they can be easily retrieved when processing the paired activation block (e.g., a dictionary mapping `(source_filename, well_id)` to `f0_value`, or by adding it to the `activation_metadata`).

### 4. Preparing Data within Each Paired "Activation" Block

For each `activation_df` in the `paired_blocks_for_analysis` list, the following data transformations are applied:

*   **Numeric Conversion of Fluorescence Data (`ensure_numeric_well_data`):
    *   **Objective:** Ensure that all columns representing fluorescence readings from wells in the `activation_df` (e.g., "A1", "A7") are of a numeric data type (typically float). This step is critical *before* ΔF/F₀ calculation.
    *   **Method:** (As described previously, using `pd.to_numeric(errors='coerce')`).
    *   **Output:** Well data columns in `activation_df` are numeric.

*   **Time Column Conversion to Seconds (`prepare_time_column`):
    *   **Objective:** Convert the time data in `activation_df` (usually a column named "Time") into total seconds.
    *   **Method:** (As described previously, using `pandas.to_timedelta()` and `.dt.total_seconds()`).
    *   **Output:** A new column, `"Time_sec"` (float type), is added to `activation_df`.

*   **ΔF/F₀ Calculation for "Activation" Blocks:**
    *   **Objective:** For each well in the `activation_df`, calculate the change in fluorescence relative to its baseline (ΔF/F₀).
    *   **Method:**
        1.  For a given well in the `activation_df`, retrieve its corresponding F₀ value calculated in Step 3.
        2.  If F₀ is valid (not `NaN` and not zero to prevent division by zero errors):
            For each time point `t` in the `activation_df` for that well:
            `ΔF/F₀ (well, t) = (Fluorescence_activation(well, t) - F₀_well) / F₀_well`
        3.  If F₀ for a well is invalid (`NaN` or zero), all ΔF/F₀ values for that well across all time points in the activation block will be `NaN`. Such wells are typically excluded from the kinetic slope analysis, and a comment should be logged.
    *   **Storage:** The original fluorescence values in the well columns of `activation_df` are **replaced** by their corresponding ΔF/F₀ values. The DataFrame now contains ΔF/F₀ time courses for each well, ready for biphasic analysis.

After these preparation steps, each `activation_df` in `paired_blocks_for_analysis` will have a `"Time_sec"` column and its well columns will contain ΔF/F₀ values.

## Biphasic Pattern Identification in Activation Blocks

For blocks identified as "activation" blocks, the script now analyzes the **ΔF/F₀ time course** to identify two distinct kinetic phases – an "increasing phase" followed by a "decreasing phase" in ΔF/F₀ – characteristic of an inverted U-shape response.

This strategy is crucial for the subsequent calculation of separate slopes for these two phases for each well within an activation block using its ΔF/F₀ data.

### Visual Confirmation (Developer/User Step)

Before relying solely on automated analysis, it is often beneficial to visually inspect plots of **ΔF/F₀** vs. `Time_sec` for some representative wells from your prepared "activation" blocks. This helps confirm if the expected inverted U-shape pattern is indeed present in the normalized data. (The script will later include features to generate these plots automatically, as described in Part 7).

### Programmatic Turning Point Identification for Activation Blocks (using ΔF/F₀)

The core logic for separating the two phases in an activation block (using its ΔF/F₀ data) relies on identifying a single "turning point" within each well's time-series data. For "activation" blocks, this is defined as follows:

1.  **Data Requirement:** A well must have a minimum number of valid (numeric time and **ΔF/F₀**) data points to be considered for this biphasic analysis (e.g., at least 4 points, this is configurable).

2.  **Identifying the Turning Point (Peak ΔF/F₀):**
    *   For each well's cleaned **ΔF/F₀ data** (after `NaN` removal for that well), the script identifies the index of the **absolute maximum ΔF/F₀ value**.
    *   This index is considered the `turning_point_index` (or `peak_delta_f_f0_index`). This data point (the peak of the ΔF/F₀ curve) is considered part of both the increasing and decreasing phases.

3.  **Defining the Phases based on the Turning Point (using ΔF/F₀):**
    *   **Increasing Phase:** Consists of all **ΔF/F₀** data points from the beginning of the well's data series up to and *including* the `turning_point_index`.
    *   **Decreasing Phase:** Consists of all **ΔF/F₀** data points from the `turning_point_index` up to and *including* the end of the well's data series.

4.  **Handling Monotonic Data or Edge Cases (using ΔF/F₀):**
    *   If the `turning_point_index` (maximum **ΔF/F₀**) is the *first* data point of the series, the "increasing phase" will effectively consist of only this single point. The "decreasing phase" will then comprise the entire **ΔF/F₀** data series starting from this point.
    *   If the `turning_point_index` is the *last* data point of the series, the "increasing phase" will comprise the entire **ΔF/F₀** data series. The "decreasing phase" will then effectively consist of only this single point.
    *   The subsequent slope calculation step (Part 5) will handle these scenarios. For example, a phase consisting of a single data point cannot have a slope calculated and will typically result in `None` for its slope and R-squared value, accompanied by an explanatory comment.

This strategy, focusing on the global maximum of the ΔF/F₀ curve, is specifically tailored for the expected "increase then decrease" pattern in activation blocks. The actual implementation of slope calculations based on these identified phases is detailed in Part 5.

## Activation Block Slope Calculation (`calculate_activation_kinetics`)

The core kinetic analysis for each well within a prepared "activation" block (now using ΔF/F₀ data) is performed by the `calculate_activation_kinetics` function. This function is specifically designed to analyze data exhibiting an "increase then decrease" pattern in ΔF/F₀, where the peak of the ΔF/F₀ curve is the key turning point.

### Inputs to `calculate_activation_kinetics`

*   `time_sec (pd.Series[float])`: A pandas Series containing the numeric time data (in seconds) for the single well being analyzed.
*   `delta_f_over_f0 (pd.Series[float])`: A pandas Series containing the numeric **ΔF/F₀** data for the well.
*   `min_points_for_analysis (int, optional)`: The minimum number of valid (non-NaN, aligned time and **ΔF/F₀**) data points required for the entire well to be considered for any biphasic analysis. Defaults to a pre-set value (e.g., 4).
*   `min_points_per_phase (int, optional)`: The minimum number of data points required *within each* identified phase (increasing or decreasing, based on **ΔF/F₀**) to calculate its slope and R-squared value. Defaults to a pre-set value (e.g., 2).

### Processing Steps within `calculate_activation_kinetics`

1.  **Initial Data Cleaning & Validation:**
    *   Combines the input `time_sec` and `delta_f_over_f0` series and removes any pairs where either value is `NaN`.
    *   Checks if the number of remaining valid data points meets `min_points_for_analysis`. If not, the analysis for the well is aborted, and the function returns `None` for all kinetic parameters with an appropriate comment.

2.  **Peak ΔF/F₀ Identification (Turning Point):**
    *   Identifies the `peak_delta_f_f0_index` by finding the index of the **maximum `delta_f_over_f0` value** in the cleaned data.

3.  **Phase Definition (using ΔF/F₀):**
    *   **Increasing Phase:** Data points from the start of the series up to and *including* the `peak_delta_f_f0_index`.
    *   **Decreasing Phase:** Data points from the `peak_delta_f_f0_index` up to and *including* the end of the series.

4.  **Slope and R-squared Calculation for Each Phase (using ΔF/F₀):**
    *   For both the increasing and decreasing phases independently:
        *   If the number of data points in the phase is less than `min_points_per_phase`, the slope and R-squared for that phase are reported as `None`, and a comment is added.
        *   If all time points within the phase are identical, the slope is typically reported as `0.0` (or `None`) and R-squared as `0.0`.
        *   Otherwise, linear regression (`scipy.stats.linregress`) is performed on the `time_sec` and `delta_f_over_f0` data of that phase to calculate its slope and R-squared value.

5.  **Comment Generation:**
    *   A `comment` string is generated to summarize the analysis outcome.

### Output (`ActivationKineticResults` Dictionary)

The function returns a dictionary with the following structure and keys (names may be adjusted to reflect ΔF/F₀):

*   `'increasing_slope_dFF0': Optional[float]` - Slope of the identified increasing phase of ΔF/F₀.
*   `'increasing_r_squared_dFF0': Optional[float]` - R-squared value for the increasing phase regression (ΔF/F₀).
*   `'decreasing_slope_dFF0': Optional[float]` - Slope of the identified decreasing phase of ΔF/F₀.
*   `'decreasing_r_squared_dFF0': Optional[float]` - R-squared value for the decreasing phase regression (ΔF/F₀).
*   `'peak_delta_f_f0_index': Optional[int]` - Index of the peak ΔF/F₀ value in the cleaned data series.
*   `'comment': Optional[str]` - A descriptive comment.

This function provides a detailed breakdown of the activation kinetic behavior (using ΔF/F₀) for a single well.

## Compiling Activation Block Results

Once the `calculate_activation_kinetics` function has determined the kinetic parameters (based on ΔF/F₀) for individual wells, the script systematically applies this analysis to all relevant wells within all identified and paired "activation" blocks and compiles these results.

### Processing Each Paired Activation Block (`analyze_block_kinetics`)

*   For each `((background_metadata, background_df), (activation_metadata, activation_df_with_dFF0))` tuple from the `paired_blocks_for_analysis` list, a function like `analyze_block_kinetics` orchestrates the analysis.
*   **Workflow within `analyze_block_kinetics`:**
    1.  Identifies well columns in `activation_df_with_dFF0`.
    2.  For each well:
        *   Extracts `Time_sec` and the **ΔF/F₀ data**.
        *   Retrieves the F₀ value for this well (calculated earlier from the corresponding `background_df`) for logging/output purposes.
        *   Calls `calculate_activation_kinetics` with the well's `Time_sec` and **ΔF/F₀ data**.
        *   Augments the results dictionary with `WellID`, `Source_File`, `Block_Name_Background`, `Block_Name_Activation`, and the `F0_Value`.
    3.  Collects all augmented result dictionaries for the block into a list.
    4.  Converts this list into a Pandas DataFrame for the current block.

### Overall Compilation of Results (Conceptual Main Script Logic)

1.  Ingest and parse XLSX files to get raw blocks.
2.  Process raw blocks to get `(metadata_dict, cleaned_dataframe)`.
3.  Filter and pair to get `paired_blocks_for_analysis`, calculating F₀ for each background block's wells and then ΔF/F₀ for each corresponding activation block's wells.
4.  Initialize an empty list for results DataFrames.
5.  **Iterate:** For each paired activation block (now with ΔF/F₀ data):
    *   Call `analyze_block_kinetics` to get a DataFrame of results for that block.
    *   Append this DataFrame to the list.
6.  **Concatenate:** Combine all results DataFrames into a single, final Pandas DataFrame.

### Structure of the Final Results DataFrame

The final compiled Pandas DataFrame will have a row for each analyzed well and include columns such as:

*   `Source_File (str)`: Original `.xlsx` filename.
*   `Block_Name_Background (str)`: Name of the background block used for F₀.
*   `Block_Name_Activation (str)`: Name of the activation block.
*   `WellID (str)`: Well identifier (e.g., "A7").
*   `F0_Value (float | None)`: The calculated F₀ (baseline fluorescence) for the well from the background block.
*   `Increasing_Slope_dFF0 (float | None)`: Slope for the increasing phase of **ΔF/F₀**.
*   `Increasing_R_Squared_dFF0 (float | None)`: R-squared for the increasing phase of **ΔF/F₀**.
*   `Decreasing_Slope_dFF0 (float | None)`: Slope for the decreasing phase of **ΔF/F₀**.
*   `Decreasing_R_Squared_dFF0 (float | None)`: R-squared for the decreasing phase of **ΔF/F₀**.
*   `Peak_DeltaF_F0_Index (int | None)`: Index of the peak **ΔF/F₀** value.
*   `Comment (str | None)`: Analysis comment for the well.

This structured DataFrame, based on ΔF/F₀ analysis, is then ready for output.

## Output Description

The script generates two main types of output:

1.  **CSV Results File:**
    *   A single CSV file (e.g., `cftr_activation_dFF0_slopes_YYYYMMDD_HHMMSS.csv`) is saved to `output_data/`.
    *   This file contains the compiled kinetic analysis results (based on ΔF/F₀) for all analyzable wells from processed "activation" blocks.
    *   The columns are as described above, reflecting the ΔF/F₀ analysis (e.g., `F0_Value`, `Increasing_Slope_dFF0`).

2.  **Well-Specific Plots (PNG Images):**
    *   For each well in an "activation" block successfully analyzed using ΔF/F₀, a PNG plot is generated.
    *   Plots are saved in `output_plots/`, possibly in subdirectories named after the activation `block_name`.
    *   Filenames might follow `source_filename_prefix_block_name_Well_well_id_dFF0.png`.
    *   These plots visually represent:
        *   The **ΔF/F₀ data points** over time for the well (Y-axis will be labeled "ΔF/F₀").
        *   The identified peak **ΔF/F₀** point.
        *   Fitted linear regression lines for the increasing and decreasing phases of **ΔF/F₀**.
        *   Plot titles include relevant information.

## Usage (Conceptual)

While a final `main.py` script will orchestrate the entire process, the core functionality can be understood as a sequence of operations:

1.  **Place Data:** Put your `.xlsx` files into the `input_data/` directory.
2.  **Run Script:** Execute the main Python script (e.g., `python main.py`).
3.  **Retrieve Outputs:** 
    *   Find the summary CSV file in `output_data/`.
    *   Browse well-specific plots in `output_plots/`.

The script will internally perform file finding, block segmentation, parsing, data preparation, kinetic analysis, result compilation, CSV saving, and plot generation.

(Detailed command-line arguments and options will be specified here if a `main.py` with such features is implemented.)

## Assumptions

This section outlines the key assumptions made by the script during its data processing and analysis workflow.

*   **Input File Format and Location:**
    *   The script expects input `.xlsx` files to be located in an `input_data/` directory at the project root.
    *   It processes only the first sheet of each Excel file.
    *   Raw experimental blocks within a sheet are assumed to be separated by a row where the first cell contains `~End`.
    *   Refer to the "Data Input and Raw Block Segmentation" and "Individual Block Parsing (`parse_block`)" sections for more details on how metadata (like block names from "Plate:" lines) and data tables (header row typically containing "Time") are expected to be structured within each raw block.

*   **"Background" and "Activation" Block Identification and Pairing:**
    *   The script identifies "background" blocks by looking for a keyword (e.g., "background", case-insensitive) and "activation" blocks by a keyword (e.g., "activation", case-insensitive) in their respective `block_name` metadata.
    *   It assumes that a "background" block immediately precedes its corresponding "activation" block within the same input XLSX file for correct pairing.

*   **F₀ Calculation:**
    *   F₀ (baseline fluorescence) for each well is calculated from its corresponding "background" block.
    *   The default method is to average the *last N* (e.g., N=3, configurable) valid numerical fluorescence readings for that well. If fewer than N points are available, all valid points are averaged.
    *   If a well has no valid numeric data in the background block, or if the calculated F₀ is zero, it may result in `NaN` for ΔF/F₀, and the well will likely be excluded from kinetic analysis with a comment.

*   **Nature of Biphasic Response in Activation Blocks (using ΔF/F₀):**
    *   For blocks identified as "activation" blocks, the script analyzes the **ΔF/F₀ time course**.
    *   It assumes an initial "increasing phase" followed by a "decreasing phase" in **ΔF/F₀**.
    *   The turning point between these two phases is determined by finding the point of **maximum ΔF/F₀ value** within that well's data for the block.

*   **Slope Calculation (`calculate_activation_kinetics` using ΔF/F₀):**
    *   Slopes for the increasing and decreasing phases are calculated using linear regression on the **ΔF/F₀ values** against `Time_sec`.
    *   Minimum data point requirements (`min_points_for_analysis`, `min_points_per_phase`) apply to the valid **ΔF/F₀** data.

*   **Replicates:**
    *   (As before: independent well processing, user performs replicate averaging post-analysis).

*   **Data Integrity and Type Conversion:**
    *   (As before: Time to seconds, fluorescence to numeric). 
    *   Critically, if F₀ is zero or `NaN`, ΔF/F₀ cannot be meaningfully calculated and will result in `NaN`s, excluding the well from slope analysis.

*   **Visualization (`plot_well_kinetics_with_fits`):**
    *   Visualizations for "activation" blocks will display **ΔF/F₀** on the Y-axis.
    *   Plots will show the raw **ΔF/F₀** data, the identified peak **ΔF/F₀**, and the fitted linear slopes based on the **ΔF/F₀** data.

## Troubleshooting / Known Issues / Limitations

*   **XLSX Format Strictness:** The script is sensitive to the `.xlsx` file structure, particularly the `~End` marker for block separation and the keywords for metadata (e.g., "Plate:", "Time"). Deviations from the documented supported formats (see "Data Input and Raw Block Segmentation" and "Individual Block Parsing") may lead to parsing errors or incorrect data extraction.
*   **Performance:** For a very large number of files or extremely large Excel files, processing time might be considerable. 
*   **Error Reporting:** While the script attempts to provide comments and skip problematic data, error reporting for complex file issues might require inspecting console output or logs (if implemented).
*   **Complex Kinetic Patterns:** The current biphasic analysis for "activation" blocks (using ΔF/F₀) is specifically designed for an "increase then decrease" (inverted U-shape) pattern in the ΔF/F₀ curve by finding the global maximum of ΔF/F₀. It may not be suitable for other complex kinetic profiles without modification.

---
*This README is actively being updated as the project develops.* 