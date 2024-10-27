# Streamlit App - XPS Data Analysis with Multiple Gaussian Fitting

This Streamlit application allows users to analyze X-ray photoelectron spectroscopy (XPS) data, perform multiple Gaussian fitting, and visualize the results. Users can upload an Excel file containing their XPS data, select specific samples, and fit Gaussian functions to the intensity peaks.

[Link to the App!](https://xpsdataanalysisapp-g385smv4qhspc9fk7wxjbq.streamlit.app/)

## Features

- Upload Excel files containing XPS data.
- Select the desired sheet from the uploaded file.
- Choose between individual sample analysis or overlaying all samples.
- Interactive plotting of original and sliced data.
- Select multiple peaks for Gaussian fitting.
- Adjust fitting parameters and visualize the updated fit.
- Download fitted data and adjusted parameters as CSV files.

## Simplified Sequence Diagram

[![](https://mermaid.ink/img/pako:eNqNVMtu3DAM_BVBZ-cHfAiw6DZFgAItYqSHwhfWoneJ6FWJSroI8u-lX9tN3M3GB0EQZ8gZitaz7oJBXeuMvwv6DrcEuwSu9Uq-CImpowie1X3GtD5tOCE4S7yJcR39_KdDe0MW16HvNrA9TOdD6qvr69NctbqPNoCZUqj-mOMUJJxjhVrdocDzHpGVB4d5wh8BV6sKd8gl-TXlTYlBXa22lKOFw4zOaLFjCv4dA82ImQiXtH8NR-0GGC5nbfWtN_RIpoBVDbgo_dEqJAl8e8QkSls9ZZlWsKxWDLXxYA-ZZt9nvS9ORlIl5lkl8DvZRoSHE_Z_Fd_6WBayAm9GkjLIQPZ85Wk-6nFOxBftSLSOzRlzjPXPkl8LuCFWrlimQcAXKDkT-A9W7okZzVT3iXj_1vG7szKTZZC5Utvw5MeJ_tT8mPhoM6r5tlTo1cYuF_PRtvzjgnDzBe5rcV1wv8jP8mZB3rReV9phckBGXoXnIdBq3qOT-apla7AH6eUwXS8ChcKhOfhO15wKVjqFstvrugfxVukSpXHLk7JA5Pf_GYKbQS9_AZ1QiOA?type=png)](https://mermaid.live/edit#pako:eNqNVMtu3DAM_BVBZ-cHfAiw6DZFgAItYqSHwhfWoneJ6FWJSroI8u-lX9tN3M3GB0EQZ8gZitaz7oJBXeuMvwv6DrcEuwSu9Uq-CImpowie1X3GtD5tOCE4S7yJcR39_KdDe0MW16HvNrA9TOdD6qvr69NctbqPNoCZUqj-mOMUJJxjhVrdocDzHpGVB4d5wh8BV6sKd8gl-TXlTYlBXa22lKOFw4zOaLFjCv4dA82ImQiXtH8NR-0GGC5nbfWtN_RIpoBVDbgo_dEqJAl8e8QkSls9ZZlWsKxWDLXxYA-ZZt9nvS9ORlIl5lkl8DvZRoSHE_Z_Fd_6WBayAm9GkjLIQPZ85Wk-6nFOxBftSLSOzRlzjPXPkl8LuCFWrlimQcAXKDkT-A9W7okZzVT3iXj_1vG7szKTZZC5Utvw5MeJ_tT8mPhoM6r5tlTo1cYuF_PRtvzjgnDzBe5rcV1wv8jP8mZB3rReV9phckBGXoXnIdBq3qOT-apla7AH6eUwXS8ChcKhOfhO15wKVjqFstvrugfxVukSpXHLk7JA5Pf_GYKbQS9_AZ1QiOA)

## Requirements

Ensure you have the following Python packages installed:

- `pandas`
- `streamlit`
- `plotly`
- `numpy`
- `scipy`

You can install the required packages using pip:

```bash
pip install pandas streamlit plotly numpy scipy
```

## Usage

1. **Launch the App**: Run the following command in your terminal:
    
    ```bash
    streamlit run XPS_Analysis_Streamlit.py
    ```
    
    Replace `app.py` with the name of your Python file containing the code.
    
2. **Upload Data**: Click on the "Upload an Excel file" button to select your XPS data file. The file should be in `.xlsx` format.
3. **Select a Sheet**: After uploading, choose the desired sheet from the dropdown menu.
4. **Choose Analysis Type**:
    - For individual sample analysis, select a sample column for Gaussian fitting and define the range of binding energies for analysis.
    - For overlaying all samples, select this option to visualize all sample data in a single plot.
5. **Fit Gaussian Peaks**:
    - For individual sample analysis, you can specify the number of peaks to fit, define peak ranges, and input center values.
    - Click the "Fit Multiple Gaussians" button to perform the fitting and visualize the results.
6. **Adjust Parameters**: If desired, adjust the Gaussian parameters after the initial fit and update the visualization.
7. **Download Results**: After fitting, you can download the fitted data and residuals as CSV files using the provided download links.

## Data Format

The Excel file should contain data with the following format:

- The first column must contain the binding energy values labeled as `Binding Energy`.
- Subsequent columns should be labeled as `Sample 0`, `Sample 1`, etc., representing intensity values for different samples.

## Code Structure

The main components of the application include:

- **Gaussian Fitting**: Functions to define Gaussian functions and combine them with a Tougaard background.
- **Data Handling**: Functions to read Excel files and add headers to the data.
- **Visualization**: Using Plotly to create interactive plots for data visualization.

## Functions

### `gaussian(x, amp, cen, sigma)`

Calculates the Gaussian function.

### `tougaard_background(x, a, b, c)`

Calculates the Tougaard background function.

### `combined_model(x, *params)`

Combines multiple Gaussian peaks and a Tougaard background for fitting.

### `add_header_to_xlsx(file_path, sheet_name)`

Adds headers to the specified Excel sheet if not present.

### `download_link(df, filename, text)`

Generates a link for downloading a DataFrame as a CSV file.

### `main()`

Runs the Streamlit app.

Generates a link for downloading a DataFrame as a CSV file.

### `main()`

Runs the Streamlit app.
