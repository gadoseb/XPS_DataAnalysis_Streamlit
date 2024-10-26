# Streamlit App - XPS Data Analysis with Multiple Gaussian Fitting

## Overview

This Streamlit app provides a user-friendly interface for analyzing X-ray Photoelectron Spectroscopy (XPS) data. It allows users to upload Excel files containing binding energy and intensity data, perform Gaussian fitting on selected samples, and visualize the results. Additionally, it offers functionality for background subtraction using the Tougaard method.

[Link to the App!](https://xpsdataanalysisapp-g385smv4qhspc9fk7wxjbq.streamlit.app/)

## Features

- **Upload Excel Files**: Users can upload Excel files containing XPS data.
- **Select Sheets**: Choose from multiple sheets within the uploaded Excel file.
- **Individual Sample Analysis**: Perform Gaussian fitting on a selected sample with the option to customize peak parameters.
- **Overlay of All Samples**: Visualise the intensity data of all samples on a single plot.
- **Background Subtraction**: Apply Tougaard background correction to enhance fitting accuracy.
- **Residuals Visualisation**: Visualise the residual spectrum, by calculating the difference between the original intensity and the fitted intensity.
- **Download Results**: Users can download fitted data as a CSV file for further analysis.

## Sequence Diagram

[![](https://mermaid.ink/img/pako:eNqNVMtu3DAM_BVBZ-cHfAiw6DZFgAItYqSHwhfWoneJ6FWJSroI8u-lX9tN3M3GB0EQZ8gZitaz7oJBXeuMvwv6DrcEuwSu9Uq-CImpowie1X3GtD5tOCE4S7yJcR39_KdDe0MW16HvNrA9TOdD6qvr69NctbqPNoCZUqj-mOMUJJxjhVrdocDzHpGVB4d5wh8BV6sKd8gl-TXlTYlBXa22lKOFw4zOaLFjCv4dA82ImQiXtH8NR-0GGC5nbfWtN_RIpoBVDbgo_dEqJAl8e8QkSls9ZZlWsKxWDLXxYA-ZZt9nvS9ORlIl5lkl8DvZRoSHE_Z_Fd_6WBayAm9GkjLIQPZ85Wk-6nFOxBftSLSOzRlzjPXPkl8LuCFWrlimQcAXKDkT-A9W7okZzVT3iXj_1vG7szKTZZC5Utvw5MeJ_tT8mPhoM6r5tlTo1cYuF_PRtvzjgnDzBe5rcV1wv8jP8mZB3rReV9phckBGXoXnIdBq3qOT-apla7AH6eUwXS8ChcKhOfhO15wKVjqFstvrugfxVukSpXHLk7JA5Pf_GYKbQS9_AZ1QiOA?type=png)](https://mermaid.live/edit#pako:eNqNVMtu3DAM_BVBZ-cHfAiw6DZFgAItYqSHwhfWoneJ6FWJSroI8u-lX9tN3M3GB0EQZ8gZitaz7oJBXeuMvwv6DrcEuwSu9Uq-CImpowie1X3GtD5tOCE4S7yJcR39_KdDe0MW16HvNrA9TOdD6qvr69NctbqPNoCZUqj-mOMUJJxjhVrdocDzHpGVB4d5wh8BV6sKd8gl-TXlTYlBXa22lKOFw4zOaLFjCv4dA82ImQiXtH8NR-0GGC5nbfWtN_RIpoBVDbgo_dEqJAl8e8QkSls9ZZlWsKxWDLXxYA-ZZt9nvS9ORlIl5lkl8DvZRoSHE_Z_Fd_6WBayAm9GkjLIQPZ85Wk-6nFOxBftSLSOzRlzjPXPkl8LuCFWrlimQcAXKDkT-A9W7okZzVT3iXj_1vG7szKTZZC5Utvw5MeJ_tT8mPhoM6r5tlTo1cYuF_PRtvzjgnDzBe5rcV1wv8jP8mZB3rReV9phckBGXoXnIdBq3qOT-apla7AH6eUwXS8ChcKhOfhO15wKVjqFstvrugfxVukSpXHLk7JA5Pf_GYKbQS9_AZ1QiOA)

## Installation

To run this app, you will need Python and the following libraries:

- `pandas`
- `streamlit`
- `plotly`
- `numpy`
- `scipy`

You can install the required libraries using pip:

```
pip install pandas streamlit plotly numpy scipy
```

## Usage

1. Clone or download this repository.
2. Navigate to the directory containing the script.
3. Run the Streamlit app:
    
    ```
    streamlit run XPS_Analysis_Streamlit.py
    ```
    
4. Open your web browser and go to `http://localhost:8501`.
5. Upload an Excel file and select the desired sheet.
6. Choose the type of analysis (Individual Sample Analysis or Overlay of All Samples).
7. For Individual Sample Analysis, select a sample and configure peak parameters for Gaussian fitting.
8. View the fitted results and download the data as needed.

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
