import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit
import base64

# Define Gaussian function and Tougaard background
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))

def tougaard_background(x, a, b, c):
    return a + b * x + c * np.sqrt(x)

# Combined model: Multiple Gaussian peaks + Tougaard background
def combined_model(x, *params):
    num_peaks = (len(params) - 3) // 3  # Number of peaks
    background = tougaard_background(x, *params[-3:])  # Last 3 params are for background
    total_gaussian = np.zeros_like(x)

    for i in range(num_peaks):
        amp = params[i*3]
        cen = params[i*3+1]
        sigma = params[i*3+2]
        total_gaussian += gaussian(x, amp, cen, sigma)

    return total_gaussian + background

# Function to add header to uploaded Excel file
def add_header_to_xlsx(file_path, sheet_name):
    # Load the sheet without a header
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Check if the first row contains string values (indicating it could be a header)
    first_row = df.iloc[0]
    
    if all(isinstance(val, str) for val in first_row):  # Check if all values in the first row are strings
        # If the first row looks like a header, use the existing first row as the header
        df.columns = first_row
        df = df.drop(0)  # Drop the first row which is now the header
    else:
        # If the first row doesn't look like a header, assign custom headers
        total_columns = df.shape[1]
        # Assign 'Binding Energy' to the first column and 'Sample {i}' to the rest
        new_header = ['Binding Energy'] + [f'Sample {i}' for i in range(1, total_columns)]
        df.columns = new_header
    
    df.columns.values[0] = 'Binding Energy'
    
    # Convert all data to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df

# Function to generate download link
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main function to run the Streamlit app
def main():
    st.title("XPS Data Analysis with Multiple Gaussian Fitting")

    # Check session states for fit parameters
    if 'initial_fit_params' not in st.session_state:
        st.session_state['initial_fit_params'] = None
    if 'updated_params' not in st.session_state:
        st.session_state['updated_params'] = None

    # File upload and sheet selection
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        sheet_name = st.selectbox("Select a sheet", sheet_names)

        df = add_header_to_xlsx(uploaded_file, sheet_name)
        binding_energy = df['Binding Energy'][::-1]  # Reverse for descending order

        # Option selector for analysis type
        st.sidebar.header("XPS Analysis")
        option = st.sidebar.selectbox("Choose an analysis type", ["Individual Sample Analysis", "Overlay of All Samples"])

        if option == "Individual Sample Analysis":
            sample_columns = [col for col in df.columns if 'Sample' in col or 'Cycle' in col] # Edit here in case of different headerds
            selected_sample = st.selectbox("Select a sample column for Gaussian fitting", sample_columns)

            # Plot original data
            fig = go.FigureWidget()
            fig.add_trace(go.Scatter(x=binding_energy, y=df[selected_sample][::-1].dropna(), mode='lines', name='Original Data'))
            fig.update_layout(
                xaxis=dict(title='Binding Energy (eV)', autorange='reversed'),
                yaxis_title='Intensity (a.u.)'
            )
            st.plotly_chart(fig)

            st.header("Slice the Data")
            #energy_min = float(binding_energy.min())
            #energy_max = float(binding_energy.max())
            #selected_range = st.slider("Select range for analysis", energy_max, energy_min, (energy_max, energy_min))

            energy_min = float(binding_energy.min())
            energy_max = float(binding_energy.max())
            start_range = st.number_input("Start of range for analysis", value=energy_max, step=0.1)
            end_range = st.number_input("End of range for analysis", value=energy_min, step=0.1)

            if start_range < end_range:
                st.warning("Start of range must be greater than the end (binding energy axis is reversed).")
            else:
                mask = (binding_energy >= end_range) & (binding_energy <= start_range)
                sliced_binding_energy = binding_energy[mask].dropna()
                intensity_clean = df[selected_sample][::-1][mask].dropna()

                aligned_data = pd.DataFrame({'Binding Energy': sliced_binding_energy,
                                             selected_sample: intensity_clean}).dropna()

            #mask = (binding_energy >= selected_range[0]) & (binding_energy <= selected_range[1])
            #sliced_binding_energy = binding_energy[mask]
            #intensity_clean = df[selected_sample][::-1][mask]

            if not aligned_data.empty:
                sliced_binding_energy = aligned_data['Binding Energy']
                intensity_clean = aligned_data[selected_sample]
                # Plot the selected range
                fig = go.FigureWidget()
                fig.add_trace(go.Scatter(x=aligned_data['Binding Energy'], y=aligned_data[selected_sample], mode='lines', name='Selected Range'))
                fig.update_layout(
                    xaxis=dict(title='Binding Energy (eV)', autorange='reversed'),
                    yaxis_title='Intensity (a.u.)'
                )
                st.plotly_chart(fig)

                # Peak selection for multiple Gaussian fitting
                st.sidebar.subheader("Select Multiple Peaks for Fitting")
                num_peaks = st.sidebar.number_input("Number of peaks to fit", min_value=1, max_value=10, value=1)

                # Numerical inputs for peak parameters
                #st.sidebar.subheader("Select Multiple Peaks for Fitting")
                #num_peaks = st.sidebar.number_input("Number of peaks to fit", min_value=1, max_value=10, value=1)

                peak_parameters = []
                for i in range(num_peaks):
                    st.sidebar.write(f"Select range and center for Peak {i+1}")
                    peak_range = st.sidebar.slider(f"Select peak range {i+1}", end_range, start_range, (start_range, end_range))
                    peak_center = st.sidebar.number_input(f"Input center value for Peak {i+1}", value=float(np.mean(peak_range)))
                    peak_parameters.append((peak_range, peak_center))

                #peak_parameters = []
                #for i in range(num_peaks):
                    #st.sidebar.write(f"Parameters for Peak {i + 1}")
                    #peak_center = st.sidebar.number_input(f"Center value for Peak {i + 1}", value=float(np.mean([start_range, end_range])), step=0.1)
                    #peak_amplitude = st.sidebar.number_input(f"Amplitude for Peak {i + 1}", value=1.0, step=0.1)
                    #peak_width = st.sidebar.number_input(f"Width for Peak {i + 1}", value=1.0, step=0.1)
                    #peak_parameters.append((peak_amplitude, peak_center, peak_width))

                # Fit multiple Gaussians button
                if st.sidebar.button("Fit Multiple Gaussians"):
                    initial_guess = []
                    for i, (peak_range, peak_center) in enumerate(peak_parameters):
                        peak_mask = (sliced_binding_energy >= peak_range[0]) & (sliced_binding_energy <= peak_range[1])
                        selected_peak_intensity = intensity_clean[peak_mask]
                        peak_max = selected_peak_intensity.max() if not selected_peak_intensity.empty else 1
                        initial_guess += [peak_max, peak_center, 1.0]

                    initial_guess += [0, 1, 1]  # Background params
                    try:
                        popt, _ = curve_fit(combined_model, sliced_binding_energy, intensity_clean, p0=initial_guess, maxfev=100000)
                        st.session_state['initial_fit_params'] = popt  # Save initial fit parameters
                        st.session_state['updated_params'] = popt.copy()  # Copy for updates

                        st.header("Peak Fitting")

                        # Plot initial fit
                        fit_values = combined_model(sliced_binding_energy, *popt)
                        residuals = intensity_clean - fit_values
                        fig = go.Figure()
                                                # Combined fit
                        fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=fit_values, 
                            mode='lines', 
                            name='Combined Fit', 
                            line=dict(color='red')
                        ))

                        # Plot each Gaussian component
                        for i in range(num_peaks):
                            amp = popt[i*3]
                            cen = popt[i*3+1]
                            sigma = popt[i*3+2]
                            gaussian_values = gaussian(sliced_binding_energy, amp, cen, sigma)

                            background_values = tougaard_background(sliced_binding_energy, *popt[-3:])
                            gaussian_values_with_background = gaussian_values + background_values

                            fig.add_trace(go.Scatter(
                                x=sliced_binding_energy, 
                                y=gaussian_values_with_background, 
                                mode='lines', 
                                name=f'Gaussian {i+1}', 
                                line=dict(dash='dash')
                            ))

                        # Plot background
                        background_values = tougaard_background(sliced_binding_energy, *popt[-3:])
                        fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=background_values, 
                            mode='lines', 
                            name='Tougaard Background', 
                            line=dict(dash='dash', color='green')
                        ))

                        # Plot residuals
                        fig.add_trace(go.Scatter(
                            x=sliced_binding_energy,
                            y=residuals,
                            mode='lines',
                            name='Residuals',
                            line=dict(color='purple')
                        ))

                        # Plot original sliced data for reference
                        fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=intensity_clean, 
                            mode='lines', 
                            name='Sliced Data', 
                            line=dict(color='blue')
                        ))

                        fig.update_layout(
                            xaxis=dict(title='Binding Energy (eV)', autorange='reversed'),
                            yaxis_title='Intensity (a.u.)'
                        )

                        st.plotly_chart(fig)
                        #fig.add_trace(go.Scatter(x=sliced_binding_energy, y=fit_values, mode='lines', name='Initial Fit', line=dict(color='red')))
                        #fig.add_trace(go.Scatter(x=sliced_binding_energy, y=intensity_clean, mode='lines', name='Sliced Data', line=dict(color='blue')))
                        #st.plotly_chart(fig)
                        # Prepare data for download
                        result_df = pd.DataFrame({
                            'Binding Energy': sliced_binding_energy,
                            'Original Intensity': intensity_clean,
                            'Fitted Intensity': fit_values,
                            'Residuals': residuals
                        })
                        for i in range(num_peaks):
                            result_df[f'Gaussian {i+1}'] = gaussian(sliced_binding_energy, popt[i*3], popt[i*3+1], popt[i*3+2])
                        result_df['Background'] = background_values

                        st.markdown(download_link(result_df, 'fitted_data.csv', 'Download Fitted Data as CSV'), unsafe_allow_html=True)

                    except RuntimeError as e:
                        st.error(f"Could not fit Gaussian: {e}")

                # Adjust peaks if initial fitting is done
                #if st.session_state['initial_fit_params'] is not None:
                    #st.sidebar.subheader("Adjust Gaussian Parameters")
                    #updated_params = st.session_state['updated_params']
                    #for i in range(num_peaks):
                        #st.sidebar.write(f"Adjust Gaussian {i+1}")
                        #updated_params[i*3] = st.sidebar.slider(f"Amplitude {i+1}", 0.1, 2*updated_params[i*3], updated_params[i*3])
                        #updated_params[i*3+1] = st.sidebar.slider(f"Center {i+1}", sliced_binding_energy.min(), sliced_binding_energy.max(), updated_params[i*3+1])
                        #updated_params[i*3+2] = st.sidebar.slider(f"Width {i+1}", 0.1, 2*updated_params[i*3+2], updated_params[i*3+2])

                if st.session_state['initial_fit_params'] is not None:
                    st.sidebar.subheader("Adjust Gaussian Parameters")
                    updated_params = st.session_state['updated_params']
                    for i in range(num_peaks):
                        st.sidebar.write(f"Adjust Parameters for Gaussian {i + 1}")
                        updated_params[i * 3] = st.sidebar.number_input(f"Amplitude {i + 1}", value=updated_params[i * 3], step=0.1)
                        updated_params[i * 3 + 1] = st.sidebar.number_input(f"Center {i + 1}", value=updated_params[i * 3 + 1], step=0.1)
                        updated_params[i * 3 + 2] = st.sidebar.number_input(f"Width {i + 1}", value=updated_params[i * 3 + 2], step=0.1)

                    if st.sidebar.button("Update Fit"):
                        updated_fit_values = combined_model(sliced_binding_energy, *updated_params)
                        updated_residuals = intensity_clean - updated_fit_values
                        new_fig = go.Figure()
                        # Combined fit
                        new_fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=updated_fit_values, 
                            mode='lines', 
                            name='Combined Fit', 
                            line=dict(color='red')
                        ))

                        # Plot each Gaussian component
                        for i in range(num_peaks):
                            amp = updated_params[i*3]
                            cen = updated_params[i*3+1]
                            sigma = updated_params[i*3+2]
                            updated_gaussian_values = gaussian(sliced_binding_energy, amp, cen, sigma)

                            updated_background_values = tougaard_background(sliced_binding_energy, *updated_params[-3:])
                            updated_gaussian_values_with_background = updated_gaussian_values + updated_background_values

                            new_fig.add_trace(go.Scatter(
                                x=sliced_binding_energy, 
                                y=updated_gaussian_values_with_background, 
                                mode='lines', 
                                name=f'Gaussian {i+1}', 
                                line=dict(dash='dash')
                            ))

                        # Plot background
                        updated_background_values = tougaard_background(sliced_binding_energy, *updated_params[-3:])
                        new_fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=updated_background_values, 
                            mode='lines', 
                            name='Tougaard Background', 
                            line=dict(dash='dash', color='green')
                        ))

                        # Plot residuals
                        new_fig.add_trace(go.Scatter(
                            x=sliced_binding_energy,
                            y=updated_residuals,
                            mode='lines',
                            name='Residuals',
                            line=dict(color='purple')
                        ))

                        # Plot original sliced data for reference
                        new_fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=intensity_clean, 
                            mode='lines', 
                            name='Sliced Data', 
                            line=dict(color='blue')
                        ))

                        new_fig.update_layout(
                            xaxis=dict(title='Binding Energy (eV)', autorange='reversed'),
                            yaxis_title='Intensity (a.u.)'
                        )

                        st.plotly_chart(new_fig)

                        #new_fig.add_trace(go.Scatter(x=sliced_binding_energy, y=updated_fit_values, mode='lines', name='Updated Fit', line=dict(color='red')))
                        #new_fig.add_trace(go.Scatter(x=sliced_binding_energy, y=intensity_clean, mode='lines', name='Sliced Data', line=dict(color='blue')))
                        #st.plotly_chart(fig)
                        st.session_state['updated_params'] = updated_params  # Save updated parameters

                        updated_results_df = pd.DataFrame({
                            'Binding Energy': sliced_binding_energy,
                            'Original Intensity': intensity_clean,
                            'Fitted Intensity': updated_fit_values,
                            'Residuals': updated_residuals
                        })
                        for i in range(num_peaks):
                            updated_results_df[f'Gaussian {i+1}'] = gaussian(sliced_binding_energy, updated_params[i*3], updated_params[i*3+1], updated_params[i*3+2])
                        updated_results_df['Background'] = updated_background_values

                        st.markdown(download_link(updated_results_df, 'user_modified_fitted_data.csv', 'Download Adjusted Fitted Data as CSV'), unsafe_allow_html=True)

        elif option == "Overlay of All Samples":
            st.subheader("Plot Samples")

            # Sidebar options for plot customization
            st.sidebar.header("Plot Customization")

            # Font options
            font_family = st.sidebar.selectbox(
                "Font Family",
                ["Arial", "Courier New", "Helvetica", "Times New Roman", "Verdana"]
            )
            font_size = st.sidebar.slider("Font Size", 10, 24, 14)

            # Grid options
            show_grid = st.sidebar.checkbox("Show Grid", True)
            grid_color = st.sidebar.color_picker("Grid Color", "#e6e6e6") if show_grid else None

            # Axis titles
            xaxis_title = st.sidebar.text_input("X-axis Title", "Binding Energy (eV)")
            yaxis_title = st.sidebar.text_input("Y-axis Title", "Intensity (a.u.)")

            # Initialize figure
            fig = go.Figure()

            # Customize each line
            for i, col in enumerate(df.columns):
                if 'Sample' in col:
                    # Line customization for each sample
                    with st.sidebar.expander(f"Customize {col}"):
                        # Input to modify the sample name
                        new_sample_name = st.text_input(f"Rename {col}", value=col)
                        line_color = st.color_picker(f"{new_sample_name} Line Color", "#1f77b4")
                        line_width = st.slider(f"{new_sample_name} Line Width", 0.5, 5.0, 2.0)
                        line_dash = st.selectbox(f"{new_sample_name} Line Style", ["solid", "dash", "dot", "dashdot"], index=0)

                    # Add trace with specific customizations
                    intensity = df[col][::-1]
                    fig.add_trace(go.Scatter(
                        x=binding_energy.loc[intensity.index],
                        y=intensity,
                        mode='lines',
                        name=new_sample_name,  # Use the modified name here
                        line=dict(color=line_color, width=line_width, dash=line_dash)
                    ))

            # Update layout with global font and grid settings
            fig.update_layout(
                xaxis=dict(
                    title=xaxis_title,
                    autorange='reversed',
                    showgrid=show_grid,
                    gridcolor=grid_color
                ),
                yaxis=dict(
                    title=yaxis_title,
                    showgrid=show_grid,
                    gridcolor=grid_color
                ),
                font=dict(
                    family=font_family,
                    size=font_size
                )
            )

            # Display the plot
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()