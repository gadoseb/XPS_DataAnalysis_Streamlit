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
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    total_columns = df.shape[1]
    new_header = ['Binding Energy'] + ['Sample {}'.format(i) for i in range(total_columns - 1)]
    df.columns = new_header
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
        option = st.sidebar.selectbox("Choose an analysis type", ["Individual Sample Analysis", "Overlay of All Samples"])

        if option == "Individual Sample Analysis":
            sample_columns = [col for col in df.columns if 'Sample' in col]
            selected_sample = st.selectbox("Select a sample column for Gaussian fitting", sample_columns)

            # Plot original data
            fig = go.FigureWidget()
            fig.add_trace(go.Scatter(x=binding_energy, y=df[selected_sample][::-1], mode='lines', name='Original Data'))
            fig.update_layout(
                xaxis=dict(title='Binding Energy (eV)', autorange='reversed'),
                yaxis_title='Intensity (a.u.)'
            )
            st.plotly_chart(fig)

            st.subheader("Slice the Data")
            energy_min = float(binding_energy.min())
            energy_max = float(binding_energy.max())
            selected_range = st.slider("Select range for analysis", energy_max, energy_min, (energy_max, energy_min))

            mask = (binding_energy >= selected_range[0]) & (binding_energy <= selected_range[1])
            sliced_binding_energy = binding_energy[mask]
            intensity_clean = df[selected_sample][::-1][mask]

            if not intensity_clean.empty:
                # Plot the selected range
                fig = go.FigureWidget()
                fig.add_trace(go.Scatter(x=sliced_binding_energy, y=intensity_clean, mode='lines', name='Selected Range'))
                fig.update_layout(
                    xaxis=dict(title='Binding Energy (eV)', autorange='reversed'),
                    yaxis_title='Intensity (a.u.)'
                )
                st.plotly_chart(fig)

                # Peak selection for multiple Gaussian fitting
                st.sidebar.subheader("Select Multiple Peaks for Fitting")
                num_peaks = st.sidebar.number_input("Number of peaks to fit", min_value=1, max_value=10, value=1)

                peak_parameters = []
                for i in range(num_peaks):
                    st.sidebar.write(f"Select range and center for Peak {i+1}")
                    peak_range = st.sidebar.slider(f"Select peak range {i+1}", selected_range[1], selected_range[0], (selected_range[0], selected_range[1]))
                    peak_center = st.sidebar.number_input(f"Input center value for Peak {i+1}", value=float(np.mean(peak_range)))
                    peak_parameters.append((peak_range, peak_center))

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
                if st.session_state['initial_fit_params'] is not None:
                    st.sidebar.subheader("Adjust Gaussian Parameters")
                    updated_params = st.session_state['updated_params']
                    for i in range(num_peaks):
                        st.sidebar.write(f"Adjust Gaussian {i+1}")
                        updated_params[i*3] = st.sidebar.slider(f"Amplitude {i+1}", 0.1, 2*updated_params[i*3], updated_params[i*3])
                        updated_params[i*3+1] = st.sidebar.slider(f"Center {i+1}", sliced_binding_energy.min(), sliced_binding_energy.max(), updated_params[i*3+1])
                        updated_params[i*3+2] = st.sidebar.slider(f"Width {i+1}", 0.1, 2*updated_params[i*3+2], updated_params[i*3+2])

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
            fig = go.Figure()
            for col in df.columns:
                if 'Sample' in col:
                    intensity = df[col].dropna()[::-1]
                    fig.add_trace(go.Scatter(x=binding_energy.loc[intensity.index], y=intensity, mode='lines', name=col))
            fig.update_layout(
                xaxis=dict(title='Binding Energy (eV)', autorange='reversed'),
                yaxis_title='Intensity (a.u.)'
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
