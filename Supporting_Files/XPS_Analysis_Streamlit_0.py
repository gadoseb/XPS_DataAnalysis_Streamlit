import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import curve_fit
import base64

def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))

def tougaard_background(x, a, b, c):
    return a + b * x + c * np.sqrt(x)

def combined_model(x, *params):
    """Combined model: Multiple Gaussian peaks + Tougaard background."""
    num_peaks = (len(params) - 3) // 3  # Number of peaks
    background = tougaard_background(x, *params[-3:])  # Last 3 params are for background
    total_gaussian = np.zeros_like(x)

    for i in range(num_peaks):
        amp = params[i*3]
        cen = params[i*3+1]
        sigma = params[i*3+2]
        total_gaussian += gaussian(x, amp, cen, sigma)

    return total_gaussian + background

def add_header_to_xlsx(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    total_columns = df.shape[1]
    new_header = ['Binding Energy'] + ['Sample {}'.format(i) for i in range(total_columns - 1)]
    df.columns = new_header
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def download_link(df, filename, text):
    """Generates a link to download the data."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    st.title("XPS Data Analysis with Multiple Gaussian Fitting")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        sheet_name = st.selectbox("Select a sheet", sheet_names)

        df = add_header_to_xlsx(uploaded_file, sheet_name)
        binding_energy = df['Binding Energy']

        option = st.sidebar.selectbox("Choose an analysis type", ["Individual Sample Analysis", "Overlay of All Samples"])

        if option == "Individual Sample Analysis":
            sample_columns = [col for col in df.columns if 'Sample' in col]
            selected_sample = st.selectbox("Select a sample column for Gaussian fitting", sample_columns)

            # Plot original data
            fig = go.FigureWidget()
            fig.add_trace(go.Scatter(x=binding_energy, y=df[selected_sample], mode='lines', name='Original Data'))
            st.plotly_chart(fig)

            # Slicing data according to binding energy range
            energy_min = float(binding_energy.min())
            energy_max = float(binding_energy.max())
            selected_range = st.slider("Select range for analysis", energy_min, energy_max, (energy_min, energy_max))

            mask = (binding_energy >= selected_range[0]) & (binding_energy <= selected_range[1])
            sliced_binding_energy = binding_energy[mask]
            intensity_clean = df[selected_sample].dropna()[mask]

            if not intensity_clean.empty:
                # Plot the selected range
                fig = go.FigureWidget()
                fig.add_trace(go.Scatter(x=sliced_binding_energy, y=intensity_clean, mode='lines', name='Selected Range'))
                st.subheader("Sliced Data")
                st.plotly_chart(fig)

                # Allow the user to select both peak range and center
                st.sidebar.subheader("Select Multiple Peaks for Fitting")
                num_peaks = st.sidebar.number_input("Number of peaks to fit", min_value=1, max_value=10, value=1)

                peak_parameters = []
                for i in range(num_peaks):
                    st.sidebar.write(f"Select range and center for Peak {i+1}")
                    peak_range = st.sidebar.slider(f"Select peak range {i+1}", selected_range[0], selected_range[1], (selected_range[0], selected_range[1]))
                    peak_center = st.sidebar.number_input(f"Input center value for Peak {i+1}", value=float(np.mean(peak_range)))
                    peak_max = st.sidebar.number_input(f"Input maximum intensity value for Peak {i+1}", value=float(max(intensity_clean)))
                    peak_parameters.append((peak_range, peak_center, peak_max))

                # Button to trigger multiple Gaussian fit
                if st.sidebar.button("Fit Multiple Gaussians"):
                    initial_guess = []

                    # Prepare fitting data for all selected peaks
                    for i, (peak_range, peak_center, peak_max) in enumerate(peak_parameters):
                        peak_mask = (sliced_binding_energy >= peak_range[0]) & (sliced_binding_energy <= peak_range[1])
                        selected_peak_intensity = intensity_clean[peak_mask]

                        if not selected_peak_intensity.empty:
                            initial_guess += [
                                peak_max,  # Amplitude from user input
                                peak_center,  # Center from user input
                                1.0  # Sigma (initial guess)
                            ]

                    # Add initial guesses for background: a, b, c
                    initial_guess += [0, 1, 1]

                    # Perform the curve fitting for multiple Gaussians with background
                    try:
                        popt, _ = curve_fit(
                            combined_model, 
                            sliced_binding_energy, 
                            intensity_clean, 
                            p0=initial_guess
                        )

                        # Unpack the optimized parameters
                        fit_values = combined_model(sliced_binding_energy, *popt)

                        # Plot the fitted data (combined Gaussian) and individual Gaussian curves
                        fig = go.Figure()

                        # Plot the original sliced data for reference
                        fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=intensity_clean, 
                            mode='lines', 
                            name='Sliced Data', 
                            line=dict(color='blue')
                        ))

                        # Plot the combined Gaussian fit (total fit)
                        fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=fit_values, 
                            mode='lines', 
                            name='Combined Gaussian Fit', 
                            line=dict(color='red')
                        ))

                        # Plot each individual Gaussian component with background overlaid
                        for i in range(num_peaks):
                            amp = popt[i*3]
                            cen = popt[i*3+1]
                            sigma = popt[i*3+2]
                            gaussian_values = gaussian(sliced_binding_energy, amp, cen, sigma)

                            # Add background to each Gaussian for correct overlay
                            # gaussian_with_background = gaussian_values + tougaard_background(sliced_binding_energy, *popt[-3:])

                            fig.add_trace(go.Scatter(
                                x=sliced_binding_energy, 
                                y=gaussian_values, 
                                mode='lines', 
                                name=f'Gaussian {i+1} with Background', 
                                line=dict(dash='dash', color=f'rgba({(i+1)*50}, 100, 200, 0.8)')
                            ))

                        # Plot the background component
                        background_values = tougaard_background(sliced_binding_energy, *popt[-3:])
                        fig.add_trace(go.Scatter(
                            x=sliced_binding_energy, 
                            y=background_values, 
                            mode='lines', 
                            name='Tougaard Background', 
                            line=dict(dash='dash', color='green')
                        ))

                        st.subheader("Fitted XPS Data")
                        st.plotly_chart(fig)

                        # Prepare data for download
                        result_df = pd.DataFrame({
                            'Binding Energy': sliced_binding_energy,
                            'Original Intensity': intensity_clean,
                            'Fitted Intensity': fit_values
                        })
                        for i in range(num_peaks):
                            result_df[f'Gaussian {i+1}'] = gaussian(sliced_binding_energy, popt[i*3], popt[i*3+1], popt[i*3+2])
                        result_df['Background'] = background_values

                        st.markdown(download_link(result_df, 'fitted_data.csv', 'Download Fitted Data as CSV'), unsafe_allow_html=True)

                    except RuntimeError as e:
                        st.error(f"Could not fit Gaussian: {e}")

        elif option == "Overlay of All Samples":
            st.subheader("Plot Samples")
            fig = go.Figure()
            for col in df.columns:
                if 'Sample' in col:
                    intensity = df[col].dropna()
                    fig.add_trace(go.Scatter(
                        x=binding_energy.loc[intensity.index],
                        y=intensity,
                        mode='lines',
                        name=col
                    ))

            fig.update_layout(
                title=f"Overlay of Samples from {sheet_name}",
                xaxis_title='Binding Energy (eV)',
                yaxis_title='Intensity (a.u.)',
                legend_title="Samples"
            )

            st.plotly_chart(fig)

if __name__ == "__main__":
    main()



