import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, root_scalar
from scipy import stats
import warnings
from io import BytesIO
import json
from datetime import datetime
import base64
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="Hydration Thermodynamics Analysis",
    page_icon="🔬",
    layout="wide"
)

# Константы
R = 8.314  # J/(mol·K)

# Настройки стиля для публикаций
PUBLICATION_STYLE = {
    'font_family': 'Times New Roman',
    'font_size': 16,
    'title_font_size': 18,
    'axis_title_font_size': 16,
    'tick_font_size': 14,
    'legend_font_size': 14,
    'line_width': 2.5,
    'marker_size': 10,
    'grid_width': 0,
    'axis_line_width': 2,
    'tick_length': 6,
    'tick_width': 1.5
}

# Инициализация session state
if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []
if 'default_params' not in st.session_state:
    st.session_state.default_params = {
        'pH2O': 0.03,
        'Acc': 0.2,
        'data': """748.8659794 0.007038664
720.8247423 0.006256591
697.7319588 0.009384886
677.9381443 0.012513181
648.2474227 0.017205624
630.9278351 0.017987698
615.257732 0.025026362
597.1134021 0.028936731
581.443299 0.033629174
555.0515464 0.040667838
530.3092784 0.051616872
508.8659794 0.064912127
484.1237113 0.08211775
470.1030928 0.096195079
453.6082474 0.110272408
437.9381443 0.126695958
423.0927835 0.148594025
408.2474227 0.170492091
397.5257732 0.193172232
387.628866 0.210377856
376.9072165 0.236186292
367.8350515 0.253391916
362.0618557 0.272943761
352.9896907 0.292495606
347.2164948 0.308919156
338.1443299 0.323778559
330.7216495 0.341766257
320.8247423 0.359753954
306.8041237 0.381652021
294.4329897 0.398857645
282.8865979 0.411370826
263.9175258 0.421537786
250.7216495 0.42857645
225.1546392 0.436397188
203.7113402 0.440307557
182.2680412 0.444217926
164.9484536 0.445782074
144.3298969 0.445782074
126.185567 0.446564148
106.3917526 0.447346221
84.12371134 0.447346221"""
    }

# ============================================================================
# NUMERICAL FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def calculate_equilibrium_oh(K, Acc, pH2O):
    """Numerical solution for equilibrium [OH] concentration"""
    if K <= 0 or pH2O <= 0 or Acc <= 0:
        return np.nan
    
    def f(oh):
        return 4 * oh**2 - K * pH2O * (Acc - oh) * (6 - Acc - oh)
    
    try:
        # Try Brent's method with physical boundaries
        sol = root_scalar(
            f, 
            bracket=[1e-12, Acc - 1e-12],
            method='brentq',
            xtol=1e-14,
            rtol=1e-14
        )
        if sol.converged and 1e-12 <= sol.root <= Acc - 1e-12:
            return float(sol.root)
        else:
            return np.nan
    except (ValueError, RuntimeError):
        # Fallback: bisection method
        try:
            low, high = 1e-12, Acc - 1e-12
            f_low = f(low)
            f_high = f(high)
            
            if f_low * f_high > 0:
                return np.nan
            
            for _ in range(100):
                mid = (low + high) / 2
                f_mid = f(mid)
                
                if abs(f_mid) < 1e-12:
                    return float(mid)
                
                if f_low * f_mid < 0:
                    high = mid
                    f_high = f_mid
                else:
                    low = mid
                    f_low = f_mid
            
            return float((low + high) / 2)
        except:
            return np.nan

def analytical_OH_numerical(T_K, pH2O, Acc, dH, dS):
    """Analytical expression for [OH] with numerical solution"""
    # Calculate Kw
    Kw = np.exp(-dH/(R * T_K) + dS/R)
    K = Kw * pH2O
    
    # Scalar input
    if isinstance(T_K, (int, float)):
        return calculate_equilibrium_oh(K, Acc, pH2O)
    
    # Array input
    results = np.zeros_like(K)
    for i in range(len(K)):
        results[i] = calculate_equilibrium_oh(K[i], Acc, pH2O)
    
    return results

def calculate_Kw_with_validation(T_K, OH, pH2O, Acc):
    """Calculate Kw with data validation"""
    # Physical constraints check
    mask_valid = (
        (OH > 0) & 
        (OH < Acc) & 
        (T_K > 0) &
        (pH2O > 0) &
        (Acc > 0) & (Acc < 6)
    )
    
    if not np.any(mask_valid):
        return np.array([]), np.array([]), np.array([])
    
    T_K_valid = T_K[mask_valid]
    OH_valid = OH[mask_valid]
    
    # Calculate Kw
    numerator = 4 * OH_valid**2
    denominator = pH2O * (Acc - OH_valid) * (6 - Acc - OH_valid)
    
    # Protection from division by zero and extreme values
    mask_finite = (
        (denominator > 1e-30) & 
        (numerator > 0) &
        (denominator < 1e30)
    )
    
    if not np.any(mask_finite):
        return np.array([]), np.array([]), np.array([])
    
    T_K_final = T_K_valid[mask_finite]
    OH_final = OH_valid[mask_finite]
    Kw_final = numerator[mask_finite] / denominator[mask_finite]
    
    # Additional filtering of extreme values
    mask_reasonable = (Kw_final > 1e-30) & (Kw_final < 1e30)
    
    return (
        T_K_final[mask_reasonable], 
        OH_final[mask_reasonable], 
        Kw_final[mask_reasonable]
    )

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def parse_input_data(input_text, file_uploader=None):
    """Parse input data from text or file"""
    if file_uploader is not None:
        try:
            if file_uploader.name.endswith('.csv'):
                df = pd.read_csv(file_uploader)
            elif file_uploader.name.endswith('.txt'):
                df = pd.read_csv(file_uploader, sep=None, engine='python')
            elif file_uploader.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_uploader)
            else:
                raise ValueError("Unsupported file format")
            
            # Auto-detect columns
            temp_col = None
            oh_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(word in col_lower for word in ['temp', 't', 'temperature', '°c', 'celsius']):
                    temp_col = col
                elif any(word in col_lower for word in ['oh', 'conc', 'concentration', '[oh]']):
                    oh_col = col
            
            if temp_col is None or oh_col is None:
                # Take first two columns
                temp_col, oh_col = df.columns[:2]
            
            data = df[[temp_col, oh_col]].values
            return data, f"File loaded: {file_uploader.name}, {len(data)} points"
            
        except Exception as e:
            st.warning(f"File reading error: {e}. Using text data.")
    
    # Parse from text
    lines = input_text.strip().split('\n')
    data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Replace separators
        line = line.replace(',', '.').replace(';', ' ').replace('\t', ' ')
        
        # Remove extra spaces
        while '  ' in line:
            line = line.replace('  ', ' ')
        
        parts = line.split()
        if len(parts) >= 2:
            try:
                t = float(parts[0])
                oh = float(parts[1])
                data.append([t, oh])
            except:
                continue
    
    if not data:
        # Demo data
        data = [[20, 0.15], [100, 0.12], [200, 0.10], [300, 0.08], 
                [400, 0.06], [500, 0.04], [600, 0.02], [700, 0.01], [800, 0.005]]
        return np.array(data), "Using demo data"
    
    return np.array(data), f"Loaded {len(data)} points from text"

def validate_input_data(data_array, Acc):
    """Validate input data - REMOVED monotonicity check"""
    if data_array is None or len(data_array) == 0:
        return False, "No data for analysis"
    
    T_C = data_array[:, 0]
    OH = data_array[:, 1]
    
    issues = []
    
    # Temperature check
    if np.any(T_C < -273.15):
        issues.append("Temperatures below absolute zero found")
    if np.any(T_C > 3000):
        issues.append("Suspiciously high temperatures (>3000°C)")
    
    # Concentration check
    if np.any(OH <= 0):
        issues.append("Non-positive [OH] concentrations found")
    if np.any(OH >= Acc):
        issues.append("[OH] concentrations ≥ [Acc] found (physically impossible)")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(T_C)) or np.any(np.isnan(OH)):
        issues.append("NaN values found in data")
    if np.any(np.isinf(T_C)) or np.any(np.isinf(OH)):
        issues.append("Infinite values found in data")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "Data is valid"

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def get_table_download_link(df, filename="results.csv"):
    """Generate download link for table"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def get_json_download_link(data, filename="parameters.json"):
    """Generate download link for JSON"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">📥 Download JSON</a>'
    return href

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_plotly_figure(title, x_title, y_title, width=800, height=600):
    """Create publication-quality figure with English labels"""
    fig = go.Figure()
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['title_font_size'],
                color='black'
            ),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(
                    family=PUBLICATION_STYLE['font_family'],
                    size=PUBLICATION_STYLE['axis_title_font_size'],
                    color='black'
                )
            ),
            showline=True,
            linewidth=PUBLICATION_STYLE['axis_line_width'],
            linecolor='black',
            mirror=True,
            showgrid=False,
            zeroline=False,
            tickfont=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['tick_font_size'],
                color='black'
            ),
            ticks='outside',
            ticklen=PUBLICATION_STYLE['tick_length'],
            tickwidth=PUBLICATION_STYLE['tick_width']
        ),
        yaxis=dict(
            title=dict(
                text=y_title,
                font=dict(
                    family=PUBLICATION_STYLE['font_family'],
                    size=PUBLICATION_STYLE['axis_title_font_size'],
                    color='black'
                )
            ),
            showline=True,
            linewidth=PUBLICATION_STYLE['axis_line_width'],
            linecolor='black',
            mirror=True,
            showgrid=False,
            zeroline=False,
            tickfont=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['tick_font_size'],
                color='black'
            ),
            ticks='outside',
            ticklen=PUBLICATION_STYLE['tick_length'],
            tickwidth=PUBLICATION_STYLE['tick_width']
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        height=height,
        margin=dict(l=80, r=40, t=80, b=60),
        font=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['font_size'],
            color='black'
        ),
        legend=dict(
            font=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['legend_font_size'],
                color='black'
            ),
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.9)'
        )
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("🔬 Hydration Thermodynamics Analysis")
st.markdown("""
*Thermodynamic analysis of AB₁₋ₓAccₓO₃₋ₓ/₂ based on proton concentration temperature profile*
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Data source
    st.subheader("Data Source")
    data_source = st.radio(
        "Data source:",
        ["Text input", "Upload file"]
    )
    
    if data_source == "Upload file":
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["csv", "txt", "xlsx", "xls"],
            help="Supported: CSV, TXT, Excel. Data should contain temperature and [OH] concentration"
        )
        data_input_text = st.session_state.default_params['data']
    else:
        uploaded_file = None
        data_input_text = st.text_area(
            "Enter data (temperature °C and [OH]):",
            value=st.session_state.default_params['data'],
            height=150,
            help="Format: temperature concentration. Separator: space, tab or ;"
        )
    
    # System parameters
    st.subheader("System Parameters")
    pH2O_value = st.number_input(
        'pH₂O (atm):',
        min_value=1e-5,
        max_value=1.0,
        value=st.session_state.default_params['pH2O'],
        step=0.01,
        format="%.5f"
    )
    
    Acc_value = st.number_input(
        '[Acc] = x:',
        min_value=0.01,
        max_value=5.99,
        value=st.session_state.default_params['Acc'],
        step=0.01,
        format="%.3f",
        help="Acceptor dopant concentration (0 < x < 6)"
    )
    
    # Fitting settings
    st.subheader("Fitting Settings")
    with st.expander("Method 1: Kw Analysis", expanded=True):
        exclude_low_T_method1 = st.slider(
            'Exclude low T points:',
            min_value=0,
            max_value=10,
            value=0,
            key="m1_low"
        )
        exclude_high_T_method1 = st.slider(
            'Exclude high T points:',
            min_value=0,
            max_value=10,
            value=0,
            key="m1_high"
        )
    
    with st.expander("Method 2: Direct Fitting", expanded=True):
        exclude_low_T_method2 = st.slider(
            'Exclude low T points:',
            min_value=0,
            max_value=10,
            value=0,
            key="m2_low"
        )
        exclude_high_T_method2 = st.slider(
            'Exclude high T points:',
            min_value=0,
            max_value=10,
            value=0,
            key="m2_high"
        )
    
    # Additional options
    st.subheader("Additional Options")
    show_intermediate = st.checkbox("Show intermediate calculations", value=False)
    calculate_3d = st.checkbox("Calculate 3D surfaces", value=False)
    use_log_pH2O = st.checkbox("Logarithmic pH₂O scale in 3D", value=False)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    with col2:
        calculate_btn = st.button("🚀 Calculate", type="primary", use_container_width=True)
    
    if reset_btn:
        st.session_state.default_params = {
            'pH2O': 0.03,
            'Acc': 0.2,
            'data': """20 0.15
100 0.12
200 0.10
300 0.08
400 0.06
500 0.04
600 0.02
700 0.01
800 0.005"""
        }
        st.rerun()
    
    # Information
    st.markdown("---")
    st.markdown("**Version:** 2.0 | **Updated:** 2024")
    st.markdown("""
    **Equations:**
    - Kw = 4[OH]² / (pH₂O·([Acc]-[OH])·(6-[Acc]-[OH]))
    - ln(Kw) = -ΔH°/RT + ΔS°/R
    """)

# Main calculation section
if calculate_btn:
    try:
        with st.spinner('Processing data...'):
            # Parse and validate data
            data_array, load_message = parse_input_data(data_input_text, uploaded_file)
            is_valid, valid_message = validate_input_data(data_array, Acc_value)
            
            if not is_valid:
                st.error(f"Validation error: {valid_message}")
                st.stop()
            
            st.success(f"{load_message}. {valid_message}")
            
            # Display data
            if show_intermediate:
                with st.expander("📊 Loaded Data", expanded=True):
                    df_data = pd.DataFrame(data_array, columns=['Temperature (°C)', '[OH]'])
                    st.dataframe(df_data, use_container_width=True)
            
            # Temperature conversion
            T_C = data_array[:, 0]
            T_K = T_C + 273.15
            OH_exp = data_array[:, 1]
            
            # ====================================================================
            # METHOD 1: Kw Analysis
            # ====================================================================
            st.markdown("---")
            st.header("📈 Method 1: Equilibrium Constant Analysis")
            
            # Apply point exclusion
            n_low_m1 = exclude_low_T_method1
            n_high_m1 = exclude_high_T_method1
            
            T_K_m1 = T_K[n_low_m1:len(T_K)-n_high_m1]
            OH_exp_m1 = OH_exp[n_low_m1:len(OH_exp)-n_high_m1]
            T_C_m1 = T_C[n_low_m1:len(T_C)-n_high_m1]
            
            # Calculate Kw with validation
            T_K_valid, OH_valid, Kw_valid = calculate_Kw_with_validation(
                T_K_m1, OH_exp_m1, pH2O_value, Acc_value
            )
            
            if len(T_K_valid) < 3:
                st.error("Insufficient valid points for Kw analysis. Check data.")
                st.stop()
            
            # Linear regression
            ln_Kw = np.log(Kw_valid)
            x_m1 = 1000 / T_K_valid
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_m1, ln_Kw)
            
            # Calculate parameters with errors
            dH_method1 = -slope * R * 1000  # J/mol
            dS_method1 = intercept * R      # J/(mol·K)
            
            # Errors
            dH_err = std_err * R * 1000
            dS_err = std_err * R
            
            # 95% confidence intervals
            n = len(x_m1)
            if n > 2:
                t_val = stats.t.ppf(0.975, n-2)  # t-statistic for 95% CI
                dH_ci = t_val * dH_err
                dS_ci = t_val * dS_err
            else:
                dH_ci = 0
                dS_ci = 0
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ΔH°", f"{dH_method1/1000:.2f} ± {dH_ci/1000:.2f} kJ/mol",
                         delta=f"{dH_method1:.0f} ± {dH_ci:.0f} J/mol")
                st.metric("Points analyzed", len(T_K_valid))
            
            with col2:
                st.metric("ΔS°", f"{dS_method1:.2f} ± {dS_ci:.2f} J/(mol·K)")
                st.metric("R² coefficient", f"{r_value**2:.4f}")
            
            with col3:
                st.metric("Standard error", f"{std_err:.4f}")
                st.metric("Significance level", f"p = {p_value:.2e}")
            
            if show_intermediate:
                with st.expander("🔍 Intermediate Calculations (Method 1)"):
                    T_C_valid = T_K_valid - 273.15
                    df_kw = pd.DataFrame({
                        'T (°C)': T_C_valid,
                        'T (K)': T_K_valid,
                        '[OH]': OH_valid,
                        'Kw': Kw_valid,
                        'ln(Kw)': ln_Kw,
                        '1000/T': x_m1
                    })
                    st.dataframe(df_kw, use_container_width=True)
            
            # ====================================================================
            # METHOD 2: Direct Fitting
            # ====================================================================
            st.markdown("---")
            st.header("📊 Method 2: Direct Profile Fitting")
            
            # Apply point exclusion
            n_low_m2 = exclude_low_T_method2
            n_high_m2 = exclude_high_T_method2
            
            T_K_m2 = T_K[n_low_m2:len(T_K)-n_high_m2]
            OH_exp_m2 = OH_exp[n_low_m2:len(OH_exp)-n_high_m2]
            T_C_m2 = T_C[n_low_m2:len(T_C)-n_high_m2]
            
            # Fitting function
            def model_OH_fit(T_K_fit, dH, dS):
                return analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, dH, dS)
            
            try:
                # Nonlinear fitting
                popt, pcov = curve_fit(
                    model_OH_fit, 
                    T_K_m2, 
                    OH_exp_m2,
                    p0=[dH_method1, dS_method1],
                    bounds=([-500000, -500], [0, 500]),
                    maxfev=10000
                )
                
                dH_method2, dS_method2 = popt
                perr = np.sqrt(np.diag(pcov))
                
                # Calculate model values
                OH_model_m2 = model_OH_fit(T_K_m2, dH_method2, dS_method2)
                
                # Statistics
                residuals = OH_exp_m2 - OH_model_m2
                SSE = np.sum(residuals**2)
                SST = np.sum((OH_exp_m2 - np.mean(OH_exp_m2))**2)
                R2_method2 = 1 - (SSE/SST) if SST > 0 else 0
                RMSE = np.sqrt(SSE / len(OH_exp_m2))
                
                # 95% confidence intervals
                dH_ci_m2 = 1.96 * perr[0]
                dS_ci_m2 = 1.96 * perr[1]
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color = "green" if R2_method2 > 0.95 else "orange" if R2_method2 > 0.9 else "red"
                    st.markdown(f"<h3 style='color:{color}'>{R2_method2:.4f}</h3>", unsafe_allow_html=True)
                    st.metric("R² coefficient", f"{R2_method2:.4f}")
                    st.metric("RMSE", f"{RMSE:.6f}")
                
                with col2:
                    st.metric("ΔH°", f"{dH_method2/1000:.2f} ± {dH_ci_m2/1000:.2f} kJ/mol",
                             delta=f"{dH_method2:.0f} ± {perr[0]:.0f} J/mol")
                    st.metric("Points analyzed", len(T_K_m2))
                
                with col3:
                    st.metric("ΔS°", f"{dS_method2:.2f} ± {dS_ci_m2:.2f} J/(mol·K)",
                             delta=f"± {perr[1]:.2f}")
                    st.metric("SSE", f"{SSE:.6f}")
                
                if show_intermediate:
                    with st.expander("🔍 Intermediate Calculations (Method 2)"):
                        df_fit = pd.DataFrame({
                            'T (°C)': T_C_m2,
                            'T (K)': T_K_m2,
                            '[OH] exp': OH_exp_m2,
                            '[OH] model': OH_model_m2,
                            'Difference': residuals,
                            'Rel. error (%)': 100 * np.abs(residuals / OH_exp_m2)
                        })
                        st.dataframe(df_fit, use_container_width=True)
                
            except Exception as e:
                st.error(f"Fitting error: {e}")
                st.info("Using parameters from Method 1")
                dH_method2, dS_method2 = dH_method1, dS_method1
                R2_method2 = 0
                SSE = np.nan
                RMSE = np.nan
                perr = [0, 0]
                OH_model_m2 = analytical_OH_numerical(T_K_m2, pH2O_value, Acc_value, dH_method2, dS_method2)
                residuals = OH_exp_m2 - OH_model_m2
            
            # ====================================================================
            # SUMMARY TABLE
            # ====================================================================
            st.markdown("---")
            st.header("📋 Summary of Results")
            
            summary_data = {
                'Parameter': [
                    'ΔH° (kJ/mol)', 
                    'ΔH 95% CI (kJ/mol)',
                    'ΔS° (J/(mol·K))',
                    'ΔS 95% CI (J/(mol·K))',
                    'R²',
                    'Points analyzed',
                    'Fitting error'
                ],
                'Method 1': [
                    f"{dH_method1/1000:.2f}",
                    f"±{dH_ci/1000:.2f}",
                    f"{dS_method1:.2f}",
                    f"±{dS_ci:.2f}",
                    f"{r_value**2:.4f}",
                    f"{len(T_K_valid)}",
                    f"std_err={std_err:.4f}"
                ],
                'Method 2': [
                    f"{dH_method2/1000:.2f}",
                    f"±{dH_ci_m2/1000:.2f}",
                    f"{dS_method2:.2f}",
                    f"±{dS_ci_m2:.2f}",
                    f"{R2_method2:.4f}",
                    f"{len(T_K_m2)}",
                    f"RMSE={RMSE:.6f}" if not np.isnan(RMSE) else "N/A"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            
            # Style table
            def color_r2(val):
                if isinstance(val, str) and '=' in val:
                    num = float(val.split('=')[1])
                elif isinstance(val, str) and val.replace('.', '').isdigit():
                    num = float(val)
                else:
                    return ''
                
                if num > 0.95:
                    return 'background-color: #d4edda'
                elif num > 0.9:
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            
            st.dataframe(
                summary_df.style.applymap(color_r2, subset=['Method 1', 'Method 2']),
                use_container_width=True
            )
            
            # Export
            st.markdown("### 📤 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown(get_table_download_link(summary_df, "thermo_results.csv"), unsafe_allow_html=True)
            
            with col_exp2:
                export_data = {
                    'parameters': {
                        'pH2O': pH2O_value,
                        'Acc': Acc_value,
                        'temperature_unit': 'Celsius'
                    },
                    'method1': {
                        'dH_kJ_mol': float(dH_method1/1000),
                        'dH_CI_kJ_mol': float(dH_ci/1000),
                        'dS_J_molK': float(dS_method1),
                        'dS_CI_J_molK': float(dS_ci),
                        'R2': float(r_value**2),
                        'n_points': int(len(T_K_valid)),
                        'excluded_low': exclude_low_T_method1,
                        'excluded_high': exclude_high_T_method1
                    },
                    'method2': {
                        'dH_kJ_mol': float(dH_method2/1000),
                        'dH_CI_kJ_mol': float(dH_ci_m2/1000),
                        'dS_J_molK': float(dS_method2),
                        'dS_CI_J_molK': float(dS_ci_m2),
                        'R2': float(R2_method2),
                        'RMSE': float(RMSE) if not np.isnan(RMSE) else None,
                        'n_points': int(len(T_K_m2)),
                        'excluded_low': exclude_low_T_method2,
                        'excluded_high': exclude_high_T_method2
                    }
                }
                st.markdown(get_json_download_link(export_data, "parameters.json"), unsafe_allow_html=True)
            
            # ====================================================================
            # VISUALIZATION - ONE PLOT PER ROW
            # ====================================================================
            
            # 1. Experimental Data
            st.markdown("---")
            st.header("📊 Visualization")
            
            fig1 = create_plotly_figure(
                "Experimental Data",
                "Temperature (°C)",
                "[OH]"
            )
            
            fig1.add_trace(go.Scatter(
                x=T_C,
                y=OH_exp,
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size'],
                    color='black',
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name='Experimental data',
                showlegend=True
            ))
            
            # Add physical boundaries
            fig1.add_hline(
                y=Acc_value, 
                line=dict(color='red', width=1, dash='dash'),
                annotation_text=f'[Acc] = {Acc_value:.3f}',
                annotation_position="top right"
            )
            
            fig1.add_hline(
                y=0, 
                line=dict(color='blue', width=1, dash='dash'),
                annotation_text='[OH] = 0',
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Method 1: ln(Kw) vs 1000/T
            fig2 = create_plotly_figure(
                "Method 1: ln(Kw) vs 1000/T",
                "1000/T (K⁻¹)",
                "ln(Kw)"
            )
            
            fig2.add_trace(go.Scatter(
                x=x_m1,
                y=ln_Kw,
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size'],
                    color='blue',
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name='Data points',
                showlegend=True
            ))
            
            # Regression line
            x_fit = np.linspace(min(x_m1), max(x_m1), 100)
            y_fit = slope * x_fit + intercept
            fig2.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                line=dict(
                    color='red', 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name=f'Linear fit: R² = {r_value**2:.4f}<br>ΔH = {dH_method1/1000:.1f} kJ/mol',
                showlegend=True
            ))
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # 3. Method 2: Fitting
            fig3 = create_plotly_figure(
                "Method 2: Profile Fitting",
                "Temperature (°C)",
                "[OH]"
            )
            
            fig3.add_trace(go.Scatter(
                x=T_C_m2,
                y=OH_exp_m2,
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size'],
                    color='green',
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name='Experimental data (fitting)',
                showlegend=True
            ))
            
            # Model curve
            T_fit = np.linspace(min(T_C), max(T_C), 200)
            T_K_fit = T_fit + 273.15
            OH_fit = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, dH_method2, dS_method2)
            
            fig3.add_trace(go.Scatter(
                x=T_fit,
                y=OH_fit,
                mode='lines',
                line=dict(
                    color='orange', 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name=f'Model fit: R² = {R2_method2:.4f}<br>ΔH = {dH_method2/1000:.1f} kJ/mol',
                showlegend=True
            ))
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # 4. Residuals with proper colorbar
            if 'residuals' in locals() and len(residuals) > 0:
                fig4 = create_plotly_figure(
                    "Method 2: Residuals",
                    "Temperature (°C)",
                    "[OH]<sub>exp</sub> - [OH]<sub>model</sub>"
                )
                
                # Calculate colors based on residual magnitude
                abs_residuals = np.abs(residuals)
                colors = abs_residuals
                
                fig4.add_trace(go.Scatter(
                    x=T_C_m2,
                    y=residuals,
                    mode='markers',
                    marker=dict(
                        size=PUBLICATION_STYLE['marker_size'],
                        color=colors,
                        colorscale='RdBu',
                        colorbar=dict(
                            title=dict(
                                text="|Residual|",
                                font=dict(
                                    family=PUBLICATION_STYLE['font_family'],
                                    size=14,
                                    color='black'
                                )
                            ),
                            thickness=15,
                            len=0.5,
                            x=1.02,
                            xanchor='left',
                            y=0.5,
                            yanchor='middle'
                        ),
                        showscale=True,
                        line=dict(width=0.5, color='black')
                    ),
                    name='Residuals',
                    showlegend=False
                ))
                
                # Zero line
                fig4.add_hline(
                    y=0, 
                    line=dict(
                        color='black', 
                        width=1,
                        dash='dash'
                    )
                )
                
                # Update layout to accommodate colorbar
                fig4.update_layout(
                    margin=dict(l=80, r=100, t=80, b=60)
                )
                
                st.plotly_chart(fig4, use_container_width=True)
            
            # 5. Method Comparison
            fig5 = create_plotly_figure(
                "Comparison of Methods",
                "Temperature (°C)",
                "[OH]"
            )
            
            # Method 1 curve
            OH_fit_m1 = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, dH_method1, dS_method1)
            
            fig5.add_trace(go.Scatter(
                x=T_fit,
                y=OH_fit_m1,
                mode='lines',
                line=dict(
                    color='blue', 
                    width=PUBLICATION_STYLE['line_width'],
                    dash='dash'
                ),
                name=f'Method 1: ΔH = {dH_method1/1000:.1f} kJ/mol',
                showlegend=True
            ))
            
            # Method 2 curve
            fig5.add_trace(go.Scatter(
                x=T_fit,
                y=OH_fit,
                mode='lines',
                line=dict(
                    color='red', 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name=f'Method 2: ΔH = {dH_method2/1000:.1f} kJ/mol',
                showlegend=True
            ))
            
            # Experimental points
            fig5.add_trace(go.Scatter(
                x=T_C,
                y=OH_exp,
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size']-2,
                    color='black',
                    symbol='circle',
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                name='Experimental data',
                showlegend=True
            ))
            
            st.plotly_chart(fig5, use_container_width=True)
            
            # 6. Temperature Dependence of Kw
            fig6 = create_plotly_figure(
                "Temperature Dependence of Kw",
                "Temperature (°C)",
                "ln(Kw)"
            )
            
            # Calculate Kw for both methods
            Kw_m1 = np.exp(-dH_method1/(R * T_K_fit) + dS_method1/R)
            Kw_m2 = np.exp(-dH_method2/(R * T_K_fit) + dS_method2/R)
            
            fig6.add_trace(go.Scatter(
                x=T_fit,
                y=np.log(Kw_m1),
                mode='lines',
                line=dict(
                    color='blue', 
                    width=PUBLICATION_STYLE['line_width'],
                    dash='dash'
                ),
                name='Method 1',
                showlegend=True
            ))
            
            fig6.add_trace(go.Scatter(
                x=T_fit,
                y=np.log(Kw_m2),
                mode='lines',
                line=dict(
                    color='red', 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name='Method 2',
                showlegend=True
            ))
            
            # Experimental Kw points
            if len(T_K_valid) > 0:
                fig6.add_trace(go.Scatter(
                    x=T_K_valid - 273.15,
                    y=np.log(Kw_valid),
                    mode='markers',
                    marker=dict(
                        size=PUBLICATION_STYLE['marker_size']-2,
                        color='black',
                        symbol='circle',
                        line=dict(width=0.5, color='black')
                    ),
                    name='Experimental (Method 1)',
                    showlegend=True
                ))
            
            st.plotly_chart(fig6, use_container_width=True)
            
            # ====================================================================
            # 3D SURFACES (OPTIONAL)
            # ====================================================================
            if calculate_3d:
                st.markdown("---")
                st.header("🌐 3D Surfaces of Proton Concentration")
                
                with st.spinner('Calculating 3D surfaces...'):
                    progress_bar = st.progress(0)
                    
                    @st.cache_data(ttl=300)
                    def calculate_3d_surface_cached(method, dH, dS, Acc, pH2O_val, use_log, resolution=25):
                        """Cached function for 3D surface calculation"""
                        T_C_range = np.linspace(20, 1000, resolution)
                        pH2O_range = np.logspace(-5, 0, resolution) if use_log else np.linspace(0.00001, 1, resolution)
                        
                        T_grid, pH2O_grid = np.meshgrid(T_C_range, pH2O_range)
                        OH_grid = np.zeros_like(T_grid)
                        
                        for i in range(resolution):
                            for j in range(resolution):
                                if method == 'method1':
                                    Kw = np.exp(-dH/(R * (T_grid[i,j] + 273.15)) + dS/R)
                                    OH_grid[i,j] = calculate_equilibrium_oh(Kw, Acc, pH2O_grid[i,j])
                                else:
                                    OH_grid[i,j] = analytical_OH_numerical(
                                        T_grid[i,j] + 273.15, 
                                        pH2O_grid[i,j], 
                                        Acc, 
                                        dH, 
                                        dS
                                    )
                        
                        return T_C_range, pH2O_range, OH_grid
                    
                    # Calculate surfaces
                    progress_bar.progress(25)
                    T_range_m1, pH2O_range_m1, OH_grid_m1 = calculate_3d_surface_cached(
                        'method1', dH_method1, dS_method1, Acc_value, 
                        pH2O_value, use_log_pH2O, resolution=25
                    )
                    
                    progress_bar.progress(50)
                    T_range_m2, pH2O_range_m2, OH_grid_m2 = calculate_3d_surface_cached(
                        'method2', dH_method2, dS_method2, Acc_value,
                        pH2O_value, use_log_pH2O, resolution=25
                    )
                    
                    progress_bar.progress(75)
                    
                    # Create 3D plots
                    col_3d1, col_3d2 = st.columns(2)
                    
                    with col_3d1:
                        T_grid1, pH2O_grid1 = np.meshgrid(T_range_m1, pH2O_range_m1)
                        
                        fig_3d1 = go.Figure(data=[
                            go.Surface(
                                x=T_grid1,
                                y=np.log10(pH2O_grid1) if use_log_pH2O else pH2O_grid1,
                                z=OH_grid_m1,
                                colorscale='Viridis',
                                contours=dict(
                                    z=dict(show=True, color='black', width=1)
                                )
                            )
                        ])
                        
                        # Add experimental points
                        fig_3d1.add_trace(go.Scatter3d(
                            x=T_C,
                            y=np.log10(np.full_like(T_C, pH2O_value)) if use_log_pH2O else np.full_like(T_C, pH2O_value),
                            z=OH_exp,
                            mode='markers',
                            marker=dict(
                                size=4,
                                color='red',
                                symbol='circle'
                            ),
                            name='Experimental'
                        ))
                        
                        fig_3d1.update_layout(
                            title=dict(
                                text='Method 1',
                                font=dict(
                                    family=PUBLICATION_STYLE['font_family'],
                                    size=16,
                                    color='black'
                                )
                            ),
                            scene=dict(
                                xaxis=dict(
                                    title='Temperature (°C)',
                                    backgroundcolor='white',
                                    gridcolor='lightgray',
                                    showbackground=True,
                                    linecolor='black',
                                    linewidth=2
                                ),
                                yaxis=dict(
                                    title='log₁₀(pH₂O)' if use_log_pH2O else 'pH₂O (atm)',
                                    backgroundcolor='white',
                                    gridcolor='lightgray',
                                    showbackground=True,
                                    linecolor='black',
                                    linewidth=2
                                ),
                                zaxis=dict(
                                    title='[OH]',
                                    backgroundcolor='white',
                                    gridcolor='lightgray',
                                    showbackground=True,
                                    linecolor='black',
                                    linewidth=2
                                )
                            ),
                            height=500,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(
                                family=PUBLICATION_STYLE['font_family'],
                                size=12,
                                color='black'
                            )
                        )
                        
                        st.plotly_chart(fig_3d1, use_container_width=True)
                    
                    with col_3d2:
                        T_grid2, pH2O_grid2 = np.meshgrid(T_range_m2, pH2O_range_m2)
                        
                        fig_3d2 = go.Figure(data=[
                            go.Surface(
                                x=T_grid2,
                                y=np.log10(pH2O_grid2) if use_log_pH2O else pH2O_grid2,
                                z=OH_grid_m2,
                                colorscale='Plasma',
                                contours=dict(
                                    z=dict(show=True, color='black', width=1)
                                )
                            )
                        ])
                        
                        fig_3d2.add_trace(go.Scatter3d(
                            x=T_C,
                            y=np.log10(np.full_like(T_C, pH2O_value)) if use_log_pH2O else np.full_like(T_C, pH2O_value),
                            z=OH_exp,
                            mode='markers',
                            marker=dict(
                                size=4,
                                color='red',
                                symbol='circle'
                            ),
                            name='Experimental'
                        ))
                        
                        fig_3d2.update_layout(
                            title=dict(
                                text='Method 2',
                                font=dict(
                                    family=PUBLICATION_STYLE['font_family'],
                                    size=16,
                                    color='black'
                                )
                            ),
                            scene=dict(
                                xaxis=dict(
                                    title='Temperature (°C)',
                                    backgroundcolor='white',
                                    gridcolor='lightgray',
                                    showbackground=True,
                                    linecolor='black',
                                    linewidth=2
                                ),
                                yaxis=dict(
                                    title='log₁₀(pH₂O)' if use_log_pH2O else 'pH₂O (atm)',
                                    backgroundcolor='white',
                                    gridcolor='lightgray',
                                    showbackground=True,
                                    linecolor='black',
                                    linewidth=2
                                ),
                                zaxis=dict(
                                    title='[OH]',
                                    backgroundcolor='white',
                                    gridcolor='lightgray',
                                    showbackground=True,
                                    linecolor='black',
                                    linewidth=2
                                )
                            ),
                            height=500,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(
                                family=PUBLICATION_STYLE['font_family'],
                                size=12,
                                color='black'
                            )
                        )
                        
                        st.plotly_chart(fig_3d2, use_container_width=True)
                    
                    progress_bar.progress(100)
                    st.success("3D surfaces calculated!")
            
            # ====================================================================
            # COMMENTS AND RECOMMENDATIONS
            # ====================================================================
            st.markdown("---")
            st.header("💡 Comments and Recommendations")
            
            recommendations = []
            
            # Fitting quality
            if r_value**2 > 0.98 and R2_method2 > 0.98:
                recommendations.append("✅ Excellent agreement of both methods with data")
            elif r_value**2 > 0.95 and R2_method2 > 0.95:
                recommendations.append("✅ Good agreement of methods with data")
            elif r_value**2 < 0.9 or R2_method2 < 0.9:
                recommendations.append("⚠️ Consider excluding more points or checking data")
            
            # Parameter consistency
            if dH_method1 != 0:
                diff_percent = abs(dH_method2 - dH_method1) / abs(dH_method1) * 100
                if diff_percent > 15:
                    recommendations.append(f"⚠️ Significant ΔH° discrepancy: {diff_percent:.1f}%")
                elif diff_percent > 5:
                    recommendations.append(f"⚠️ Moderate ΔH° discrepancy: {diff_percent:.1f}%")
                else:
                    recommendations.append("✅ Good consistency in ΔH° between methods")
            
            # Display recommendations
            for rec in recommendations:
                if rec.startswith("✅"):
                    st.success(rec)
                elif rec.startswith("⚠️"):
                    st.warning(rec)
                else:
                    st.info(rec)
            
            # Final recommendations
            st.info(f"""
            **For publications:**
            - Method 1: ΔH° = {dH_method1/1000:.1f} ± {dH_ci/1000:.2f} kJ/mol
            - Method 2: ΔH° = {dH_method2/1000:.1f} ± {dH_ci_m2/1000:.2f} kJ/mol
            
            **For modeling:**
            - Recommended: Method 2 (direct fitting)
            - ΔH° = {dH_method2/1000:.1f} ± {dH_ci_m2/1000:.2f} kJ/mol
            - ΔS° = {dS_method2:.1f} ± {dS_ci_m2:.1f} J/(mol·K)
            
            **Average values:**
            - ΔH° = {(dH_method1+dH_method2)/2000:.1f} kJ/mol
            - ΔS° = {(dS_method1+dS_method2)/2:.1f} J/(mol·K)
            """)
            
            # Save to history
            calculation_entry = {
                'timestamp': datetime.now().isoformat(),
                'input_parameters': {
                    'pH2O': pH2O_value,
                    'Acc': Acc_value,
                    'data_points': len(data_array)
                },
                'results': {
                    'method1': {
                        'dH': float(dH_method1),
                        'dH_CI': float(dH_ci),
                        'dS': float(dS_method1),
                        'dS_CI': float(dS_ci),
                        'R2': float(r_value**2)
                    },
                    'method2': {
                        'dH': float(dH_method2),
                        'dH_CI': float(dH_ci_m2),
                        'dS': float(dS_method2),
                        'dS_CI': float(dS_ci_m2),
                        'R2': float(R2_method2)
                    }
                }
            }
            
            st.session_state.calculation_history.append(calculation_entry)
            
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        st.info("""
        **Possible reasons:**
        1. Incorrect data format
        2. Physically impossible parameter values
        3. Numerical convergence issues
        
        **Recommendations:**
        - Check data format
        - Ensure all [OH] values < [Acc]
        - Try excluding extreme points
        """)
        
        if show_intermediate:
            with st.expander("Technical error information"):
                import traceback
                st.code(traceback.format_exc())

# Show calculation history if exists
if len(st.session_state.calculation_history) > 0:
    with st.sidebar.expander("📜 Calculation History", expanded=False):
        for i, calc in enumerate(reversed(st.session_state.calculation_history[-5:])):
            st.markdown(f"**Calculation {i+1}**")
            st.markdown(f"Time: {calc['timestamp'][11:19]}")
            st.markdown(f"ΔH₁: {calc['results']['method1']['dH']/1000:.1f} kJ/mol")
            st.markdown(f"ΔH₂: {calc['results']['method2']['dH']/1000:.1f} kJ/mol")
            st.markdown("---")

# Initial information
if not calculate_btn:
    st.markdown("""
    ## 📖 Instructions
    
    1. **Load data** in text field or choose file (CSV, TXT, Excel)
    2. **Set system parameters**: pH₂O and acceptor concentration [Acc]
    3. **Configure fitting**: exclude extreme points if necessary
    4. **Click "Calculate"** to get thermodynamic parameters
    
    ## 🎯 Key Features
    
    ✅ **Reliable numerical solution** instead of analytical formulas  
    ✅ **Errors and confidence intervals** for all parameters  
    ✅ **File upload** in various formats  
    ✅ **Data validation** with physical correctness check  
    ✅ **Export results** to CSV, JSON  
    ✅ **Calculation caching** for fast operation  
    ✅ **3D visualization** (optional)  
    ✅ **Calculation history**  
    ✅ **Publication-quality graphs** with English labels  
    
    ## 📊 Data Format
    
    Supported formats:
    ```
    Temperature [OH]         # Separator: space
    20.5;0.15               # Separator: semicolon
    300\t0.08              # Separator: tab
    ```
    
    **Units:**
    - Temperature: °C
    - [OH] concentration: dimensionless (relative)
    - pH₂O: atmospheres (atm)
    - [Acc]: dimensionless (0 < x < 6)
    
    **Note:** Experimental data may show constant or slightly increasing [OH] with temperature within measurement error.
    """)
