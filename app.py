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
import plotly.express as px
warnings.filterwarnings('ignore')

# Настройки страницы
st.set_page_config(
    page_title="Hydration Thermodynamics Analysis",
    page_icon="🔬",
    layout="wide"
)

# Константы
R = 8.314  # J/(mol·K)

# Настройки стиля для научных публикаций
PUBLICATION_STYLE = {
    'font_family': 'Times New Roman, serif',
    'font_size': 16,  # Увеличен базовый размер шрифта
    'title_font_size': 18,
    'axis_title_font_size': 16,  # Подписи осей 16 ppt
    'tick_font_size': 12,  # Значения на осях 12 ppt
    'legend_font_size': 12,
    'line_width': 2.0,
    'marker_size': 8,
    'grid_width': 0,
    'axis_line_width': 2.0,
    'tick_length': 8,
    'tick_width': 1.5,
    'plot_width': 533,
    'plot_height': 400,
    'plot_ratio': 3/4
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
# PLOTTING FUNCTIONS - UPDATED FOR SCIENTIFIC PUBLICATIONS
# ============================================================================

def create_publication_figure(title, x_title, y_title, width=None, height=None):
    """Create publication-quality figure with correct aspect ratio 3:4"""
    if width is None:
        width = PUBLICATION_STYLE['plot_width']
    if height is None:
        height = PUBLICATION_STYLE['plot_height']
    
    fig = go.Figure()
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['title_font_size'],
                color='black',
                weight='bold'
            ),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(
                    family=PUBLICATION_STYLE['font_family'],
                    size=PUBLICATION_STYLE['axis_title_font_size'],
                    color='black',
                    weight='bold'
                )
            ),
            showline=True,
            linewidth=PUBLICATION_STYLE['axis_line_width'],
            linecolor='black',
            mirror=False,  # Только нижняя линия
            showgrid=False,
            zeroline=False,
            tickfont=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['tick_font_size'],
                color='black'
            ),
            ticks='outside',
            ticklen=PUBLICATION_STYLE['tick_length'],
            tickwidth=PUBLICATION_STYLE['tick_width'],
            tickcolor='black',
            gridcolor='rgba(0,0,0,0)'
        ),
        yaxis=dict(
            title=dict(
                text=y_title,
                font=dict(
                    family=PUBLICATION_STYLE['font_family'],
                    size=PUBLICATION_STYLE['axis_title_font_size'],
                    color='black',
                    weight='bold'
                )
            ),
            showline=True,
            linewidth=PUBLICATION_STYLE['axis_line_width'],
            linecolor='black',
            mirror=False,  # Только левая линия
            showgrid=False,
            zeroline=False,
            tickfont=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['tick_font_size'],
                color='black'
            ),
            ticks='outside',
            ticklen=PUBLICATION_STYLE['tick_length'],
            tickwidth=PUBLICATION_STYLE['tick_width'],
            tickcolor='black',
            gridcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        height=height,
        margin=dict(l=80, r=40, t=100, b=70),
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
            bgcolor='rgba(255,255,255,0.9)',
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top'
        ),
        showlegend=True
    )
    
    return fig

def create_combined_fitting_figure(title, x_title, y_title_top, y_title_bottom, width=None, height=None):
    """Create combined figure with main plot and residual plot below"""
    if width is None:
        width = PUBLICATION_STYLE['plot_width']
    if height is None:
        height = int(PUBLICATION_STYLE['plot_height'] * 1.4)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=('', ''),
        shared_xaxes=True
    )
    
    # Update layout for top plot
    fig.update_xaxes(
        title_text="",  # Нет подписи сверху
        showline=True,
        linewidth=PUBLICATION_STYLE['axis_line_width'],
        linecolor='black',
        mirror=False,  # Только нижняя линия
        showgrid=False,
        zeroline=False,
        tickfont=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['tick_font_size'],
            color='black'
        ),
        ticks='outside',
        ticklen=PUBLICATION_STYLE['tick_length'],
        tickwidth=PUBLICATION_STYLE['tick_width'],
        tickcolor='black',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text=y_title_top,
        title_font=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['axis_title_font_size'],
            color='black',
            weight='bold'
        ),
        showline=True,
        linewidth=PUBLICATION_STYLE['axis_line_width'],
        linecolor='black',
        mirror=False,  # Только левая линия
        showgrid=False,
        zeroline=False,
        tickfont=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['tick_font_size'],
            color='black'
        ),
        ticks='outside',
        ticklen=PUBLICATION_STYLE['tick_length'],
        tickwidth=PUBLICATION_STYLE['tick_width'],
        tickcolor='black',
        row=1, col=1
    )
    
    # Update layout for bottom plot (residuals)
    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['axis_title_font_size'],
            color='black',
            weight='bold'
        ),
        showline=True,
        linewidth=PUBLICATION_STYLE['axis_line_width'],
        linecolor='black',
        mirror=False,  # Только нижняя линия
        showgrid=False,
        zeroline=False,
        tickfont=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['tick_font_size'],
            color='black'
        ),
        ticks='outside',
        ticklen=PUBLICATION_STYLE['tick_length'],
        tickwidth=PUBLICATION_STYLE['tick_width'],
        tickcolor='black',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text=y_title_bottom,
        title_font=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['axis_title_font_size'],
            color='black',
            weight='bold'
        ),
        showline=True,
        linewidth=PUBLICATION_STYLE['axis_line_width'],
        linecolor='black',
        mirror=False,  # Только левая линия
        showgrid=False,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='black',
        tickfont=dict(
            family=PUBLICATION_STYLE['font_family'],
            size=PUBLICATION_STYLE['tick_font_size'],
            color='black'
        ),
        ticks='outside',
        ticklen=PUBLICATION_STYLE['tick_length'],
        tickwidth=PUBLICATION_STYLE['tick_width'],
        tickcolor='black',
        row=2, col=1
    )
    
    # Update overall layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['title_font_size'],
                color='black',
                weight='bold'
            ),
            x=0.5,
            xanchor='center',
            y=0.98
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=width,
        height=height,
        margin=dict(l=80, r=40, t=100, b=80),
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
            bgcolor='rgba(255,255,255,0.9)',
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top'
        ),
        showlegend=True
    )
    
    # Update layout to accommodate colorbar
    fig.update_layout(
        margin=dict(l=80, r=100, t=100, b=80),  # Увеличиваем правый отступ для цветовой шкалы
        legend=dict(
            font=dict(
                family=PUBLICATION_STYLE['font_family'],
                size=PUBLICATION_STYLE['legend_font_size'],
                color='black'
            ),
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.9)',
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top'
        )
    )
    
    return fig

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
# CALCULATION FUNCTIONS
# ============================================================================

def perform_calculations(data_input_text, uploaded_file, pH2O_value, Acc_value,
                        exclude_low_T_method1, exclude_high_T_method1,
                        exclude_low_T_method2, exclude_high_T_method2,
                        use_log_pH2O, colors, scale_design, palette_design):
    """Perform all calculations and return results"""
    
    # Parse and validate data
    data_array, load_message = parse_input_data(data_input_text, uploaded_file)
    is_valid, valid_message = validate_input_data(data_array, Acc_value)
    
    if not is_valid:
        return None, f"Validation error: {valid_message}", None, None
    
    # Temperature conversion
    T_C = data_array[:, 0]
    T_K = T_C + 273.15
    OH_exp = data_array[:, 1]
    
    n_points = len(data_array)
    
    # ========================================================================
    # METHOD 1: Kw Analysis
    # ========================================================================
    
    # Apply point exclusion - CORRECTED: exclude_low_T excludes from beginning, exclude_high_T excludes from end
    n_low_m1 = exclude_low_T_method1  # Points to exclude from start (low temperatures)
    n_high_m1 = exclude_high_T_method1  # Points to exclude from end (high temperatures)
    
    T_K_m1 = T_K[n_low_m1:len(T_K)-n_high_m1]
    OH_exp_m1 = OH_exp[n_low_m1:len(OH_exp)-n_high_m1]
    T_C_m1 = T_C[n_low_m1:len(T_C)-n_high_m1]
    
    # Calculate Kw with validation
    T_K_valid, OH_valid, Kw_valid = calculate_Kw_with_validation(
        T_K_m1, OH_exp_m1, pH2O_value, Acc_value
    )
    
    if len(T_K_valid) < 3:
        return None, "Insufficient valid points for Kw analysis", None, None
    
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
    
    # ========================================================================
    # METHOD 2: Direct Fitting
    # ========================================================================
    
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
        
    except Exception as e:
        # Use Method 1 parameters if fitting fails
        dH_method2, dS_method2 = dH_method1, dS_method1
        R2_method2 = 0
        SSE = np.nan
        RMSE = np.nan
        perr = [0, 0]
        dH_ci_m2 = 0
        dS_ci_m2 = 0
        OH_model_m2 = analytical_OH_numerical(T_K_m2, pH2O_value, Acc_value, dH_method2, dS_method2)
        residuals = OH_exp_m2 - OH_model_m2
    
    # Prepare results dictionary
    results = {
        'data': {
            'T_C': T_C,
            'T_K': T_K,
            'OH_exp': OH_exp,
            'n_points': n_points
        },
        'method1': {
            'T_C': T_C_m1,
            'T_K': T_K_m1,
            'OH_exp': OH_exp_m1,
            'T_valid': T_K_valid,
            'OH_valid': OH_valid,
            'Kw_valid': Kw_valid,
            'x_m1': x_m1,
            'ln_Kw': ln_Kw,
            'dH': dH_method1,
            'dH_ci': dH_ci,
            'dS': dS_method1,
            'dS_ci': dS_ci,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'std_err': std_err,
            'p_value': p_value,
            'n_valid': len(T_K_valid)
        },
        'method2': {
            'T_C': T_C_m2,
            'T_K': T_K_m2,
            'OH_exp': OH_exp_m2,
            'OH_model': OH_model_m2,
            'residuals': residuals,
            'dH': dH_method2,
            'dH_ci': dH_ci_m2,
            'dS': dS_method2,
            'dS_ci': dS_ci_m2,
            'R2': R2_method2,
            'SSE': SSE,
            'RMSE': RMSE,
            'perr': perr,
            'n_points': len(T_K_m2)
        },
        'parameters': {
            'pH2O': pH2O_value,
            'Acc': Acc_value,
            'exclude_low_m1': exclude_low_T_method1,
            'exclude_high_m1': exclude_high_T_method1,
            'exclude_low_m2': exclude_low_T_method2,
            'exclude_high_m2': exclude_high_T_method2,
            'use_log_pH2O': use_log_pH2O,
            'colors': colors,
            'scale_design': scale_design,
            'palette_design': palette_design
        }
    }
    
    return results, load_message, valid_message, data_array

def create_3d_surface(results, colors, palette_design, use_log_pH2O):
    """Create 3D surface plot"""
    if results is None:
        return None
    
    # Создаем сетку для температуры и pH2O
    T_min = min(results['data']['T_C'])
    T_max = max(results['data']['T_C'])
    T_range = np.linspace(T_min, T_max, 50)
    
    pH2O_min = 0.00001
    pH2O_max = 1
    
    if use_log_pH2O:
        # Логарифмическая шкала для pH2O
        pH2O_range = np.logspace(np.log10(pH2O_min), np.log10(pH2O_max), 50)
    else:
        # Линейная шкала для pH2O
        pH2O_range = np.linspace(pH2O_min, pH2O_max, 50)
    
    T_grid, pH2O_grid = np.meshgrid(T_range, pH2O_range)
    
    # Рассчитываем [OH] для каждой комбинации
    OH_grid = np.zeros_like(T_grid)
    for i in range(len(pH2O_range)):
        for j in range(len(T_range)):
            T_K_val = T_range[j] + 273.15
            OH_grid[i, j] = analytical_OH_numerical(
                T_K_val, 
                pH2O_range[i], 
                results['parameters']['Acc'],
                results['method2']['dH'],
                results['method2']['dS']
            )
    
    # Настройки палитры
    if palette_design == 'Viridis':
        colorscale = 'Viridis'
    elif palette_design == 'Plasma':
        colorscale = 'Plasma'
    elif palette_design == 'Inferno':
        colorscale = 'Inferno'
    elif palette_design == 'Magma':
        colorscale = 'Magma'
    elif palette_design == 'Cividis':
        colorscale = 'Cividis'
    else:  # 'Rainbow'
        colorscale = 'Rainbow'
    
    # Создаем 3D поверхность
    fig = go.Figure(data=[
        go.Surface(
            x=T_grid,
            y=pH2O_grid,
            z=OH_grid,
            colorscale=colorscale,
            opacity=0.7,  # Прозрачность поверхности
            showscale=True,
            contours={
                "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
            }
        )
    ])
    
    # Добавляем экспериментальные точки
    fig.add_trace(go.Scatter3d(
        x=results['data']['T_C'],
        y=[results['parameters']['pH2O']] * len(results['data']['T_C']),
        z=results['data']['OH_exp'],
        mode='markers',
        marker=dict(
            size=5,
            color=colors['experimental'],
            opacity=0.8,
            symbol='circle'
        ),
        name='Experimental data'
    ))
    
    # Определяем заголовок оси Y в зависимости от шкалы
    y_axis_title = 'log(pH₂O) (atm)' if use_log_pH2O else 'pH₂O (atm)'
    
    fig.update_layout(
        title=dict(
            text='3D Surface: [OH] = f(T, pH₂O)' + (' (log scale)' if use_log_pH2O else ''),
            font=dict(size=18, family='Times New Roman', color='black', weight='bold')
        ),
        scene=dict(
            xaxis=dict(
                title='Temperature (°C)',
                title_font=dict(size=16, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=12, family='Times New Roman', color='black'),
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False
            ),
            yaxis=dict(
                title=y_axis_title,
                title_font=dict(size=16, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=12, family='Times New Roman', color='black'),
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False,
                type='log' if use_log_pH2O else 'linear',
                tickmode='auto',
                exponentformat='power' if use_log_pH2O else 'none',
                showexponent='all',
                nticks=6 if use_log_pH2O else None
            ),
            zaxis=dict(
                title='[OH]',
                title_font=dict(size=16, family='Times New Roman', color='black', weight='bold'),
                tickfont=dict(size=12, family='Times New Roman', color='black'),
                showline=True,
                linewidth=2,
                linecolor='black',
                showgrid=False
            ),
            bgcolor='white'
        ),
        width=600,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        font=dict(family='Times New Roman', size=14, color='black')
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
        ["Text input", "Upload file"],
        key="data_source"
    )
    
    if data_source == "Upload file":
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["csv", "txt", "xlsx", "xls"],
            help="Supported: CSV, TXT, Excel. Data should contain temperature and [OH] concentration",
            key="file_uploader"
        )
        data_input_text = st.session_state.default_params['data']
    else:
        uploaded_file = None
        data_input_text = st.text_area(
            "Enter data (temperature °C and [OH]):",
            value=st.session_state.default_params['data'],
            height=150,
            help="Format: temperature concentration. Separator: space, tab or ;",
            key="data_input"
        )
    
    # System parameters
    st.subheader("System Parameters")
    pH2O_value = st.number_input(
        'pH₂O (atm):',
        min_value=1e-5,
        max_value=1.0,
        value=st.session_state.default_params['pH2O'],
        step=0.01,
        format="%.5f",
        key="pH2O_input"
    )
    
    Acc_value = st.number_input(
        '[Acc] = x:',
        min_value=0.01,
        max_value=5.99,
        value=st.session_state.default_params['Acc'],
        step=0.01,
        format="%.3f",
        help="Acceptor dopant concentration (0 < x < 6)",
        key="Acc_input"
    )
    
    # Parse data to determine number of points
    data_array, _ = parse_input_data(data_input_text, uploaded_file)
    n_total_points = len(data_array)
    
    # Calculate maximum exclusions (n/2 - 1)
    max_exclusion = max(0, n_total_points // 2 - 1) if n_total_points > 0 else 0
    
    # Fitting settings
    st.subheader("Fitting Settings")
    
    with st.expander("Method 1: Kw Analysis", expanded=True):
        st.markdown("**Points to exclude:**")
        col_m1_low, col_m1_high = st.columns(2)
        with col_m1_low:
            exclude_low_T_method1 = st.slider(
                'From start (low T):',
                min_value=0,
                max_value=max_exclusion,
                value=0,
                key="m1_low",
                help=f"Exclude first N points (0-{max_exclusion})"
            )
        with col_m1_high:
            exclude_high_T_method1 = st.slider(
                'From end (high T):',
                min_value=0,
                max_value=max_exclusion,
                value=0,
                key="m1_high",
                help=f"Exclude last N points (0-{max_exclusion})"
            )
    
    with st.expander("Method 2: Direct Fitting", expanded=True):
        st.markdown("**Points to exclude:**")
        col_m2_low, col_m2_high = st.columns(2)
        with col_m2_low:
            exclude_low_T_method2 = st.slider(
                'From start (low T):',
                min_value=0,
                max_value=max_exclusion,
                value=0,
                key="m2_low",
                help=f"Exclude first N points (0-{max_exclusion})"
            )
        with col_m2_high:
            exclude_high_T_method2 = st.slider(
                'From end (high T):',
                min_value=0,
                max_value=max_exclusion,
                value=0,
                key="m2_high",
                help=f"Exclude last N points (0-{max_exclusion})"
            )
    
    # Color settings
    st.subheader("Color Settings")
    
    col_exp, col_m1, col_m2 = st.columns(3)
    with col_exp:
        exp_color = st.color_picker("Experimental", "#000000", key="exp_color")
    with col_m1:
        m1_color = st.color_picker("Method 1", "#0000FF", key="m1_color")
    with col_m2:
        m2_color = st.color_picker("Method 2", "#FF0000", key="m2_color")
    
    colors = {
        'experimental': exp_color,
        'method1': m1_color,
        'method2': m2_color
    }
    
    # Scale and palette settings
    st.subheader("Visualization Settings")
    
    scale_design = st.selectbox(
        "Residuals scale design:",
        ["Linear", "Logarithmic", "Symmetric"],
        index=0,
        key="scale_design"
    )
    
    palette_design = st.selectbox(
        "3D surface palette:",
        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Rainbow"],
        index=0,
        key="palette_design"
    )
    
    # Additional options
    st.subheader("Additional Options")
    use_log_pH2O = st.checkbox("Logarithmic pH₂O scale in 3D", value=False, key="use_log_pH2O")
    
    # Comparison Method Options
    st.subheader("Comparison Method Options")
    show_method1_comparison = st.checkbox("Show Method 1 curve", value=True, key="show_method1_comparison")
    show_method2_comparison = st.checkbox("Show Method 2 curve", value=True, key="show_method2_comparison")
    show_experimental_comparison = st.checkbox("Show Experimental data", value=True, key="show_experimental_comparison")
    
    # Reset button
    if st.button("🔄 Reset to Defaults", use_container_width=True, key="reset_button"):
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
    st.markdown(f"**Total data points:** {n_total_points}")
    st.markdown(f"**Max exclusions:** {max_exclusion} points")
    st.markdown("**Version:** 2.3 | **Updated:** 2024")

# Main calculation and display
if n_total_points > 0:
    # Perform calculations
    results, load_message, valid_message, data_array = perform_calculations(
        data_input_text, uploaded_file, pH2O_value, Acc_value,
        exclude_low_T_method1, exclude_high_T_method1,
        exclude_low_T_method2, exclude_high_T_method2,
        use_log_pH2O, colors, scale_design, palette_design
    )
    
    if results is None:
        st.error(load_message)
    else:
        # Display status
        st.success(f"{load_message}. {valid_message}")
        
        # ====================================================================
        # METHOD 1 RESULTS
        # ====================================================================
        st.markdown("---")
        st.header("📈 Method 1: Equilibrium Constant Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ΔH°", f"{results['method1']['dH']/1000:.2f} ± {results['method1']['dH_ci']/1000:.2f} kJ/mol",
                     delta=f"{results['method1']['dH']:.0f} ± {results['method1']['dH_ci']:.0f} J/mol")
            st.metric("Points analyzed", results['method1']['n_valid'])
        
        with col2:
            st.metric("ΔS°", f"{results['method1']['dS']:.2f} ± {results['method1']['dS_ci']:.2f} J/(mol·K)")
            st.metric("R² coefficient", f"{results['method1']['r_squared']:.4f}")
        
        with col3:
            st.metric("Standard error", f"{results['method1']['std_err']:.4f}")
            st.metric("Significance level", f"p = {results['method1']['p_value']:.2e}")
        
        # ====================================================================
        # METHOD 2 RESULTS
        # ====================================================================
        st.markdown("---")
        st.header("📊 Method 2: Direct Profile Fitting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = "green" if results['method2']['R2'] > 0.95 else "orange" if results['method2']['R2'] > 0.9 else "red"
            st.markdown(f"<h3 style='color:{color}'>{results['method2']['R2']:.4f}</h3>", unsafe_allow_html=True)
            st.metric("R² coefficient", f"{results['method2']['R2']:.4f}")
            st.metric("RMSE", f"{results['method2']['RMSE']:.6f}" if not np.isnan(results['method2']['RMSE']) else "N/A")
        
        with col2:
            st.metric("ΔH°", f"{results['method2']['dH']/1000:.2f} ± {results['method2']['dH_ci']/1000:.2f} kJ/mol",
                     delta=f"{results['method2']['dH']:.0f} ± {results['method2']['perr'][0]:.0f} J/mol" if 'perr' in results['method2'] else f"{results['method2']['dH']:.0f} J/mol")
            st.metric("Points analyzed", results['method2']['n_points'])
        
        with col3:
            st.metric("ΔS°", f"{results['method2']['dS']:.2f} ± {results['method2']['dS_ci']:.2f} J/(mol·K)",
                     delta=f"± {results['method2']['perr'][1]:.2f}" if 'perr' in results['method2'] else "")
            st.metric("SSE", f"{results['method2']['SSE']:.6f}" if not np.isnan(results['method2']['SSE']) else "N/A")
        
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
                f"{results['method1']['dH']/1000:.2f}",
                f"±{results['method1']['dH_ci']/1000:.2f}",
                f"{results['method1']['dS']:.2f}",
                f"±{results['method1']['dS_ci']:.2f}",
                f"{results['method1']['r_squared']:.4f}",
                f"{results['method1']['n_valid']}",
                f"std_err={results['method1']['std_err']:.4f}"
            ],
            'Method 2': [
                f"{results['method2']['dH']/1000:.2f}",
                f"±{results['method2']['dH_ci']/1000:.2f}",
                f"{results['method2']['dS']:.2f}",
                f"±{results['method2']['dS_ci']:.2f}",
                f"{results['method2']['R2']:.4f}",
                f"{results['method2']['n_points']}",
                f"RMSE={results['method2']['RMSE']:.6f}" if not np.isnan(results['method2']['RMSE']) else "N/A"
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
                    'pH2O': results['parameters']['pH2O'],
                    'Acc': results['parameters']['Acc'],
                    'temperature_unit': 'Celsius'
                },
                'method1': {
                    'dH_kJ_mol': float(results['method1']['dH']/1000),
                    'dH_CI_kJ_mol': float(results['method1']['dH_ci']/1000),
                    'dS_J_molK': float(results['method1']['dS']),
                    'dS_CI_J_molK': float(results['method1']['dS_ci']),
                    'R2': float(results['method1']['r_squared']),
                    'n_points': int(results['method1']['n_valid']),
                    'excluded_low': results['parameters']['exclude_low_m1'],
                    'excluded_high': results['parameters']['exclude_high_m1']
                },
                'method2': {
                    'dH_kJ_mol': float(results['method2']['dH']/1000),
                    'dH_CI_kJ_mol': float(results['method2']['dH_ci']/1000),
                    'dS_J_molK': float(results['method2']['dS']),
                    'dS_CI_J_molK': float(results['method2']['dS_ci']),
                    'R2': float(results['method2']['R2']),
                    'RMSE': float(results['method2']['RMSE']) if not np.isnan(results['method2']['RMSE']) else None,
                    'n_points': int(results['method2']['n_points']),
                    'excluded_low': results['parameters']['exclude_low_m2'],
                    'excluded_high': results['parameters']['exclude_high_m2']
                },
                'visualization': {
                    'colors': colors,
                    'scale_design': scale_design,
                    'palette_design': palette_design
                }
            }
            st.markdown(get_json_download_link(export_data, "parameters.json"), unsafe_allow_html=True)
        
        # ====================================================================
        # VISUALIZATION
        # ====================================================================
        st.markdown("---")
        st.header("📊 Visualization")
        
        # 1. Experimental Data
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_publication_figure(
                "Experimental Data",
                "Temperature (°C)",
                "[OH]"
            )
            
            fig1.add_trace(go.Scatter(
                x=results['data']['T_C'],
                y=results['data']['OH_exp'],
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size'],
                    color=colors['experimental'],
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
                annotation_position="top right",
                annotation_font=dict(size=12, color='red')
            )
            
            fig1.add_hline(
                y=0, 
                line=dict(color='blue', width=1, dash='dash'),
                annotation_text='[OH] = 0',
                annotation_position="bottom right",
                annotation_font=dict(size=12, color='blue')
            )
            
            st.plotly_chart(fig1, use_container_width=False)
        
        # 2. Method 1: ln(Kw) vs 1000/T
        with col2:
            fig2 = create_publication_figure(
                "Method 1: ln(Kw) vs 1000/T",
                "1000/T (K⁻¹)",
                "ln(Kw)"
            )
            
            fig2.add_trace(go.Scatter(
                x=results['method1']['x_m1'],
                y=results['method1']['ln_Kw'],
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size'],
                    color=colors['experimental'],
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name='Data points',
                showlegend=True
            ))
            
            # Regression line
            x_min = min(results['method1']['x_m1'])
            x_max = max(results['method1']['x_m1'])
            x_fit = np.linspace(x_min, x_max, 100)
            y_fit = results['method1']['slope'] * x_fit + results['method1']['intercept']
            
            fig2.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                line=dict(
                    color=colors['method1'], 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name=f'Linear fit: R² = {results["method1"]["r_squared"]:.4f}<br>ΔH = {results["method1"]["dH"]/1000:.1f} kJ/mol',
                showlegend=True
            ))
            
            st.plotly_chart(fig2, use_container_width=False)
        
        st.markdown("### Method 2: Profile Fitting with Residuals")
        col3, col4 = st.columns([3, 1])
        
        with col3:
            fig3 = create_combined_fitting_figure(
                "Method 2: Profile Fitting with Residuals",
                "Temperature (°C)",
                "[OH]",
                "[OH]<sub>exp</sub> - [OH]<sub>model</sub>"
            )
        
            # Top plot: Model fit
            T_fit = np.linspace(min(results['data']['T_C']), max(results['data']['T_C']), 200)
            T_K_fit = T_fit + 273.15
            OH_fit = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, 
                                            results['method2']['dH'], results['method2']['dS'])
            
            # Add model curve
            fig3.add_trace(go.Scatter(
                x=T_fit,
                y=OH_fit,
                mode='lines',
                line=dict(
                    color=colors['method2'], 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name=f'Model fit: R² = {results["method2"]["R2"]:.4f}',
                showlegend=True
            ), row=1, col=1)
            
            # Добавляем экспериментальные точки в верхний график
            fig3.add_trace(go.Scatter(
                x=results['method2']['T_C'],
                y=results['method2']['OH_exp'],
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size'],
                    color=colors['experimental'],
                    symbol='circle',
                    line=dict(width=1, color='black')
                ),
                name='Experimental data',
                showlegend=True
            ), row=1, col=1)
            
            # Bottom plot: Residuals с тепловой шкалой
            residuals = results['method2']['residuals']
            if len(residuals) > 0:
                abs_residuals = np.abs(residuals)
                max_abs = np.max(abs_residuals) if np.max(abs_residuals) > 0 else 1.0
                
                # Создаем цветовую шкалу
                # Создаем dummy trace для colorbar в верхнем графике
                fig3.add_trace(
                    go.Scatter(
                        x=[0], y=[0],                   # хотя бы одна точка (можно и невидимая)
                        mode='markers',
                        marker=dict(
                            size=0,                     # можно сделать невидимым
                            color=[0],                  # минимальное значение
                            colorscale='RdBu_r',
                            cmin=0,
                            cmax=max_abs,
                            showscale=True,
                            colorbar=dict(
                                title='|Δ[OH]|',
                                titleside='right',
                                thickness=15,
                                len=0.3,
                                x=1.02,
                                y=0.5,
                                yanchor='middle'
                            )
                        ),
                        showlegend=False
                    ),
                    row=1, col=1   # или row=2, col=1 — без разницы, лишь бы был в фигуре
                )
                
                # Добавляем остатки с цветом по величине
                colors_normalized = abs_residuals / max_abs
                # Используем встроенную colorscale через числовые значения
                fig3.add_trace(go.Scatter(
                    x=results['method2']['T_C'],
                    y=results['method2']['residuals'],
                    mode='markers',
                    marker=dict(
                        size=PUBLICATION_STYLE['marker_size'] - 2,
                        color=abs_residuals,  # Используем числовые значения
                        colorscale='RdBu_r',  # Обратная шкала Red-Blue
                        cmin=0,
                        cmax=max_abs,
                        showscale=False,  # Не показываем шкалу здесь
                        symbol='circle',
                        line=dict(width=0.5, color='black')
                    ),
                    name='Residuals',
                    showlegend=False
                ), row=2, col=1)
            else:
                # Fallback если нет остатков
                fig3.add_trace(go.Scatter(
                    x=results['method2']['T_C'],
                    y=results['method2']['residuals'],
                    mode='markers',
                    marker=dict(
                        size=PUBLICATION_STYLE['marker_size'] - 2,
                        color='red',
                        symbol='circle',
                        line=dict(width=0.5, color='black')
                    ),
                    name='Residuals',
                    showlegend=False
                ), row=2, col=1)
            
            # Add zero line to residuals
            fig3.add_hline(
                y=0, 
                line=dict(color='black', width=1),
                row=2, col=1
            )
            
            st.plotly_chart(fig3, use_container_width=False)
        
            # Model curve
            T_fit = np.linspace(min(results['data']['T_C']), max(results['data']['T_C']), 200)
            T_K_fit = T_fit + 273.15
            OH_fit = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, 
                                            results['method2']['dH'], results['method2']['dS'])
            
            fig3.add_trace(go.Scatter(
                x=T_fit,
                y=OH_fit,
                mode='lines',
                line=dict(
                    color=colors['method2'], 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name=f'Model fit: R² = {results["method2"]["R2"]:.4f}<br>ΔH = {results["method2"]["dH"]/1000:.1f} kJ/mol',
                showlegend=True
            ), row=1, col=1)
            
            # Bottom plot: Residuals
            fig3.add_trace(go.Scatter(
                x=results['method2']['T_C'],
                y=results['method2']['residuals'],
                mode='markers',
                marker=dict(
                    size=PUBLICATION_STYLE['marker_size'] - 2,
                    color='red',
                    symbol='circle',
                    line=dict(width=0.5, color='black')
                ),
                name='Residuals',
                showlegend=False
            ), row=2, col=1)
            
            # Add zero line to residuals
            fig3.add_hline(
                y=0, 
                line=dict(color='black', width=1),
                row=2, col=1
            )
            
            st.plotly_chart(fig3, use_container_width=False)
        
        with col4:
            st.markdown("#### Residuals Scale")
            if scale_design == "Linear":
                st.markdown("**Linear scale**")
                st.markdown("Residuals shown as-is")
            elif scale_design == "Logarithmic":
                st.markdown("**Logarithmic scale**")
                st.markdown("log(|residuals| + ε)")
            else:  # Symmetric
                st.markdown("**Symmetric scale**")
                st.markdown("± symmetric range")
            
            st.markdown("---")
            st.markdown("#### Legend")
            st.markdown(f"<span style='color:{colors['experimental']}'>●</span> Experimental data", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{colors['method1']}'>━━━</span> Method 1", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{colors['method2']}'>━━━</span> Method 2", unsafe_allow_html=True)
        
        # 4. Method Comparison
        st.markdown("### Comparison of Methods")
        col5, col6 = st.columns(2)
        
        with col5:
            fig4 = create_publication_figure(
                "Comparison of Methods",
                "Temperature (°C)",
                "[OH]"
            )
            
            # Method 1 curve (только если выбрано в виджете)
            if show_method1_comparison:
                OH_fit_m1 = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, 
                                                  results['method1']['dH'], results['method1']['dS'])
                
                fig4.add_trace(go.Scatter(
                    x=T_fit,
                    y=OH_fit_m1,
                    mode='lines',
                    line=dict(
                        color=colors['method1'], 
                        width=PUBLICATION_STYLE['line_width'],
                        dash='dash'
                    ),
                    name=f'Method 1: ΔH = {results["method1"]["dH"]/1000:.1f} kJ/mol',
                    showlegend=True
                ))
            
            # Method 2 curve (только если выбрано в виджете)
            if show_method2_comparison:
                OH_fit_m2 = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, 
                                                  results['method2']['dH'], results['method2']['dS'])
                
                # Определяем легенду в зависимости от видимости Method 1
                if show_method1_comparison:
                    legend_name = f'Method 2: ΔH = {results["method2"]["dH"]/1000:.1f} kJ/mol'
                else:
                    legend_name = 'Modelled data'
                
                fig4.add_trace(go.Scatter(
                    x=T_fit,
                    y=OH_fit_m2,
                    mode='lines',
                    line=dict(
                        color=colors['method2'], 
                        width=PUBLICATION_STYLE['line_width']
                    ),
                    name=legend_name,
                    showlegend=True
                ))
            
            # Experimental points (только если выбрано в виджете)
            if show_experimental_comparison:
                fig4.add_trace(go.Scatter(
                    x=results['data']['T_C'],
                    y=results['data']['OH_exp'],
                    mode='markers',
                    marker=dict(
                        size=PUBLICATION_STYLE['marker_size']-2,
                        color=colors['experimental'],
                        symbol='circle',
                        opacity=0.7,
                        line=dict(width=0.5, color='black')
                    ),
                    name='Experimental data',
                    showlegend=True
                ))
            
            # Если ничего не выбрано, показываем сообщение
            if not (show_method1_comparison or show_method2_comparison or show_experimental_comparison):
                fig4.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="Select at least one curve to display",
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
            
            st.plotly_chart(fig4, use_container_width=False)
        
        # 5. Temperature Dependence of Kw
        with col6:
            fig5 = create_publication_figure(
                "Temperature Dependence of Kw",
                "Temperature (°C)",
                "ln(Kw)"
            )
            
            # Calculate Kw for both methods
            Kw_m1 = np.exp(-results['method1']['dH']/(R * T_K_fit) + results['method1']['dS']/R)
            Kw_m2 = np.exp(-results['method2']['dH']/(R * T_K_fit) + results['method2']['dS']/R)
            
            fig5.add_trace(go.Scatter(
                x=T_fit,
                y=np.log(Kw_m1),
                mode='lines',
                line=dict(
                    color=colors['method1'], 
                    width=PUBLICATION_STYLE['line_width'],
                    dash='dash'
                ),
                name='Method 1',
                showlegend=True
            ))
            
            fig5.add_trace(go.Scatter(
                x=T_fit,
                y=np.log(Kw_m2),
                mode='lines',
                line=dict(
                    color=colors['method2'], 
                    width=PUBLICATION_STYLE['line_width']
                ),
                name='Method 2',
                showlegend=True
            ))
            
            # Experimental Kw points
            if len(results['method1']['T_valid']) > 0:
                fig5.add_trace(go.Scatter(
                    x=results['method1']['T_valid'] - 273.15,
                    y=np.log(results['method1']['Kw_valid']),
                    mode='markers',
                    marker=dict(
                        size=PUBLICATION_STYLE['marker_size']-2,
                        color=colors['experimental'],
                        symbol='circle',
                        line=dict(width=0.5, color='black')
                    ),
                    name='Experimental (Method 1)',
                    showlegend=True
                ))
            
            st.plotly_chart(fig5, use_container_width=False)
        
        # 6. 3D Surface Plot (обязательно генерируется)
        st.markdown("### 3D Surface Plot")
        fig6 = create_3d_surface(results, colors, palette_design, use_log_pH2O)
        if fig6:
            st.plotly_chart(fig6, use_container_width=True)
        
        # ====================================================================
        # COMMENTS AND RECOMMENDATIONS
        # ====================================================================
        st.markdown("---")
        st.header("💡 Comments and Recommendations")
        
        recommendations = []
        
        # Fitting quality
        r2_m1 = results['method1']['r_squared']
        r2_m2 = results['method2']['R2']
        
        if r2_m1 > 0.98 and r2_m2 > 0.98:
            recommendations.append("✅ Excellent agreement of both methods with data")
        elif r2_m1 > 0.95 and r2_m2 > 0.95:
            recommendations.append("✅ Good agreement of methods with data")
        elif r2_m1 < 0.9 or r2_m2 < 0.9:
            recommendations.append("⚠️ Consider excluding more points or checking data")
        
        # Parameter consistency
        if results['method1']['dH'] != 0:
            diff_percent = abs(results['method2']['dH'] - results['method1']['dH']) / abs(results['method1']['dH']) * 100
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
        - Method 1: ΔH° = {results['method1']['dH']/1000:.1f} ± {results['method1']['dH_ci']/1000:.2f} kJ/mol
        - Method 2: ΔH° = {results['method2']['dH']/1000:.1f} ± {results['method2']['dH_ci']/1000:.2f} kJ/mol
        
        **For modeling:**
        - Recommended: Method 2 (direct fitting)
        - ΔH° = {results['method2']['dH']/1000:.1f} ± {results['method2']['dH_ci']/1000:.2f} kJ/mol
        - ΔS° = {results['method2']['dS']:.1f} ± {results['method2']['dS_ci']:.1f} J/(mol·K)
        
        **Average values:**
        - ΔH° = {(results['method1']['dH'] + results['method2']['dH'])/2000:.1f} kJ/mol
        - ΔS° = {(results['method1']['dS'] + results['method2']['dS'])/2:.1f} J/(mol·K)
        """)
else:
    # Initial information
    st.markdown("""
    ## 📖 Instructions
    
    1. **Load data** in text field or choose file (CSV, TXT, Excel)
    2. **Set system parameters**: pH₂O and acceptor concentration [Acc]
    3. **Configure fitting**: exclude extreme points if necessary
    4. **Graphs update automatically** when parameters change
    
    ## 🎯 Key Features
    
    ✅ **Real-time calculation** - no calculate button needed  
    ✅ **Reliable numerical solution** instead of analytical formulas  
    ✅ **Errors and confidence intervals** for all parameters  
    ✅ **File upload** in various formats  
    ✅ **Data validation** with physical correctness check  
    ✅ **Export results** to CSV, JSON  
    ✅ **Publication-quality graphs** with correct aspect ratio (3:4)  
    ✅ **Black axes and ticks** for scientific publications  
    ✅ **Bold axis titles** with larger font size  
    ✅ **Rietveld-style combined plots** (fitting + residuals)  
    ✅ **Correct point exclusion logic**  
    ✅ **Customizable colors** for experimental data and models  
    ✅ **3D surface plot** with adjustable palette  
    ✅ **Compact graph layout** with proper sizing  
    ✅ **12 ppt axis values, 16 ppt axis labels**  
    
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

# Information
st.markdown("---")
st.markdown("*Application automatically updates calculations when parameters change*")

















