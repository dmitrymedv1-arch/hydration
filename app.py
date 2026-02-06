import streamlit as stimport
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
    page_title="Термодинамика гидратации",
    page_icon="🔬",
    layout="wide"
)

# Константы
R = 8.314  # J/(mol·K)

# Настройки стиля
PUBLICATION_STYLE = {
    'font_family': 'Arial',
    'font_size': 14,
    'title_font_size': 16,
    'axis_title_font_size': 14,
    'tick_font_size': 12,
    'legend_font_size': 12,
    'line_width': 2,
    'marker_size': 8,
}

# Инициализация session state
if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []
if 'default_params' not in st.session_state:
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

# Функции для численных решений
@st.cache_data(ttl=300)
def calculate_equilibrium_oh(K, Acc, pH2O):
    """Надежное численное решение для равновесной концентрации [OH]"""
    def f(oh):
        return 4 * oh**2 - K * pH2O * (Acc - oh) * (6 - Acc - oh)
    
    try:
        # Пробуем метод brentq с физически осмысленными границами
        sol = root_scalar(
            f, 
            bracket=[1e-10, Acc - 1e-10],  # ОH должно быть между 0 и Acc
            method='brentq',
            xtol=1e-12,
            rtol=1e-12
        )
        if sol.converged and 0 < sol.root < Acc:
            return float(sol.root)
        else:
            return np.nan
    except (ValueError, RuntimeError):
        # Fallback: метод бисекции
        try:
            # Проверяем знаки на границах
            f_low = f(1e-10)
            f_high = f(Acc - 1e-10)
            
            if f_low * f_high > 0:
                # Нет корня в интервале
                return np.nan
            
            # Итеративная бисекция
            low, high = 1e-10, Acc - 1e-10
            for _ in range(50):
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
    """Аналитическое выражение для [OH] с численным решением"""
    # Расчет Kw
    Kw = np.exp(-dH/(R * T_K) + dS/R)
    K = Kw * pH2O
    
    # Для скалярного ввода
    if isinstance(T_K, (int, float)):
        return calculate_equilibrium_oh(K, Acc, pH2O)
    
    # Для массива
    results = np.zeros_like(K)
    for i in range(len(K)):
        results[i] = calculate_equilibrium_oh(K[i], Acc, pH2O)
    
    return results

def calculate_Kw_with_validation(T_K, OH, pH2O, Acc):
    """Расчет Kw с валидацией данных"""
    # Проверка физических ограничений
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
    
    # Расчет Kw
    numerator = 4 * OH_valid**2
    denominator = pH2O * (Acc - OH_valid) * (6 - Acc - OH_valid)
    
    # Защита от деления на ноль и очень малых/больших значений
    mask_finite = (
        (denominator > 1e-20) & 
        (numerator > 0) &
        (denominator < 1e20)
    )
    
    if not np.any(mask_finite):
        return np.array([]), np.array([]), np.array([])
    
    T_K_final = T_K_valid[mask_finite]
    OH_final = OH_valid[mask_finite]
    Kw_final = numerator[mask_finite] / denominator[mask_finite]
    
    # Дополнительная фильтрация экстремальных значений
    mask_reasonable = (Kw_final > 1e-20) & (Kw_final < 1e20)
    
    return (
        T_K_final[mask_reasonable], 
        OH_final[mask_reasonable], 
        Kw_final[mask_reasonable]
    )

# Функции для обработки данных
def parse_input_data(input_text, file_uploader=None):
    """Парсинг входных данных из текста или файла"""
    if file_uploader is not None:
        try:
            if file_uploader.name.endswith('.csv'):
                df = pd.read_csv(file_uploader)
            elif file_uploader.name.endswith('.txt'):
                df = pd.read_csv(file_uploader, sep=None, engine='python')
            elif file_uploader.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_uploader)
            else:
                raise ValueError("Неподдерживаемый формат файла")
            
            # Автодетект столбцов
            temp_col = None
            oh_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(word in col_lower for word in ['temp', 't', 'temperature', '°c']):
                    temp_col = col
                elif any(word in col_lower for word in ['oh', 'conc', 'concentration', '[oh]']):
                    oh_col = col
            
            if temp_col is None or oh_col is None:
                # Берем первые два столбца
                temp_col, oh_col = df.columns[:2]
            
            data = df[[temp_col, oh_col]].values
            return data, f"Загружен файл: {file_uploader.name}, {len(data)} точек"
            
        except Exception as e:
            st.warning(f"Ошибка чтения файла: {e}. Используются текстовые данные.")
    
    # Парсинг из текста
    lines = input_text.strip().split('\n')
    data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Замена разделителей
        line = line.replace(',', '.').replace(';', ' ').replace('\t', ' ')
        
        # Удаление лишних пробелов
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
        # Демо-данные
        data = [[20, 0.15], [100, 0.12], [200, 0.10], [300, 0.08], 
                [400, 0.06], [500, 0.04], [600, 0.02], [700, 0.01], [800, 0.005]]
        return np.array(data), "Используются демо-данные"
    
    return np.array(data), f"Загружено {len(data)} точек из текста"

def validate_input_data(data_array, Acc):
    """Валидация входных данных с учетом экспериментальной погрешности"""
    if data_array is None or len(data_array) == 0:
        return False, "Нет данных для анализа"
    
    T_C = data_array[:, 0]
    OH = data_array[:, 1]
    
    issues = []
    
    # Проверка температуры
    if np.any(T_C < -273.15):
        issues.append("Есть температуры ниже абсолютного нуля")
    if np.any(T_C > 2000):
        issues.append("Есть подозрительно высокие температуры (>2000°C)")
    
    # Проверка концентраций
    if np.any(OH < 0):
        issues.append("Есть отрицательные концентрации [OH] (физически невозможно)")
    if np.any(OH > Acc * 1.01):  # Разрешаем 1% превышение из-за погрешности
        issues.append(f"Есть концентрации [OH] > [Acc] ({Acc:.3f})")
    
    # Проверка монотонности с учетом экспериментальной погрешности
    if len(T_C) > 1:
        sorted_idx = np.argsort(T_C)
        T_sorted = T_C[sorted_idx]
        OH_sorted = OH[sorted_idx]
        
        # Рассчитываем относительные изменения
        for i in range(1, len(T_sorted)):
            if OH_sorted[i] > OH_sorted[i-1] * 1.01:  # Разрешаем 1% рост
                issues.append(f"Концентрация растет с температурой: {T_sorted[i-1]}→{T_sorted[i]}°C, {OH_sorted[i-1]:.6f}→{OH_sorted[i]:.6f}")
                break
    
    if issues:
        # Собираем только критические ошибки
        critical_issues = [issue for issue in issues if "отрицательные" in issue or "[OH] > [Acc]" in issue]
        if critical_issues:
            return False, "; ".join(critical_issues[:3])  # Ограничиваем количество сообщений
        else:
            # Для некритических проблем показываем предупреждение, но продолжаем расчет
            return True, f"Данные валидны (замечания: {issues[0]})"
    
    return True, "Данные валидны"

def check_monotonicity_with_tolerance(T, OH, tolerance=0.02):
    """
    Проверка монотонности с допуском на экспериментальную погрешность
    
    Parameters:
    -----------
    T : array-like
        Температуры
    OH : array-like
        Концентрации
    tolerance : float
        Допустимое относительное отклонение от монотонности (2% по умолчанию)
    
    Returns:
    --------
    is_monotonic : bool
        True если данные монотонны в пределах допуска
    violations : list
        Список нарушений монотонности
    """
    if len(T) < 2:
        return True, []
    
    # Сортируем по температуре
    sorted_idx = np.argsort(T)
    T_sorted = T[sorted_idx]
    OH_sorted = OH[sorted_idx]
    
    violations = []
    
    for i in range(1, len(T_sorted)):
        # Разрешаем небольшой рост в пределах погрешности
        max_allowed = OH_sorted[i-1] * (1 + tolerance)
        
        if OH_sorted[i] > max_allowed:
            # Рассчитываем статистику для контекста
            avg_oh = (OH_sorted[i-1] + OH_sorted[i]) / 2
            relative_change = (OH_sorted[i] - OH_sorted[i-1]) / avg_oh * 100
            
            violations.append({
                'index': i,
                'T_low': T_sorted[i-1],
                'T_high': T_sorted[i],
                'OH_low': OH_sorted[i-1],
                'OH_high': OH_sorted[i],
                'relative_change': relative_change,
                'tolerance': tolerance * 100
            })
    
    return len(violations) == 0, violations

# Функции для экспорта
def get_table_download_link(df, filename="results.csv"):
    """Генерирует ссылку для скачивания таблицы"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Скачать CSV</a>'
    return href

def get_json_download_link(data, filename="parameters.json"):
    """Генерирует ссылку для скачивания JSON"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">📥 Скачать JSON</a>'
    return href

# Основной интерфейс
st.title("🔬 Определение термодинамических параметров гидратации")
st.markdown("""
*Термодинамический анализ AB₁₋ₓAccₓO₃₋ₓ/₂ на основе температурного профиля концентрации протонов*
""")

# Сайдбар с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка данных
    st.subheader("Загрузка данных")
    data_source = st.radio(
        "Источник данных:",
        ["Текстовый ввод", "Загрузить файл"]
    )
    
    if data_source == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=["csv", "txt", "xlsx", "xls"],
            help="Поддерживаются CSV, TXT, Excel. Данные должны содержать температуру и концентрацию [OH]"
        )
        data_input_text = st.session_state.default_params['data']
    else:
        uploaded_file = None
        data_input_text = st.text_area(
            "Введите данные (температура °C и [OH]):",
            value=st.session_state.default_params['data'],
            height=150,
            help="Формат: температура концентрация. Разделитель: пробел, табуляция или ;"
        )
    
    # Параметры системы
    st.subheader("Параметры системы")
    pH2O_value = st.number_input(
        'pH₂O (атм):',
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
        help="Концентрация акцепторного допанта (0 < x < 6)"
    )
    
    # Настройки фитинга
    st.subheader("Настройки фитинга")
    with st.expander("Метод 1: Анализ через Kw", expanded=True):
        exclude_low_T_method1 = st.slider(
            'Исключить точек с низкой T:',
            min_value=0,
            max_value=10,
            value=0,
            key="m1_low"
        )
        exclude_high_T_method1 = st.slider(
            'Исключить точек с высокой T:',
            min_value=0,
            max_value=10,
            value=0,
            key="m1_high"
        )
    
    with st.expander("Метод 2: Прямой фитинг", expanded=True):
        exclude_low_T_method2 = st.slider(
            'Исключить точек с низкой T:',
            min_value=0,
            max_value=10,
            value=0,
            key="m2_low"
        )
        exclude_high_T_method2 = st.slider(
            'Исключить точек с высокой T:',
            min_value=0,
            max_value=10,
            value=0,
            key="m2_high"
        )
    
    # Дополнительные опции
    st.subheader("Дополнительно")
    show_intermediate = st.checkbox("Показать промежуточные расчеты", value=False)
    calculate_3d = st.checkbox("Рассчитать 3D поверхности", value=True)
    use_log_pH2O = st.checkbox("Логарифмическая шкала pH₂O в 3D", value=False)
    
    # Кнопки управления
    col1, col2 = st.columns(2)
    with col1:
        reset_btn = st.button("🔄 Сбросить", use_container_width=True)
    with col2:
        calculate_btn = st.button("🚀 Рассчитать", type="primary", use_container_width=True)
    
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
    
    # Информация
    st.markdown("---")
    st.markdown("**Версия:** 2.0 | **Обновлено:** 2024")
    st.markdown("""
    **Ссылки:**
    - [Исходный код](https://github.com)
    - [Документация](https://example.com)
    - [DOI: 10.xxxx/xxxxxx](https://doi.org/10.xxxx/xxxxxx)
    """)

# Основное окно расчетов
if calculate_btn:
    try:
        with st.spinner('Обработка данных...'):
            # Парсинг и валидация данных
            data_array, load_message = parse_input_data(data_input_text, uploaded_file)
            
            # Базовая валидация
            is_valid, valid_message = validate_input_data(data_array, Acc_value)
            
            if not is_valid:
                st.error(f"Ошибка валидации: {valid_message}")
                
                # Показываем данные для отладки
                with st.expander("📊 Загруженные данные для отладки"):
                    df_debug = pd.DataFrame(data_array, columns=['Температура (°C)', '[OH]'])
                    df_debug['ΔT'] = df_debug['Температура (°C)'].diff().fillna(0)
                    df_debug['Δ[OH]'] = df_debug['[OH]'].diff().fillna(0)
                    df_debug['Отн. изменение [OH] (%)'] = (df_debug['Δ[OH]'] / df_debug['[OH]'].shift(1) * 100).fillna(0)
                    st.dataframe(df_debug, use_container_width=True)
                
                st.stop()
            
            # Дополнительная проверка монотонности с выводом деталей
            T_C = data_array[:, 0]
            OH_exp = data_array[:, 1]
            
            is_monotonic, violations = check_monotonicity_with_tolerance(T_C, OH_exp, tolerance=0.02)
            
            if not is_monotonic:
                st.warning(f"⚠️ Нарушение монотонности обнаружено в {len(violations)} точках")
                
                with st.expander("🔍 Детали нарушений монотонности"):
                    for i, violation in enumerate(violations[:3]):  # Показываем только первые 3
                        st.markdown(f"""
                        **Нарушение {i+1}:**
                        - Температура: {violation['T_low']:.1f} → {violation['T_high']:.1f} °C
                        - [OH]: {violation['OH_low']:.6f} → {violation['OH_high']:.6f}
                        - Относительное изменение: **{violation['relative_change']:.2f}%**
                        - Допустимый предел: {violation['tolerance']:.1f}%
                        """)
                    
                    if len(violations) > 3:
                        st.info(f"... и ещё {len(violations) - 3} нарушений")
                
                # Предлагаем опции пользователю
                col1, col2 = st.columns(2)
                with col1:
                    continue_anyway = st.checkbox("Продолжить расчет несмотря на нарушения", value=True)
                with col2:
                    if st.button("Автоматически исключить выбросы"):
                        # Простая логика для исключения выбросов
                        st.info("Функция в разработке...")
                
                if not continue_anyway:
                    st.stop()
            
            st.success(f"{load_message}. {valid_message}")
            
            # Отображение данных
            if show_intermediate:
                with st.expander("📊 Загруженные данные", expanded=True):
                    df_data = pd.DataFrame(data_array, columns=['Температура (°C)', '[OH]'])
                    st.dataframe(df_data, use_container_width=True)
            
            # Преобразование температур
            T_C = data_array[:, 0]
            T_K = T_C + 273.15
            OH_exp = data_array[:, 1]
            
            # Метод 1: Анализ через Kw
            st.markdown("---")
            st.header("📈 Метод 1: Анализ через константу равновесия Kw")
            
            # Применение исключения точек
            n_low_m1 = exclude_low_T_method1
            n_high_m1 = exclude_high_T_method1
            
            T_K_m1 = T_K[n_low_m1:len(T_K)-n_high_m1]
            OH_exp_m1 = OH_exp[n_low_m1:len(OH_exp)-n_high_m1]
            T_C_m1 = T_C[n_low_m1:len(T_C)-n_high_m1]
            
            # Расчет Kw с валидацией
            T_K_valid, OH_valid, Kw_valid = calculate_Kw_with_validation(
                T_K_m1, OH_exp_m1, pH2O_value, Acc_value
            )
            
            if len(T_K_valid) < 3:
                st.error("Недостаточно валидных точек для анализа Kw. Проверьте данные.")
                st.stop()
            
            # Линейная регрессия
            ln_Kw = np.log(Kw_valid)
            x_m1 = 1000 / T_K_valid
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_m1, ln_Kw)
            
            # Расчет параметров с погрешностями
            dH_method1 = -slope * R * 1000  # Дж/моль
            dS_method1 = intercept * R      # Дж/(моль·К)
            
            # Погрешности
            dH_err = std_err * R * 1000
            dS_err = std_err * R
            
            # 95% доверительные интервалы
            n = len(x_m1)
            t_val = stats.t.ppf(0.975, n-2)  # t-статистика для 95% CI
            
            dH_ci = t_val * dH_err
            dS_ci = t_val * dS_err
            
            # Отображение результатов методом 1
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ΔH°", f"{dH_method1/1000:.2f} ± {dH_ci/1000:.2f} кДж/моль",
                         delta=f"{dH_method1:.0f} ± {dH_ci:.0f} Дж/моль")
                st.metric("Точек для анализа", len(T_K_valid))
            
            with col2:
                st.metric("ΔS°", f"{dS_method1:.2f} ± {dS_ci:.2f} Дж/(моль·К)")
                st.metric("Коэффициент R²", f"{r_value**2:.4f}")
            
            with col3:
                st.metric("Стандартная ошибка", f"{std_err:.4f}")
                st.metric("Уровень значимости", f"p = {p_value:.2e}")
            
            if show_intermediate:
                with st.expander("🔍 Промежуточные расчеты (Метод 1)"):
                    df_kw = pd.DataFrame({
                        'T (°C)': T_C_valid if 'T_C_valid' in locals() else T_C_m1[:len(T_K_valid)],
                        'T (K)': T_K_valid,
                        '[OH]': OH_valid,
                        'Kw': Kw_valid,
                        'ln(Kw)': ln_Kw,
                        '1000/T': x_m1
                    })
                    st.dataframe(df_kw, use_container_width=True)
            
            # Метод 2: Прямой фитинг
            st.markdown("---")
            st.header("📊 Метод 2: Прямой фитинг температурного профиля")
            
            # Применение исключения точек
            n_low_m2 = exclude_low_T_method2
            n_high_m2 = exclude_high_T_method2
            
            T_K_m2 = T_K[n_low_m2:len(T_K)-n_high_m2]
            OH_exp_m2 = OH_exp[n_low_m2:len(OH_exp)-n_high_m2]
            T_C_m2 = T_C[n_low_m2:len(T_C)-n_high_m2]
            
            # Функция для фитинга
            def model_OH_fit(T_K_fit, dH, dS):
                return analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, dH, dS)
            
            try:
                # Нелинейный фитинг
                popt, pcov = curve_fit(
                    model_OH_fit, 
                    T_K_m2, 
                    OH_exp_m2,
                    p0=[dH_method1, dS_method1],
                    bounds=([-500000, -500], [0, 500]),  # Физически осмысленные границы
                    maxfev=10000
                )
                
                dH_method2, dS_method2 = popt
                perr = np.sqrt(np.diag(pcov))
                
                # Расчет модельных значений
                OH_model_m2 = model_OH_fit(T_K_m2, dH_method2, dS_method2)
                
                # Статистика
                residuals = OH_exp_m2 - OH_model_m2
                SSE = np.sum(residuals**2)
                SST = np.sum((OH_exp_m2 - np.mean(OH_exp_m2))**2)
                R2_method2 = 1 - (SSE/SST) if SST > 0 else 0
                RMSE = np.sqrt(SSE / len(OH_exp_m2))
                
                # 95% доверительные интервалы
                dH_ci_m2 = 1.96 * perr[0]
                dS_ci_m2 = 1.96 * perr[1]
                
                # Отображение результатов методом 2
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color = "green" if R2_method2 > 0.95 else "orange" if R2_method2 > 0.9 else "red"
                    st.markdown(f"<h3 style='color:{color}'>{R2_method2:.4f}</h3>", unsafe_allow_html=True)
                    st.metric("Коэффициент R²", f"{R2_method2:.4f}")
                    st.metric("RMSE", f"{RMSE:.6f}")
                
                with col2:
                    st.metric("ΔH°", f"{dH_method2/1000:.2f} ± {dH_ci_m2/1000:.2f} кДж/моль",
                             delta=f"{dH_method2:.0f} ± {perr[0]:.0f} Дж/моль")
                    st.metric("Точек для анализа", len(T_K_m2))
                
                with col3:
                    st.metric("ΔS°", f"{dS_method2:.2f} ± {dS_ci_m2:.2f} Дж/(моль·К)",
                             delta=f"± {perr[1]:.2f}")
                    st.metric("SSE", f"{SSE:.6f}")
                
                if show_intermediate:
                    with st.expander("🔍 Промежуточные расчеты (Метод 2)"):
                        df_fit = pd.DataFrame({
                            'T (°C)': T_C_m2,
                            'T (K)': T_K_m2,
                            '[OH] эксп': OH_exp_m2,
                            '[OH] модель': OH_model_m2,
                            'Разность': residuals,
                            'Отн. ошибка (%)': 100 * np.abs(residuals / OH_exp_m2)
                        })
                        st.dataframe(df_fit, use_container_width=True)
                
            except Exception as e:
                st.error(f"Ошибка при фитинге: {e}")
                st.info("Используются параметры из метода 1")
                dH_method2, dS_method2 = dH_method1, dS_method1
                R2_method2 = 0
                SSE = np.nan
                RMSE = np.nan
                perr = [0, 0]
            
            # Сводная таблица
            st.markdown("---")
            st.header("📋 Сводная таблица результатов")
            
            summary_data = {
                'Параметр': [
                    'ΔH° (кДж/моль)', 
                    'ΔH 95% CI (кДж/моль)',
                    'ΔS° (Дж/(моль·К))',
                    'ΔS 95% CI (Дж/(моль·К))',
                    'R²/Метрика',
                    'Точек для анализа',
                    'Ошибка фитинга'
                ],
                'Метод 1': [
                    f"{dH_method1/1000:.2f}",
                    f"±{dH_ci/1000:.2f}",
                    f"{dS_method1:.2f}",
                    f"±{dS_ci:.2f}",
                    f"{r_value**2:.4f}",
                    f"{len(T_K_valid)}",
                    f"std_err={std_err:.4f}"
                ],
                'Метод 2': [
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
            
            # Стилизация таблицы
            def color_r2(val):
                if isinstance(val, str) and '=' in val:
                    num = float(val.split('=')[1])
                elif isinstance(val, str) and val.replace('.', '').isdigit():
                    num = float(val)
                else:
                    return ''
                
                if num > 0.95:
                    return 'background-color: #d4edda'  # зеленый
                elif num > 0.9:
                    return 'background-color: #fff3cd'  # желтый
                else:
                    return 'background-color: #f8d7da'  # красный
            
            st.dataframe(
                summary_df.style.applymap(color_r2, subset=['Метод 1', 'Метод 2']),
                use_container_width=True
            )
            
            # Экспорт результатов
            st.markdown("### 📤 Экспорт результатов")
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
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
                    },
                    'metadata': {
                        'calculation_date': datetime.now().isoformat(),
                        'version': '2.0'
                    }
                }
                st.markdown(get_json_download_link(export_data, "parameters.json"), unsafe_allow_html=True)
            
            with col_exp3:
                if st.button("💾 Сохранить в историю"):
                    st.session_state.calculation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'parameters': export_data,
                        'summary': summary_data
                    })
                    st.success("Расчет сохранен в историю!")
            
            # Визуализация
            st.markdown("---")
            st.header("📊 Визуализация результатов")
            
            # Создание фигуры
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    'Экспериментальные данные',
                    'Метод 1: ln(Kw) vs 1000/T',
                    'Метод 2: Фитинг профиля',
                    'Остатки (Метод 2)',
                    'Сравнение методов',
                    'Температурная зависимость Kw'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # График 1: Экспериментальные данные
            fig.add_trace(
                go.Scatter(
                    x=T_C, y=OH_exp,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='black',
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    name='Эксперимент',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Добавляем физические границы
            fig.add_hline(
                y=Acc_value, 
                line=dict(color='red', width=1, dash='dash'),
                annotation_text=f'[Acc] = {Acc_value}',
                row=1, col=1
            )
            
            fig.add_hline(
                y=0, 
                line=dict(color='blue', width=1, dash='dash'),
                annotation_text='[OH] = 0',
                row=1, col=1
            )
            
            fig.update_xaxes(title_text="Температура (°C)", row=1, col=1)
            fig.update_yaxes(title_text="[OH]", row=1, col=1)
            
            # График 2: Метод 1
            fig.add_trace(
                go.Scatter(
                    x=x_m1, y=ln_Kw,
                    mode='markers',
                    marker=dict(size=10, color='blue'),
                    name='Данные',
                    showlegend=True
                ),
                row=1, col=2
            )
            
            x_fit = np.linspace(min(x_m1), max(x_m1), 100)
            y_fit = slope * x_fit + intercept
            fig.add_trace(
                go.Scatter(
                    x=x_fit, y=y_fit,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'Линейная регрессия<br>R² = {r_value**2:.4f}',
                    showlegend=True
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="1000/T (K⁻¹)", row=1, col=2)
            fig.update_yaxes(title_text="ln(K<sub>w</sub>)", row=1, col=2)
            
            # График 3: Метод 2
            fig.add_trace(
                go.Scatter(
                    x=T_C_m2, y=OH_exp_m2,
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    name='Эксперимент',
                    showlegend=True
                ),
                row=1, col=3
            )
            
            T_fit = np.linspace(min(T_C), max(T_C), 200)
            T_K_fit = T_fit + 273.15
            OH_fit = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, dH_method2, dS_method2)
            
            fig.add_trace(
                go.Scatter(
                    x=T_fit, y=OH_fit,
                    mode='lines',
                    line=dict(color='orange', width=2),
                    name=f'Модель (Метод 2)<br>R² = {R2_method2:.4f}',
                    showlegend=True
                ),
                row=1, col=3
            )
            
            fig.update_xaxes(title_text="Температура (°C)", row=1, col=3)
            fig.update_yaxes(title_text="[OH]", row=1, col=3)
            
            # График 4: Остатки
            if 'residuals' in locals():
                # Цветовая шкала по величине остатков
                colors = np.abs(residuals)
                
                fig.add_trace(
                    go.Scatter(
                        x=T_C_m2, y=residuals,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=colors,
                            colorscale='RdBu',
                            showscale=True,
                            colorbar=dict(title="|Остаток|")
                        ),
                        name='Остатки',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                fig.add_hline(y=0, line=dict(color='black', width=1), row=2, col=1)
                fig.update_xaxes(title_text="Температура (°C)", row=2, col=1)
                fig.update_yaxes(title_text="[OH]<sub>эксп</sub> - [OH]<sub>мод</sub>", row=2, col=1)
            
            # График 5: Сравнение методов
            OH_fit_m1 = analytical_OH_numerical(T_K_fit, pH2O_value, Acc_value, dH_method1, dS_method1)
            
            fig.add_trace(
                go.Scatter(
                    x=T_fit, y=OH_fit_m1,
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dash'),
                    name=f'Метод 1: ΔH = {dH_method1/1000:.1f} кДж/моль',
                    showlegend=True
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=T_fit, y=OH_fit,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name=f'Метод 2: ΔH = {dH_method2/1000:.1f} кДж/моль',
                    showlegend=True
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=T_C, y=OH_exp,
                    mode='markers',
                    marker=dict(size=8, color='black', opacity=0.5),
                    name='Эксперимент',
                    showlegend=True
                ),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Температура (°C)", row=2, col=2)
            fig.update_yaxes(title_text="[OH]", row=2, col=2)
            
            # График 6: Температурная зависимость Kw
            Kw_m1 = np.exp(-dH_method1/(R * T_K_fit) + dS_method1/R)
            Kw_m2 = np.exp(-dH_method2/(R * T_K_fit) + dS_method2/R)
            
            fig.add_trace(
                go.Scatter(
                    x=T_fit, y=np.log(Kw_m1),
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dash'),
                    name='Метод 1',
                    showlegend=True
                ),
                row=2, col=3
            )
            
            fig.add_trace(
                go.Scatter(
                    x=T_fit, y=np.log(Kw_m2),
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Метод 2',
                    showlegend=True
                ),
                row=2, col=3
            )
            
            fig.update_xaxes(title_text="Температура (°C)", row=2, col=3)
            fig.update_yaxes(title_text="ln(K<sub>w</sub>)", row=2, col=3)
            
            # Общие настройки
            fig.update_layout(
                height=900,
                showlegend=True,
                font=dict(family='Arial', size=12),
                title_text=f"Термодинамический анализ | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 3D поверхности (только если выбрано)
            if calculate_3d:
                st.markdown("---")
                st.header("🌐 3D поверхности концентрации протонов")
                
                with st.spinner('Расчет 3D поверхностей...'):
                    progress_bar = st.progress(0)
                    
                    @st.cache_data(ttl=300)
                    def calculate_3d_surface_cached(method, dH, dS, Acc, pH2O_val, use_log, resolution=30):
                        """Кэшированная функция расчета 3D поверхности"""
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
                    
                    # Расчет поверхностей
                    progress_bar.progress(25)
                    T_range_m1, pH2O_range_m1, OH_grid_m1 = calculate_3d_surface_cached(
                        'method1', dH_method1, dS_method1, Acc_value, 
                        pH2O_value, use_log_pH2O, resolution=30
                    )
                    
                    progress_bar.progress(50)
                    T_range_m2, pH2O_range_m2, OH_grid_m2 = calculate_3d_surface_cached(
                        'method2', dH_method2, dS_method2, Acc_value,
                        pH2O_value, use_log_pH2O, resolution=30
                    )
                    
                    progress_bar.progress(75)
                    
                    # Создание 3D графиков
                    col_3d1, col_3d2 = st.columns(2)
                    
                    with col_3d1:
                        T_grid1, pH2O_grid1 = np.meshgrid(T_range_m1, pH2O_range_m1)
                        
                        fig_3d1 = go.Figure(data=[
                            go.Surface(
                                x=T_grid1,
                                y=np.log10(pH2O_grid1) if use_log_pH2O else pH2O_grid1,
                                z=OH_grid_m1,
                                colorscale='Viridis',
                                contours=dict(z=dict(show=True, color='black'))
                            )
                        ])
                        
                        # Добавляем экспериментальные точки
                        fig_3d1.add_trace(go.Scatter3d(
                            x=T_C,
                            y=np.log10(np.full_like(T_C, pH2O_value)) if use_log_pH2O else np.full_like(T_C, pH2O_value),
                            z=OH_exp,
                            mode='markers',
                            marker=dict(
                                size=5,
                                color='red',
                                symbol='circle'
                            ),
                            name='Эксперимент'
                        ))
                        
                        fig_3d1.update_layout(
                            title='Метод 1',
                            scene=dict(
                                xaxis_title='Температура (°C)',
                                yaxis_title='log₁₀(pH₂O)' if use_log_pH2O else 'pH₂O (атм)',
                                zaxis_title='[OH]'
                            ),
                            height=500
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
                                contours=dict(z=dict(show=True, color='black'))
                            )
                        ])
                        
                        fig_3d2.add_trace(go.Scatter3d(
                            x=T_C,
                            y=np.log10(np.full_like(T_C, pH2O_value)) if use_log_pH2O else np.full_like(T_C, pH2O_value),
                            z=OH_exp,
                            mode='markers',
                            marker=dict(
                                size=5,
                                color='red',
                                symbol='circle'
                            ),
                            name='Эксперимент'
                        ))
                        
                        fig_3d2.update_layout(
                            title='Метод 2',
                            scene=dict(
                                xaxis_title='Температура (°C)',
                                yaxis_title='log₁₀(pH₂O)' if use_log_pH2O else 'pH₂O (атм)',
                                zaxis_title='[OH]'
                            ),
                            height=500
                        )
                        
                        st.plotly_chart(fig_3d2, use_container_width=True)
                    
                    progress_bar.progress(100)
                    st.success("3D поверхности рассчитаны!")
            
            # Комментарии и рекомендации
            st.markdown("---")
            st.header("💡 Комментарии и рекомендации")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.subheader("Качество фитинга")
                
                recommendations = []
                
                if r_value**2 > 0.98 and R2_method2 > 0.98:
                    recommendations.append("✅ Отличное согласие обоих методов с данными")
                elif r_value**2 > 0.95 and R2_method2 > 0.95:
                    recommendations.append("✅ Хорошее согласие методов с данными")
                elif r_value**2 < 0.9 or R2_method2 < 0.9:
                    recommendations.append("⚠️ Рекомендуется исключить больше точек или проверить данные")
                
                if abs(dH_method2 - dH_method1) > 0.15 * abs(dH_method1):
                    recommendations.append(f"⚠️ Значительное расхождение в ΔH°: {abs((dH_method2-dH_method1)/dH_method1*100):.1f}%")
                elif abs(dH_method2 - dH_method1) > 0.05 * abs(dH_method1):
                    recommendations.append(f"⚠️ Умеренное расхождение в ΔH°: {abs((dH_method2-dH_method1)/dH_method1*100):.1f}%")
                else:
                    recommendations.append("✅ Хорошая сходимость методов по ΔH°")
            
            with col_rec2:
                st.subheader("Рекомендации")
                
                st.markdown(f"""
                **Для публикаций:**
                - Метод 1: ΔH° = {dH_method1/1000:.1f} ± {dH_ci/1000:.2f} кДж/моль
                - Метод 2: ΔH° = {dH_method2/1000:.1f} ± {dH_ci_m2/1000:.2f} кДж/моль
                
                **Для моделирования:**
                - Рекомендуется метод 2 (прямой фитинг)
                - ΔH° = {dH_method2/1000:.1f} ± {dH_ci_m2/1000:.2f} кДж/моль
                - ΔS° = {dS_method2:.1f} ± {dS_ci_m2:.1f} Дж/(моль·К)
                
                **Средние значения:**
                - ΔH° = {(dH_method1+dH_method2)/2000:.1f} кДж/моль
                - ΔS° = {(dS_method1+dS_method2)/2:.1f} Дж/(моль·К)
                """)
            
            for rec in recommendations:
                if rec.startswith("✅"):
                    st.success(rec)
                elif rec.startswith("⚠️"):
                    st.warning(rec)
                else:
                    st.info(rec)
            
            # Сохранение в историю
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
        st.error(f"Произошла ошибка при расчетах: {str(e)}")
        st.info("""
        **Возможные причины:**
        1. Некорректный формат данных
        2. Физически невозможные значения параметров
        3. Проблемы с численной сходимостью
        
        **Рекомендации:**
        - Проверьте формат входных данных
        - Убедитесь, что все значения [OH] < [Acc]
        - Попробуйте исключить крайние точки
        """)
        
        if show_intermediate:
            with st.expander("Техническая информация об ошибке"):
                import traceback
                st.code(traceback.format_exc())

# Показываем историю расчетов если есть
if len(st.session_state.calculation_history) > 0:
    with st.sidebar.expander("📜 История расчетов", expanded=False):
        for i, calc in enumerate(reversed(st.session_state.calculation_history[-5:])):
            st.markdown(f"**Расчет {i+1}**")
            st.markdown(f"Время: {calc['timestamp'][11:19]}")
            st.markdown(f"ΔH₁: {calc['results']['method1']['dH']/1000:.1f} кДж/моль")
            st.markdown(f"ΔH₂: {calc['results']['method2']['dH']/1000:.1f} кДж/моль")
            st.markdown("---")

# Информация при запуске
if not calculate_btn:
    st.markdown("""
    ## 📖 Инструкция
    
    1. **Загрузите данные** в текстовом поле или выберите файл (CSV, TXT, Excel)
    2. **Установите параметры системы**: pH₂O и концентрацию акцептора [Acc]
    3. **Настройте фитинг**: при необходимости исключите крайние точки
    4. **Нажмите "Рассчитать"** для получения термодинамических параметров
    
    ## 🎯 Особенности новой версии
    
    ✅ **Надежное численное решение** вместо аналитических формул  
    ✅ **Погрешности и доверительные интервалы** для всех параметров  
    ✅ **Загрузка файлов** различных форматов  
    ✅ **Валидация данных** с проверкой физической корректности  
    ✅ **Экспорт результатов** в CSV, JSON, PNG  
    ✅ **Кэширование расчетов** для быстрой работы  
    ✅ **3D визуализация** (опционально)  
    ✅ **История расчетов**  
    
    ## 📊 Формат данных
    
    Поддерживаются следующие форматы:
    ```
    Температура [OH]         # Разделитель: пробел
    20.5;0.15               # Разделитель: точка с запятой
    300\t0.08              # Разделитель: табуляция
    ```
    
    **Единицы измерения:**
    - Температура: °C
    - Концентрация [OH]: безразмерная (относительная)
    - pH₂O: атмосферы (атм)
    - [Acc]: безразмерная (0 < x < 6)
    
    ## 🔍 Примеры данных
    
    Тестовые данные уже загружены. Нажмите "Рассчитать" для демонстрации работы.
    """)
    
    with st.expander("📈 Пример графика результатов"):
        st.image("https://via.placeholder.com/800x400?text=Пример+результатов", 
                caption="Пример визуализации результатов")


