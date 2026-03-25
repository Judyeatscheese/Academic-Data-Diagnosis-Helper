import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="学术数据回归诊断与调优系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'target_var' not in st.session_state:
        st.session_state.target_var = None
    if 'feature_vars' not in st.session_state:
        st.session_state.feature_vars = []
    if 'extra_terms' not in st.session_state:
        st.session_state.extra_terms = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'previous_model' not in st.session_state:
        st.session_state.previous_model = None
    if 'initial_model' not in st.session_state:
        st.session_state.initial_model = None
    if 'current_gate' not in st.session_state:
        st.session_state.current_gate = 0
    if 'baseline_initialized' not in st.session_state:
        st.session_state.baseline_initialized = False
    if 'gates_completed' not in st.session_state:
        st.session_state.gates_completed = {1: False, 2: False, 3: False, 4: False}
    if 'gates_ignored' not in st.session_state:
        st.session_state.gates_ignored = {1: False, 2: False, 3: False, 4: False}
    if 'use_robust_se' not in st.session_state:
        st.session_state.use_robust_se = False
    if 'last_action' not in st.session_state:
        st.session_state.last_action = None
    if 'model_history' not in st.session_state:
        st.session_state.model_history = []
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False

def render_comparison_block():
    """展示修复前后对比并提供进入下一步按钮"""
    if st.session_state.previous_model and st.session_state.current_model:
        display_comparison_board(
            st.session_state.previous_model,
            st.session_state.current_model,
            st.session_state.last_action
        )
        
        if st.button("👉 进入下一步", type="primary", use_container_width=True):
            st.session_state.show_comparison = False
            st.session_state.current_gate += 1
            st.rerun()

def handle_step_transition():
    """处理步骤过渡"""
    st.session_state.show_comparison = False
    st.rerun()

def highlight_significance(df):
    """高亮显著性，p > 0.05 标红"""
    def color_pval(val):
        if isinstance(val, (int, float)):
            if val > 0.05:
                return 'color: red; font-weight: bold'
        elif isinstance(val, str):
            # 处理带有箭头的 p 值字符串，如 "0.042 ↓"
            try:
                p_value = float(val.split()[0])
                if p_value > 0.05:
                    return 'color: red; font-weight: bold'
            except:
                pass
        return ''
    
    styled_df = df.style.applymap(color_pval, subset=['p值'])
    return styled_df

def compare_pvalues(prev_model, curr_model):
    """比较p值变化，返回带箭头的p值"""
    if not prev_model or not curr_model:
        return {}
    
    prev_pvals = prev_model.pvalues
    curr_pvals = curr_model.pvalues
    
    pval_changes = {}
    for var in curr_pvals.index:
        if var in prev_pvals:
            prev_p = prev_pvals[var]
            curr_p = curr_pvals[var]
            
            if curr_p < prev_p:
                pval_changes[var] = f"{curr_p:.4f} ↓"
            elif curr_p > prev_p:
                pval_changes[var] = f"{curr_p:.4f} ↑"
            else:
                pval_changes[var] = f"{curr_p:.4f} -"
    
    return pval_changes

def go_back_step():
    """回到上一步"""
    if st.session_state.current_gate > 1:
        st.session_state.current_gate -= 1
        st.rerun()

def go_to_next_step(next_step):
    """统一的关卡跳转函数"""
    st.session_state.current_gate = next_step
    st.session_state.show_comparison = False
    st.rerun()

def classify_variables(df, feature_vars):
    """变量分类：binary_vars, continuous_vars, discrete_vars, control_vars"""
    binary_vars = []
    continuous_vars = []
    discrete_vars = []
    control_vars = []
    
    for var in feature_vars:
        series = df[var].dropna()
        nunique = series.nunique()
        
        # 判断是否为控制变量
        if 'Brand' in var:
            control_vars.append(var)
        
        # 判断是否为哑变量 (0/1)
        if nunique == 2:
            unique_vals = sorted(series.unique())
            if set(unique_vals) == {0, 1}:
                binary_vars.append(var)
            else:
                discrete_vars.append(var)
        elif nunique <= 5:
            discrete_vars.append(var)
        elif nunique > 10:
            continuous_vars.append(var)
        else:
            discrete_vars.append(var)
    
    return {
        'binary_vars': binary_vars,
        'continuous_vars': continuous_vars,
        'discrete_vars': discrete_vars,
        'control_vars': control_vars
    }

def generate_quadratic_terms(df, target_var, feature_vars, var_classes):
    """生成平方项 - 仅对 continuous_vars（非控制变量）"""
    significant_quadratic = []
    
    # 只对连续且非控制变量生成平方项
    eligible_vars = [var for var in var_classes['continuous_vars'] 
                     if var not in var_classes['control_vars']]
    
    for var in eligible_vars:
        try:
            X = df[feature_vars].copy()
            X[f"{var}_sq"] = df[var] ** 2
            X = sm.add_constant(X)
            y = df[target_var]
            
            model = sm.OLS(y, X).fit()
            sq_pval = model.pvalues.get(f"{var}_sq")
            
            if sq_pval is not None and sq_pval < 0.05:
                significant_quadratic.append({
                    'var': var,
                    'p_value': sq_pval,
                    'coefficient': model.params.get(f"{var}_sq")
                })
        except Exception as e:
            continue
    
    return significant_quadratic

def generate_interaction_terms_filtered(df, target_var, feature_vars, var_classes, max_interactions=30):
    """生成筛选后的交互项"""
    recommended = []
    cautious = []
    control_interactions = []
    
    n_features = len(feature_vars)
    interaction_count = 0
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if interaction_count >= max_interactions:
                break
            
            var1 = feature_vars[i]
            var2 = feature_vars[j]
            
            # 判断变量类型
            var1_is_binary = var1 in var_classes['binary_vars']
            var1_is_continuous = var1 in var_classes['continuous_vars']
            var2_is_binary = var2 in var_classes['binary_vars']
            var2_is_continuous = var2 in var_classes['continuous_vars']
            var1_is_control = var1 in var_classes['control_vars']
            var2_is_control = var2 in var_classes['control_vars']
            
            # 排除 binary × binary
            if var1_is_binary and var2_is_binary:
                continue
            
            try:
                X = df[feature_vars].copy()
                X[f"{var1}_x_{var2}"] = df[var1] * df[var2]
                X = sm.add_constant(X)
                y = df[target_var]
                
                model = sm.OLS(y, X).fit()
                int_pval = model.pvalues.get(f"{var1}_x_{var2}")
                
                if int_pval is not None and int_pval < 0.05:
                    term = {
                        'vars': [var1, var2],
                        'p_value': int_pval,
                        'coefficient': model.params.get(f"{var1}_x_{var2}")
                    }
                    
                    # 分类
                    if var1_is_control or var2_is_control:
                        control_interactions.append(term)
                    elif var1_is_continuous and var2_is_continuous:
                        recommended.append(term)
                    elif (var1_is_continuous and var2_is_binary) or (var2_is_continuous and var1_is_binary):
                        cautious.append(term)
                        
            except Exception as e:
                continue
            
            interaction_count += 1
    
    return {
        'recommended': recommended,
        'cautious': cautious,
        'control': control_interactions
    }

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return None, "不支持的文件格式，请上传 .csv 或 .xlsx 文件"
        
        if df.empty:
            return None, "上传的文件为空"
        
        return df, None
    except Exception as e:
        return None, f"文件读取失败: {str(e)}"

def fit_ols_model(df, target_var, feature_vars, extra_terms=None, use_robust_se=False):
    try:
        X = df[feature_vars].copy()
        
        if extra_terms:
            for term in extra_terms:
                if term['type'] == 'quadratic':
                    var_name = term['var']
                    X[f"{var_name}_sq"] = df[var_name] ** 2
                elif term['type'] == 'interaction':
                    var1, var2 = term['vars']
                    X[f"{var1}_x_{var2}"] = df[var1] * df[var2]
        
        X = sm.add_constant(X)
        y = df[target_var]
        
        model = sm.OLS(y, X).fit()
        
        if use_robust_se:
            model = model.get_robustcov_results(cov_type='HC3')
        
        return model, None
    except Exception as e:
        return None, f"模型拟合失败: {str(e)}"

def extract_model_summary(model):
    summary = model.summary2()
    results_df = summary.tables[1]
    
    if isinstance(results_df, pd.DataFrame):
        results_df = results_df.reset_index()
        results_df.columns = ['变量', '系数', '标准误', 't值', 'p值', '[0.025', '0.975]']
    else:
        results_df = pd.DataFrame()
    
    model_info = {
        'R-squared': model.rsquared,
        'Adj. R-squared': model.rsquared_adj,
        'F-statistic': model.fvalue,
        'Prob (F-statistic)': model.f_pvalue,
        'AIC': model.aic,
        'BIC': model.bic,
        '样本量': int(model.nobs)
    }
    
    return model_info, results_df

def display_model_result(model, title="模型结果", previous_model=None):
    if model is None:
        st.warning("暂无模型结果")
        return
    
    model_info, results_df = extract_model_summary(model)
    
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R-squared", f"{model_info['R-squared']:.4f}")
    with col2:
        st.metric("Adj. R-squared", f"{model_info['Adj. R-squared']:.4f}")
    with col3:
        st.metric("F-statistic", f"{model_info['F-statistic']:.4f}")
    with col4:
        st.metric("样本量", model_info['样本量'])
    
    if not results_df.empty:
        st.write("**回归系数表：**")
        
        display_df = results_df.copy()
        display_df['p值'] = pd.to_numeric(display_df['p值'], errors='coerce')
        
        # 如果提供了 previous_model，在 p 值列中显示变化箭头
        if previous_model:
            prev_pvals = previous_model.pvalues
            for idx, row in display_df.iterrows():
                var = row['变量']
                if var in prev_pvals:
                    prev_p = prev_pvals[var]
                    curr_p = row['p值']
                    if curr_p < prev_p:
                        display_df.at[idx, 'p值'] = f"{curr_p:.4f} ↓"
                    elif curr_p > prev_p:
                        display_df.at[idx, 'p值'] = f"{curr_p:.4f} ↑"
        
        styled_df = highlight_significance(display_df)
        st.dataframe(styled_df, use_container_width=True)
        
        insignificant_vars = display_df[display_df['p值'].apply(lambda x: isinstance(x, (int, float)) and x > 0.05 or (isinstance(x, str) and float(x.split()[0]) > 0.05))]['变量'].tolist()
        if insignificant_vars:
            st.warning(f"⚠️ 以下变量在统计上不显著 (p > 0.05): {', '.join(insignificant_vars)}")

def is_suitable_for_iqr(series):
    """判断变量是否适合使用 IQR 方法进行极值检测"""
    if not pd.api.types.is_numeric_dtype(series):
        return False, "非数值型变量"
    
    unique_values = series.nunique()
    total_values = len(series)
    
    if unique_values <= 2:
        return False, "哑变量/二分类变量"
    
    if unique_values < 10 and (unique_values / total_values) < 0.05:
        return False, f"低取值离散变量 (仅 {unique_values} 个唯一值)"
    
    skewness = stats.skew(series.dropna())
    is_long_tailed = abs(skewness) > 1
    
    return True, "连续变量" + (" (疑似长尾分布)" if is_long_tailed else "")

def detect_outliers_enhanced(df, columns):
    """增强版极值检测，包含变量适用性判断"""
    outlier_info = {}
    unsuitable_info = {}
    long_tailed_warning = []
    
    for col in columns:
        series = df[col]
        suitable, reason = is_suitable_for_iqr(series)
        
        if not suitable:
            unsuitable_info[col] = reason
            continue
        
        if "疑似长尾分布" in reason:
            long_tailed_warning.append(col)
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / len(df) * 100
        
        outlier_info[col] = {
            'count': outlier_count,
            'ratio': outlier_ratio,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'has_outliers': outlier_count > 0,
            'is_long_tailed': col in long_tailed_warning
        }
    
    return outlier_info, unsuitable_info, long_tailed_warning

def detect_outliers_iqr(df, columns):
    """原 IQR 极值检测函数（保留向后兼容）"""
    outlier_info = {}
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_ratio = outlier_count / len(df) * 100
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': outlier_count,
                    'ratio': outlier_ratio,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
    
    return outlier_info

def apply_winsorize(df, columns, lower_percentile=1, upper_percentile=99):
    df_winsorized = df.copy()
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lower_limit = np.percentile(df[col].dropna(), lower_percentile)
            upper_limit = np.percentile(df[col].dropna(), upper_percentile)
            
            df_winsorized[col] = df_winsorized[col].clip(lower=lower_limit, upper=upper_limit)
    
    return df_winsorized

def calculate_vif(df, feature_vars):
    try:
        X = df[feature_vars].copy()
        X = sm.add_constant(X)
        
        vif_data = []
        for i, var in enumerate(feature_vars):
            vif = variance_inflation_factor(X.values, i + 1)
            vif_data.append({'变量': var, 'VIF': vif})
        
        vif_df = pd.DataFrame(vif_data)
        return vif_df
    except Exception as e:
        st.error(f"VIF 计算失败: {str(e)}")
        return pd.DataFrame()

def remove_highest_vif_feature(vif_df, feature_vars):
    if vif_df.empty:
        return feature_vars
    
    highest_vif_row = vif_df.loc[vif_df['VIF'].idxmax()]
    var_to_remove = highest_vif_row['变量']
    
    updated_features = [var for var in feature_vars if var != var_to_remove]
    
    return updated_features, var_to_remove

def test_quadratic_terms(df, target_var, feature_vars, current_model):
    """保留原函数（向后兼容）"""
    significant_quadratic = []
    
    for var in feature_vars:
        try:
            X = df[feature_vars].copy()
            X[f"{var}_sq"] = df[var] ** 2
            X = sm.add_constant(X)
            y = df[target_var]
            
            model = sm.OLS(y, X).fit()
            sq_pval = model.pvalues.get(f"{var}_sq")
            
            if sq_pval is not None and sq_pval < 0.05:
                significant_quadratic.append({
                    'var': var,
                    'p_value': sq_pval,
                    'coefficient': model.params.get(f"{var}_sq")
                })
        except Exception as e:
            continue
    
    return significant_quadratic

def test_interaction_terms(df, target_var, feature_vars, current_model, max_interactions=20):
    """保留原函数（向后兼容）"""
    significant_interactions = []
    
    n_features = len(feature_vars)
    total_interactions = n_features * (n_features - 1) // 2
    
    if total_interactions > max_interactions:
        st.info(f"⚠️ 变量组合过多 ({total_interactions} 个)，仅测试前 {max_interactions} 个组合")
    
    interaction_count = 0
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if interaction_count >= max_interactions:
                break
            
            var1 = feature_vars[i]
            var2 = feature_vars[j]
            
            try:
                X = df[feature_vars].copy()
                X[f"{var1}_x_{var2}"] = df[var1] * df[var2]
                X = sm.add_constant(X)
                y = df[target_var]
                
                model = sm.OLS(y, X).fit()
                int_pval = model.pvalues.get(f"{var1}_x_{var2}")
                
                if int_pval is not None and int_pval < 0.05:
                    significant_interactions.append({
                        'vars': [var1, var2],
                        'p_value': int_pval,
                        'coefficient': model.params.get(f"{var1}_x_{var2}")
                    })
            except Exception as e:
                continue
            
            interaction_count += 1
    
    return significant_interactions

def refit_with_selected_terms(df, target_var, feature_vars, selected_terms, use_robust_se=False):
    extra_terms = []
    
    for term in selected_terms:
        if term.startswith('quadratic_'):
            var = term.replace('quadratic_', '')
            extra_terms.append({'type': 'quadratic', 'var': var})
        elif term.startswith('interaction_'):
            vars_str = term.replace('interaction_', '')
            var1, var2 = vars_str.split('_x_')
            extra_terms.append({'type': 'interaction', 'vars': [var1, var2]})
    
    model, error = fit_ols_model(df, target_var, feature_vars, extra_terms, use_robust_se)
    
    return model, extra_terms, error

def test_heteroskedasticity(model):
    try:
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, sm.add_constant(fitted_values))
        white_stat, white_pval, _, _ = het_white(residuals, sm.add_constant(fitted_values))
        
        result = {
            'breusch_pagan': {
                'statistic': bp_stat,
                'p_value': bp_pval,
                'has_heteroskedasticity': bp_pval < 0.05
            },
            'white': {
                'statistic': white_stat,
                'p_value': white_pval,
                'has_heteroskedasticity': white_pval < 0.05
            }
        }
        
        return result
    except Exception as e:
        st.error(f"异方差检验失败: {str(e)}")
        return None

def refit_with_robust_se(df, target_var, feature_vars, extra_terms=None):
    model, error = fit_ols_model(df, target_var, feature_vars, extra_terms, use_robust_se=True)
    return model, error

def display_comparison_board(previous_model, current_model, action_description):
    st.subheader("📊 修复前后对比")
    st.info(f"操作: {action_description}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔙 修复前")
        display_model_result(previous_model, "")
    
    with col2:
        st.markdown("### 🔜 修复后")
        # 传递 previous_model 以便在 p 值列中显示变化箭头
        display_model_result(current_model, "", previous_model)
    
    prev_info, _ = extract_model_summary(previous_model)
    curr_info, _ = extract_model_summary(current_model)
    
    st.markdown("### 📈 关键指标变化")
    
    comparison_data = {
        '指标': ['R-squared', 'Adj. R-squared', '样本量'],
        '修复前': [
            f"{prev_info['R-squared']:.4f}",
            f"{prev_info['Adj. R-squared']:.4f}",
            prev_info['样本量']
        ],
        '修复后': [
            f"{curr_info['R-squared']:.4f}",
            f"{curr_info['Adj. R-squared']:.4f}",
            curr_info['样本量']
        ],
        '变化': [
            f"{(curr_info['R-squared'] - prev_info['R-squared']):+.4f}",
            f"{(curr_info['Adj. R-squared'] - prev_info['Adj. R-squared']):+.4f}",
            f"{curr_info['样本量'] - prev_info['样本量']:+d}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.astype(str), use_container_width=True)

def render_progress_bar():
    gates = [
        "Step 0: 数据加载",
        "Step 1: 极值诊断",
        "Step 2: 共线性诊断",
        "Step 3: 非线性/交互项诊断",
        "Step 4: 异方差诊断",
        "Step 5: 调优完成"
    ]
    
    current_gate = st.session_state.current_gate
    progress = current_gate / 5
    
    st.progress(progress)
    
    cols = st.columns(6)
    for i, (col, gate) in enumerate(zip(cols, gates)):
        if i < current_gate:
            col.markdown(f"✅ {gate}")
        elif i == current_gate:
            col.markdown(f"🔵 {gate}")
        else:
            col.markdown(f"⚪ {gate}")

def render_header():
    st.title("📊 学术数据回归诊断与调优系统")
    st.markdown("""
    **面向论文实证分析的逐级诊断工具**
    
    本系统通过闯关式诊断机制，帮助您逐步识别并修复数据质量、变量关系、模型设定、误差结构中的问题。
    系统遵循"高优阻断、逐级修复、用户决策优先"的原则。
    """)

def render_data_upload_section():
    st.subheader("📁 数据上传与变量选择")
    
    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=['csv', 'xlsx'],
        help="支持 .csv 和 .xlsx 格式"
    )
    
    if uploaded_file is not None:
        df, error = load_data(uploaded_file)
        
        if error:
            st.error(error)
            return
        
        st.session_state.raw_df = df
        st.session_state.current_df = df.copy()
        
        st.success(f"✅ 数据加载成功！共 {len(df)} 行，{len(df.columns)} 列")
        
        with st.expander("查看数据预览", expanded=True):
            st.dataframe(df.head().astype(str), use_container_width=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ 数据中没有数值型变量，无法进行回归分析")
            return
        
        st.info(f"📋 检测到 {len(numeric_cols)} 个数值型变量: {', '.join(numeric_cols)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 根据 session_state 初始化因变量选择
            target_var = st.selectbox(
                "选择因变量 Y",
                options=numeric_cols,
                index=numeric_cols.index(st.session_state.get('target_var', numeric_cols[0])) if 'target_var' in st.session_state and st.session_state.target_var in numeric_cols else 0,
                help="选择您要解释的目标变量"
            )
        
        with col2:
            # 根据 session_state 初始化自变量选择
            available_features = [col for col in numeric_cols if col != target_var]
            default_features = []
            if 'feature_vars' in st.session_state:
                default_features = [var for var in st.session_state.feature_vars if var in available_features]
            else:
                default_features = available_features[:min(5, len(available_features))]
            
            feature_vars = st.multiselect(
                "选择自变量 X",
                options=available_features,
                default=default_features,
                help="选择用于解释因变量的自变量，可多选"
            )
        
        if st.button("🚀 建立基线模型", type="primary", use_container_width=True):
            if not feature_vars:
                st.error("❌ 请至少选择一个自变量")
                return
            
            st.session_state.target_var = target_var
            st.session_state.feature_vars = feature_vars
            
            model, error = fit_ols_model(
                st.session_state.current_df,
                target_var,
                feature_vars
            )
            
            if error:
                st.error(f"❌ {error}")
                return
            
            st.session_state.current_model = model
            st.session_state.initial_model = model  # 保存初始模型
            st.session_state.baseline_initialized = True
            st.session_state.current_gate = 1
            st.session_state.last_action = "建立基线模型"
            st.session_state.model_history.append({
                'step': 'baseline',
                'model': model,
                'description': '初始基线模型'
            })
            
            st.success("✅ 基线模型建立成功！")
            st.rerun()

def render_baseline_model_section():
    st.subheader("📈 当前基线模型结果")
    
    if st.session_state.current_model:
        display_model_result(st.session_state.current_model, "基线模型")

def render_outlier_diagnosis():
    st.subheader("🔍 Step 1: 极值诊断")
    
    # 添加返回 Step0 按钮
    if st.button("← 返回上一步（变量选择）", use_container_width=True):
        # 重置相关状态
        st.session_state.current_gate = 0
        st.session_state.baseline_initialized = False
        # 重置模型相关状态
        if 'current_model' in st.session_state:
            del st.session_state['current_model']
        if 'initial_model' in st.session_state:
            del st.session_state['initial_model']
        if 'previous_model' in st.session_state:
            del st.session_state['previous_model']
        if 'extra_terms' in st.session_state:
            del st.session_state['extra_terms']
        if 'use_robust_se' in st.session_state:
            del st.session_state['use_robust_se']
        if 'last_action' in st.session_state:
            del st.session_state['last_action']
        if 'model_history' in st.session_state:
            del st.session_state['model_history']
        if 'show_comparison' in st.session_state:
            del st.session_state['show_comparison']
        # 重置门控状态
        if 'gates_completed' in st.session_state:
            del st.session_state['gates_completed']
        if 'gates_ignored' in st.session_state:
            del st.session_state['gates_ignored']
        st.rerun()
    
    st.markdown("""
    **为什么这一步重要？**
    极端值可能对回归结果产生不成比例的影响，导致系数估计偏差和标准误增大。
    """)
    
    all_vars = [st.session_state.target_var] + st.session_state.feature_vars
    outlier_info, unsuitable_info, long_tailed_vars = detect_outliers_enhanced(
        st.session_state.current_df, all_vars
    )
    
    suitable_vars_with_outliers = [var for var, info in outlier_info.items() if info['has_outliers']]
    
    # 检查因变量是否有极值
    target_var = st.session_state.target_var
    target_has_outliers = target_var in suitable_vars_with_outliers
    
    if not suitable_vars_with_outliers:
        if outlier_info:
            st.success("✅ 在适合 IQR 检测的变量中未检测到明显极值")
            # 特别提示因变量情况
            if target_var in outlier_info:
                st.info("💡 因变量未检测到极端值")
        else:
            st.success("✅ 没有适合 IQR 检测的变量，或未检测到极值")
        
        if st.button("➡️ 跳过", use_container_width=True):
            st.session_state.gates_ignored[1] = True
            go_to_next_step(2)
        return
    
    st.warning(f"⚠️ 检测到 {len(suitable_vars_with_outliers)} 个变量存在极值")
    
    if long_tailed_vars:
        st.info("💡 若变量为长尾分布，极值可能是分布特征，建议谨慎处理")
    
    outlier_rows = []
    for var in suitable_vars_with_outliers:
        info = outlier_info[var]
        # 标记变量类型
        var_type = "因变量" if var == target_var else "自变量"
        row = {
            '变量': var,
            '变量类型': var_type,
            '极值比例': f"{info['ratio']:.2f}%",
            '是否长尾': '是' if info['is_long_tailed'] else '否'
        }
        outlier_rows.append(row)
    
    outlier_df = pd.DataFrame(outlier_rows)
    st.dataframe(outlier_df.astype(str), use_container_width=True)
    
    st.markdown("**请选择要执行 1%~99% 缩尾处理的变量：**")
    selected_vars = []
    
    for var in suitable_vars_with_outliers:
        is_long_tailed = outlier_info[var]['is_long_tailed']
        default_checked = not is_long_tailed
        if st.checkbox(
            f"{var} (极值: {outlier_info[var]['count']} 个, {outlier_info[var]['ratio']:.2f}%)",
            key=f"winsorize_{var}",
            value=default_checked
        ):
            selected_vars.append(var)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶ 对选中变量做缩尾", type="primary", use_container_width=True):
            if not selected_vars:
                st.warning("⚠️ 请至少选择一个变量")
                return
            
            st.session_state.previous_model = st.session_state.current_model
            
            st.session_state.current_df = apply_winsorize(
                st.session_state.current_df,
                selected_vars
            )
            
            model, error = fit_ols_model(
                st.session_state.current_df,
                st.session_state.target_var,
                st.session_state.feature_vars,
                st.session_state.extra_terms,
                st.session_state.use_robust_se
            )
            
            if error:
                st.error(f"❌ {error}")
                return
            
            st.session_state.current_model = model
            st.session_state.gates_completed[1] = True
            st.session_state.last_action = f"对 {len(selected_vars)} 个变量进行 1%~99% 缩尾处理: {', '.join(selected_vars)}"
            st.session_state.model_history.append({
                'step': 'outlier_fix',
                'model': model,
                'description': st.session_state.last_action
            })
            
            st.success(f"✅ 已对 {len(selected_vars)} 个变量完成缩尾处理！")
            st.info("💡 缩尾（winsorize）仅对极端值进行截断，不删除样本，因此样本量不变")
            st.session_state.show_comparison = True
            st.rerun()
    
    with col2:
        if st.button("⏭️ 跳过", use_container_width=True):
            st.session_state.gates_ignored[1] = True
            go_to_next_step(2)

def render_collinearity_diagnosis():
    st.subheader("🔍 Step 2: 共线性诊断")
    
    # 添加回到上一步按钮
    if st.button("⬅ 回到上一步", use_container_width=True):
        go_back_step()
    
    st.markdown("""
    **为什么这一步重要？**
    
    多重共线性会导致回归系数估计不稳定、标准误增大，使得变量显著性检验不可靠。
    在学术研究中，VIF > 10 通常被视为存在严重共线性的标志。
    """)
    
    vif_df = calculate_vif(
        st.session_state.current_df,
        st.session_state.feature_vars
    )
    
    if vif_df.empty:
        st.error("❌ VIF 计算失败")
        return
    
    max_vif = vif_df['VIF'].max()
    
    if max_vif < 5:
        st.success("✅ 未检测到明显共线性")
        st.markdown("**你的选择：**")
        if st.button("➡️ 忽略共线性，继续下一步", use_container_width=True):
            st.session_state.gates_ignored[2] = True
            go_to_next_step(3)
        return
    elif 5 <= max_vif < 10:
        st.info(f"💡 检测到中等共线性 (max VIF = {max_vif:.2f})")
    else:  # max_vif >= 10
        st.warning(f"⚠️ 检测到严重共线性 (max VIF = {max_vif:.2f})")
    
    high_vif_vars = vif_df[vif_df['VIF'] > 5]
    
    def highlight_vif(val):
        if val > 10:
            return 'background-color: #f8d7da; font-weight: bold'
        return ''
    
    styled_vif = vif_df.style.applymap(highlight_vif, subset=['VIF'])
    st.dataframe(vif_df.astype(str), use_container_width=True)
    
    highest_vif_var = vif_df.loc[vif_df['VIF'].idxmax(), '变量']
    st.info(f"📌 VIF 最高的变量: {highest_vif_var} (VIF = {vif_df['VIF'].max():.2f})")
    
    st.markdown("**你现在有哪些选择：**")
    st.markdown("1. **剔除 VIF 最高变量**：自动移除共线性最严重的变量")
    st.markdown("2. **忽略并继续**：保持当前变量组合，进入下一阶段")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶ 自动剔除 VIF 最高的变量", type="primary", use_container_width=True):
            st.session_state.previous_model = st.session_state.current_model
            
            updated_features, removed_var = remove_highest_vif_feature(
                vif_df,
                st.session_state.feature_vars
            )
            
            st.session_state.feature_vars = updated_features
            
            model, error = fit_ols_model(
                st.session_state.current_df,
                st.session_state.target_var,
                st.session_state.feature_vars,
                st.session_state.extra_terms,
                st.session_state.use_robust_se
            )
            
            if error:
                st.error(f"❌ {error}")
                return
            
            st.session_state.current_model = model
            st.session_state.gates_completed[2] = True
            st.session_state.last_action = f"剔除高共线性变量: {removed_var}"
            st.session_state.model_history.append({
                'step': 'collinearity_fix',
                'model': model,
                'description': f'剔除变量: {removed_var}'
            })
            
            st.success(f"✅ 已剔除变量: {removed_var}")
            st.session_state.show_comparison = True
            st.rerun()
    
    with col2:
        if st.button("⏭️ 忽略共线性，继续下一步", use_container_width=True):
            st.session_state.gates_ignored[2] = True
            go_to_next_step(3)

def render_nonlinearity_diagnosis():
    st.subheader("🔍 Step 3: 非线性与交互项诊断")
    
    # 添加回到上一步按钮
    if st.button("⬅ 回到上一步", use_container_width=True):
        go_back_step()
    
    st.markdown("""
    **为什么这一步重要？**
    现实世界中变量关系往往不是线性的。遗漏非线性关系或交互效应可能导致模型设定偏误。
    """)
    
    # 变量分类
    var_classes = classify_variables(
        st.session_state.current_df,
        st.session_state.feature_vars
    )
    
    st.info("⏳ 正在检测非线性关系和交互效应...")
    
    # 生成平方项
    quadratic_terms = generate_quadratic_terms(
        st.session_state.current_df,
        st.session_state.target_var,
        st.session_state.feature_vars,
        var_classes
    )
    
    # 生成交互项
    interaction_terms = generate_interaction_terms_filtered(
        st.session_state.current_df,
        st.session_state.target_var,
        st.session_state.feature_vars,
        var_classes
    )
    
    has_recommendations = (len(quadratic_terms) > 0 or 
                           len(interaction_terms['recommended']) > 0 or 
                           len(interaction_terms['cautious']) > 0)
    
    if not has_recommendations:
        st.success("✅ 未检测到显著的非线性关系或交互效应")
        if st.button("➡️ 跳过", use_container_width=True):
            st.session_state.gates_ignored[3] = True
            go_to_next_step(4)
        return
    
    st.info("💡 显著性不代表有研究意义，请结合理论选择")
    
    # 显示各项
    st.markdown("### A. 非线性项（平方项）")
    if quadratic_terms:
        quadratic_df = pd.DataFrame([
            {
                '变量': term['var'],
                'p值': f"{term['p_value']:.4f}",
                '系数': f"{term['coefficient']:.4f}"
            }
            for term in quadratic_terms
        ])
        st.dataframe(quadratic_df.astype(str), use_container_width=True)
    else:
        st.info("本数据未检测到显著非线性关系（平方项均不显著）")
    
    if interaction_terms['recommended'] or interaction_terms['cautious']:
        st.markdown("### B. 交互项")
        
        if interaction_terms['recommended']:
            st.markdown("#### 连续 × 连续")
            st.info("💡 建议先对变量中心化（减均值）以减少共线性")
            rec_df = pd.DataFrame([
                {
                    '变量组合': f"{term['vars'][0]} × {term['vars'][1]}",
                    'p值': f"{term['p_value']:.4f}",
                    '系数': f"{term['coefficient']:.4f}"
                }
                for term in interaction_terms['recommended']
            ])
            st.dataframe(rec_df.astype(str), use_container_width=True)
        
        if interaction_terms['cautious']:
            st.markdown("#### 连续 × 哑变量")
            cautious_df = pd.DataFrame([
                {
                    '变量组合': f"{term['vars'][0]} × {term['vars'][1]}",
                    'p值': f"{term['p_value']:.4f}",
                    '系数': f"{term['coefficient']:.4f}"
                }
                for term in interaction_terms['cautious']
            ])
            st.dataframe(cautious_df.astype(str), use_container_width=True)
    
    selected_terms = []
    
    if quadratic_terms:
        st.markdown("**选择要引入的非线性项（默认不选）：**")
        for term in quadratic_terms:
            var_name = term['var']
            checkbox_key = f"quad_{var_name}"
            if st.checkbox(
                f"{var_name}² (p = {term['p_value']:.4f})",
                key=checkbox_key,
                value=False
            ):
                selected_terms.append(f"quadratic_{var_name}")
    
    if interaction_terms['recommended'] or interaction_terms['cautious']:
        st.markdown("**选择要引入的交互项（默认不选）：**")
        
        if interaction_terms['recommended']:
            for term in interaction_terms['recommended']:
                vars_str = f"{term['vars'][0]}_x_{term['vars'][1]}"
                checkbox_key = f"int_rec_{vars_str}"
                if st.checkbox(
                    f"{term['vars'][0]} × {term['vars'][1]} (p = {term['p_value']:.4f})",
                    key=checkbox_key,
                    value=False
                ):
                    selected_terms.append(f"interaction_{vars_str}")
        
        if interaction_terms['cautious']:
            for term in interaction_terms['cautious']:
                vars_str = f"{term['vars'][0]}_x_{term['vars'][1]}"
                checkbox_key = f"int_caut_{vars_str}"
                if st.checkbox(
                    f"{term['vars'][0]} × {term['vars'][1]} (p = {term['p_value']:.4f})",
                    key=checkbox_key,
                    value=False
                ):
                    selected_terms.append(f"interaction_{vars_str}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶ 重构模型", type="primary", use_container_width=True):
            if not selected_terms:
                st.warning("⚠️ 请至少选择一项")
                return
            
            st.session_state.previous_model = st.session_state.current_model
            
            model, extra_terms, error = refit_with_selected_terms(
                st.session_state.current_df,
                st.session_state.target_var,
                st.session_state.feature_vars,
                selected_terms,
                st.session_state.use_robust_se
            )
            
            if error:
                st.error(f"❌ {error}")
                return
            
            st.session_state.current_model = model
            st.session_state.extra_terms = extra_terms
            st.session_state.gates_completed[3] = True
            st.session_state.last_action = f"引入 {len(selected_terms)} 个新项"
            st.session_state.model_history.append({
                'step': 'nonlinearity_fix',
                'model': model,
                'description': f'引入 {len(selected_terms)} 个新项'
            })
            
            st.success(f"✅ 已引入 {len(selected_terms)} 个新项")
            st.session_state.show_comparison = True
            st.rerun()
    
    with col2:
        if st.button("⏭️ 跳过", use_container_width=True):
            st.session_state.gates_ignored[3] = True
            go_to_next_step(4)

def render_heteroskedasticity_diagnosis():
    st.subheader("🔍 Step 4: 异方差诊断")
    
    # 添加回到上一步按钮
    if st.button("⬅ 回到上一步", use_container_width=True):
        go_back_step()
    
    st.markdown("""
    **为什么这一步重要？**
    
    异方差违反了 OLS 的同方差假设，会导致标准误估计不准确，从而影响 t 检验和 F 检验的可靠性。
    使用稳健标准误可以在不改变系数估计的情况下，修正标准误的估计。
    """)
    
    het_result = test_heteroskedasticity(st.session_state.current_model)
    
    if het_result is None:
        st.error("❌ 异方差检验失败")
        return
    
    bp_result = het_result['breusch_pagan']
    white_result = het_result['white']
    
    st.markdown("### Breusch-Pagan 检验")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("检验统计量", f"{bp_result['statistic']:.4f}")
    with col2:
        st.metric("p 值", f"{bp_result['p_value']:.4f}")
    with col3:
        if bp_result['has_heteroskedasticity']:
            st.error("存在异方差")
        else:
            st.success("无异方差")
    
    st.markdown("### White 检验")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("检验统计量", f"{white_result['statistic']:.4f}")
    with col2:
        st.metric("p 值", f"{white_result['p_value']:.4f}")
    with col3:
        if white_result['has_heteroskedasticity']:
            st.error("存在异方差")
        else:
            st.success("无异方差")
    
    has_heteroskedasticity = bp_result['has_heteroskedasticity'] or white_result['has_heteroskedasticity']
    
    if not has_heteroskedasticity:
        st.success("✅ 未检测到显著的异方差问题")
        st.markdown("**你的选择：**")
        if st.button("➡️ 结束诊断", use_container_width=True):
            st.session_state.gates_ignored[4] = True
            go_to_next_step(5)
        return
    
    st.warning("⚠️ 检测到异方差问题")
    
    st.markdown("**你现在有哪些选择：**")
    st.markdown("1. **使用稳健标准误**：采用 HC3 稳健标准误重新估计模型")
    st.markdown("2. **结束诊断**：保持当前模型，进入最终结果展示")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🛡️ 使用稳健标准误（HC3）重新回归", type="primary", use_container_width=True):
            st.session_state.previous_model = st.session_state.current_model
            
            model, error = refit_with_robust_se(
                st.session_state.current_df,
                st.session_state.target_var,
                st.session_state.feature_vars,
                st.session_state.extra_terms
            )
            
            if error:
                st.error(f"❌ {error}")
                return
            
            st.session_state.current_model = model
            st.session_state.use_robust_se = True
            st.session_state.gates_completed[4] = True
            st.session_state.last_action = "使用 HC3 稳健标准误"
            st.session_state.model_history.append({
                'step': 'heteroskedasticity_fix',
                'model': model,
                'description': '使用 HC3 稳健标准误'
            })
            
            st.success("✅ 已使用稳健标准误重新估计模型")
            st.session_state.show_comparison = True
            st.rerun()
    
    with col2:
        if st.button("🏁 结束诊断", use_container_width=True):
            st.session_state.gates_ignored[4] = True
            go_to_next_step(5)

def render_completion_section():
    st.subheader("🎉 调优完成")
    
    st.success("🎊 恭喜！您已完成所有诊断步骤！")
    
    # 1. 最终模型结果
    st.markdown("### 📊 最终模型结果")
    display_model_result(st.session_state.current_model, "最终模型")
    
    # 2. 初始 vs 最终对比（只保留核心指标）
    if st.session_state.initial_model and st.session_state.current_model:
        st.markdown("---")
        st.subheader("📊 初始模型 vs 最终模型")
        
        # 只展示核心指标
        prev_info, _ = extract_model_summary(st.session_state.initial_model)
        curr_info, _ = extract_model_summary(st.session_state.current_model)
        
        comparison_data = {
            '指标': ['R-squared', 'Adj. R-squared'],
            '初始模型': [
                f"{prev_info['R-squared']:.4f}",
                f"{prev_info['Adj. R-squared']:.4f}"
            ],
            '最终模型': [
                f"{curr_info['R-squared']:.4f}",
                f"{curr_info['Adj. R-squared']:.4f}"
            ],
            '变化': [
                f"{(curr_info['R-squared'] - prev_info['R-squared']):+.4f}",
                f"{(curr_info['Adj. R-squared'] - prev_info['Adj. R-squared']):+.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.astype(str), use_container_width=True)
        
        # 3. 回归系数对比（精简版）
        st.markdown("### 📋 回归系数对比")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 初始模型")
            _, initial_results = extract_model_summary(st.session_state.initial_model)
            if not initial_results.empty:
                initial_coef_df = initial_results[['变量', '系数', 'p值']].copy()
                initial_coef_df['系数'] = initial_coef_df['系数'].apply(lambda x: f"{x:.4f}")
                initial_coef_df['p值'] = initial_coef_df['p值'].apply(lambda x: f"{x:.4f}")
                st.dataframe(initial_coef_df.astype(str), use_container_width=True)
        
        with col2:
            st.markdown("#### 最终模型")
            _, final_results = extract_model_summary(st.session_state.current_model)
            if not final_results.empty:
                final_coef_df = final_results[['变量', '系数', 'p值']].copy()
                final_coef_df['系数'] = final_coef_df['系数'].apply(lambda x: f"{x:.4f}")
                final_coef_df['p值'] = final_coef_df['p值'].apply(lambda x: f"{x:.4f}")
                st.dataframe(final_coef_df.astype(str), use_container_width=True)
        
        # 4. 一句话总结
        r2_change = curr_info['R-squared'] - prev_info['R-squared']
        if r2_change > 0.01:
            improvement = "有所提升"
        elif r2_change < -0.01:
            improvement = "略有下降"
        else:
            improvement = "无明显变化"
        
        # 分析主要优化步骤
        completed_steps = [k for k, v in st.session_state.gates_completed.items() if v]
        main_optimizations = []
        if 1 in completed_steps:
            main_optimizations.append("极值处理")
        if 2 in completed_steps:
            main_optimizations.append("共线性处理")
        if 3 in completed_steps:
            main_optimizations.append("交互项引入")
        if 4 in completed_steps:
            main_optimizations.append("异方差修正")
        
        if main_optimizations:
            summary = f"模型整体解释力{improvement}，主要优化来自{('与'.join(main_optimizations))}"
        else:
            summary = f"模型整体解释力{improvement}"
        
        st.info(f"💡 {summary}")
    
    if st.button("🔄 重新开始", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        init_session_state()
        st.rerun()

def render_comparison_section():
    if st.session_state.previous_model and st.session_state.current_model:
        display_comparison_board(
            st.session_state.previous_model,
            st.session_state.current_model,
            st.session_state.last_action
        )

def main():
    init_session_state()
    render_header()
    render_progress_bar()
    
    st.markdown("---")
    
    if not st.session_state.baseline_initialized:
        render_data_upload_section()
    else:
        # 只有在非 Step5 页面时显示基线模型结果
        if st.session_state.current_gate != 5:
            render_baseline_model_section()
            st.markdown("---")
        
        if st.session_state.current_gate == 1:
            render_outlier_diagnosis()
        elif st.session_state.current_gate == 2:
            render_collinearity_diagnosis()
        elif st.session_state.current_gate == 3:
            render_nonlinearity_diagnosis()
        elif st.session_state.current_gate == 4:
            render_heteroskedasticity_diagnosis()
        elif st.session_state.current_gate == 5:
            render_completion_section()
        
        # 只有在非 Step5 页面时显示对比块
        if st.session_state.show_comparison and st.session_state.current_gate != 5:
            st.markdown("---")
            render_comparison_block()

if __name__ == "__main__":
    main()
