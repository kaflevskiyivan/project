import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import joblib
import shap

st.set_page_config(page_title="КП", layout="wide")
st.title("Виявлення шахрайських фінансових операцій у платіжних системах")

@st.cache_resource
def load_models():
    models = {}

    models['real'] = joblib.load('model_real.pkl')
    models['real_cols'] = joblib.load('cols_real.pkl')

    models['synth'] = joblib.load('model_synthetic.pkl')
    models['synth_cols'] = joblib.load('cols_synthetic.pkl')

    return models

models_dict = load_models()

if 'audit_log' not in st.session_state:
    st.session_state.audit_log = pd.DataFrame(columns=[
        "Час перевірки", "Модель", "Основні параметри", "Ймовірність ризику", "Результат"
    ])

v_features_data = {
    'V17': [0.05, 0.15, -1.12, -0.59, 0.02, -8.10, -9.25, -6.85, -11.48, -7.35],
    'V14': [0.08, 0.18, -0.32, -0.12, 0.05, -7.21, -6.80, -5.10, -9.35, -8.21],
    'V12': [0.12, 0.22, -0.45, -0.24, -0.01, -6.45, -7.10, -4.90, -8.15, -6.05],
    'V10': [0.02, 0.09, -0.12, -0.35, 0.01, -5.34, -6.11, -4.20, -7.08, -5.11],
    'V16': [0.04, 0.11, -0.28, -0.11, -0.03, -4.89, -5.40, -3.75, -6.12, -4.08],
    'V3':  [0.10, 0.25, -0.15, -0.40, 0.02, -4.50, -5.12, -3.20, -6.14, -5.09],
    'V7':  [0.03, 0.14, -0.11, -0.25, 0.01, -3.95, -4.12, -2.90, -5.09, -4.05],
    'V11': [-0.01, -0.15, 0.45, 0.12, 0.02, 5.12, 5.80, 4.10, 7.05, 6.11],
    'V4':  [-0.05, -0.21, 0.65, 0.35, 0.01, 6.21, 7.02, 4.80, 8.12, 6.08],
    'V18': [0.01, 0.08, -0.22, -0.15, 0.03, -2.85, -3.15, -1.95, -4.05, -3.02]
}

formatted_options = {}
for feat, values in v_features_data.items():
    formatted_options[feat] = [f"{v} (Нормальна)" if i < 5 else f"{v} (Шахрайська)" for i, v in enumerate(values)]

col_left, col_right = st.columns([1.2, 1], gap="large")
current_result = None

with col_left:
    st.subheader("Введення даних")
    
    with st.expander("Модель 2: Штучні дані", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            synth_amount = st.number_input("Transaction Amount", min_value=0.0, value=150.0, step=10.0)
            synth_hour = st.number_input("Transaction Hour", min_value=0, max_value=23, value=14)
            synth_minute = st.number_input("Transaction Minute", min_value=0, max_value=59, value=30, step=5)
        with col2:
            synth_merchant = st.number_input("Merchant ID", min_value=-5.0, max_value=50.0, value=1.0, step=0.05)
            synth_day = st.number_input("Transaction Day", min_value=1, max_value=31, value=15)
            synth_second = st.number_input("Transaction Second", min_value=0, max_value=59, value=0, step=5)
        with col3:
            synth_month = st.number_input("Transaction Month", min_value=1, max_value=12, value=5)
            synth_year = st.number_input("Transaction Year", min_value=2000, max_value=2030, value=2024)

        if st.button("Аналіз транзакції"):
            input_data = {
                'Transaction Amount': synth_amount,
                'Merchant ID': synth_merchant,
                'Transaction Month': synth_month,
                'Transaction Year': synth_year,
                'Transaction Day': synth_day,
                'Transaction Hour': synth_hour,
                'Transaction Minute': synth_minute,
                'Transaction Second': synth_second
            }

            full_vector = []
            for col in models_dict['synth_cols']:
                full_vector.append(input_data.get(col, 0.0))

            pred_proba = models_dict['synth'].predict_proba([full_vector])[0][1]

            THRESHOLD = 0.6
            pred_class = 1 if pred_proba >= THRESHOLD else 0

            current_result = {
                "type": "Штучна модель",
                "proba": pred_proba,
                "class": pred_class,
                "params": f"Сума: ${synth_amount}, Merchant: {synth_merchant}",
                "features": input_data,
                "full_vector": full_vector
            }

    with st.expander("Модель 1: Реальні дані", expanded=True):
        st.write("Оберіть значення для кожної головної компоненти:")

        c1, c2 = st.columns(2)
        selected_real_features = {}
        features_order = ['V17', 'V14', 'V12', 'V10', 'V16', 'V3', 'V7', 'V11', 'V4', 'V18']

        for i, feat in enumerate(features_order):
            if i < 5:
                with c1:
                    choice = st.selectbox(feat, formatted_options[feat], key=f"sel_{feat}")
                    selected_real_features[feat] = float(choice.split(" ")[0])
            else:
                with c2:
                    choice = st.selectbox(feat, formatted_options[feat], key=f"sel_{feat}")
                    selected_real_features[feat] = float(choice.split(" ")[0])

        if st.button("Аналіз транзакції "):
            full_vector = []
            for col in models_dict['real_cols']:
                full_vector.append(selected_real_features.get(col, 0.0))

            pred_proba = models_dict['real'].predict_proba([full_vector])[0][1]
            pred_class = models_dict['real'].predict([full_vector])[0]

            current_result = {
                "type": "Реальна модель",
                "proba": pred_proba,
                "class": pred_class,
                "params": f"V17: {selected_real_features['V17']}, V12: {selected_real_features['V12']}, V14: {selected_real_features['V14']}",
                "features": selected_real_features,
                "full_vector": full_vector
            }

with col_right:
    st.subheader("Результат перевірки та пояснення")
    if current_result:
        risk_percentage = int(current_result['proba'] * 100)

        if current_result['class'] == 1:
            st.error(f"ШАХРАЙСТВО (Ризик: {risk_percentage}%)")
            status_emoji = "Шахрайство"
        else:
            st.success(f"ЛЕГІТИМНА (Ризик: {risk_percentage}%)")
            status_emoji = "Легітимна"

        st.write(f"**Деталі:** {current_result['params']}")
        st.write("**Пояснення за ознаками (Вплив на рішення моделі):**")

        contributions = []

        feature_names = list(current_result['features'].keys())
        feature_values = list(current_result['features'].values())
        vec_to_analyze = current_result['full_vector']

        active_model = (
            models_dict['real']
            if current_result['type'] == "Реальна модель"
            else models_dict['synth']
        )

        try:
            explainer = shap.TreeExplainer(active_model)
            shap_vals = explainer.shap_values(np.array([vec_to_analyze]))

            if isinstance(shap_vals, list):
                contributions = shap_vals[1][0]
            else:
                hap_vals = np.array(shap_vals)

                if len(shap_vals.shape) == 3:
                    contributions = shap_vals[0, :, 1]
                else:
                    contributions = shap_vals[0]

        except Exception as e:
            st.error(f"Помилка SHAP: {e}")
            contributions = [0] * len(feature_names)

        if len(contributions) != len(feature_names):
            contributions = contributions[:len(feature_names)] + [0] * max(0, len(feature_names) - len(contributions))

        shap_df = pd.DataFrame({
            'Ознака': feature_names,
            'Вплив (SHAP)': contributions,
            'Значення': feature_values
        })

        shap_df['Абсолютний вплив'] = shap_df['Вплив (SHAP)'].abs()
        shap_df = shap_df.sort_values(by='Абсолютний вплив', ascending=False).head(8)

        fig = px.bar(
            shap_df,
            x='Вплив (SHAP)',
            y='Ознака',
            orientation='h',
            color='Вплив (SHAP)',
            color_continuous_scale=px.colors.diverging.RdBu_r,
            color_continuous_midpoint=0,
            hover_data=['Значення'],
        )

        fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=20))
        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")

        st.plotly_chart(fig, use_container_width=True)

        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        new_row = pd.DataFrame([{
            "Час перевірки": now_str,
            "Модель": current_result['type'],
            "Основні параметри": current_result['params'],
            "Ймовірність ризику": f"{risk_percentage}%",
            "Результат": status_emoji
        }])

        st.session_state.audit_log = pd.concat(
            [new_row, st.session_state.audit_log],
            ignore_index=True
        )

st.write("---")
st.subheader(" Журнал")

if not st.session_state.audit_log.empty:
    st.dataframe(st.session_state.audit_log, use_container_width=True)
