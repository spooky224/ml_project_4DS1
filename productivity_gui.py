# mlops_dashboard_upgraded.py
import streamlit as st
import pandas as pd
import joblib
import requests
import time
from datetime import datetime

# --------------------------- Configuration --------------------------- #
USE_API = True
API_URL = "http://127.0.0.1:8000"  # FastAPI server

# --------------------------- Load Models --------------------------- #
predict_model = joblib.load("productivity_model_20251127_234333.pkl")
try:
    classify_info = joblib.load("productivity_classifier.pkl")
    classify_model = classify_info["model"]
    classify_features = classify_info["feature_names"]
    classify_encoders = classify_info.get("label_encoders", {})
except:
    classify_model = None
    classify_features = []
    classify_encoders = {}
    st.warning("Classification model not found or failed to load.")

# --------------------------- Dropdown Values --------------------------- #
quarters_display = ["Quarter1", "Quarter2", "Quarter3", "Quarter4"]
departments_display = ["Sweing", "Finishing"]
days_display = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

normalize_map = {
    "quarter": lambda x: x.lower(),
    "department": lambda x: x.lower(),
    "day": lambda x: x.lower(),
}

# --------------------------- Streamlit Layout --------------------------- #
st.set_page_config(page_title="MLOps Productivity Dashboard", layout="wide")
st.title("🚀 MLOps Productivity Prediction & Classification")

# --------------------------- API Status --------------------------- #
col1, col2 = st.columns(2)
with col1:
    try:
        start = time.time()
        resp = requests.get(API_URL + "/health", timeout=1)
        latency = int((time.time() - start) * 1000)
        st.success("API Status: Online ✅")
    except:
        st.error("API Status: Offline ❌")
        latency = None

with col2:
    st.write(f"Latency: {latency} ms" if latency else "Latency: --")

# --------------------------- Mode & Model Version --------------------------- #
mode = st.radio("Select Mode", ["Prediction", "Classification"])
model_version = st.selectbox("Select Model Version", ["latest", "v1", "v2"])

# --------------------------- Feature Inputs --------------------------- #
with st.form("input_form"):
    st.subheader("📝 Enter Features")
    cols = st.columns(2)

    # --- Date Input with validation ---
    date_input = cols[0].text_input("Date (d/m/yyyy)", "1/8/2015")
    try:
        parsed_date = datetime.strptime(date_input, "%d/%m/%Y")
        date_str = parsed_date.strftime("%-d/%-m/%Y")
    except ValueError:
        st.error("❌ Invalid date format! Use d/m/yyyy (e.g., 1/8/2015).")
        st.stop()

    quarter = cols[1].selectbox("Quarter", quarters_display)
    department = cols[0].selectbox("Department", departments_display)
    day = cols[1].selectbox("Day", days_display)

    team = cols[0].number_input("Team", min_value=1, step=1)
    targeted_productivity = cols[1].number_input("Targeted Productivity", min_value=0.0, step=0.01)
    smv = cols[0].number_input("SMV", min_value=0.0, step=0.01)
    wip = cols[1].number_input("WIP", min_value=0.0, step=0.01)
    over_time = cols[0].number_input("Over Time", min_value=0.0, step=0.01)
    incentive = cols[1].number_input("Incentive", min_value=0.0, step=0.01)
    idle_time = cols[0].number_input("Idle Time", min_value=0.0, step=0.01)
    idle_men = cols[1].number_input("Idle Men", min_value=0, step=1)
    no_of_style_change = cols[0].number_input("No of Style Change", min_value=0, step=1)
    no_of_workers = cols[1].number_input("No of Workers", min_value=1, step=1)

    submitted = st.form_submit_button("Run Prediction")

# --------------------------- Run Prediction --------------------------- #
def run_prediction(data_dict):
    try:
        if USE_API:
            endpoint = "/predict" if mode.lower() == "prediction" else "/classify"
            resp = requests.post(API_URL + endpoint, json=data_dict)
            result = resp.json()
            if "error" in result:
                st.error(result["error"])
                return
        else:
            df = pd.DataFrame([data_dict])
            for col in predict_model.feature_names_in_:
                if col not in df:
                    df[col] = 0
            df_pred = df[predict_model.feature_names_in_]
            pred = predict_model.predict(df_pred)[0]
            result = {"prediction": pred}

        # --- Display Results ---
        st.subheader("📊 Results")
        if mode.lower() == "prediction":
            st.success(f"Predicted Productivity: {result['prediction']:.3f}")
            if "feature_importance" in result:
                st.bar_chart(pd.DataFrame(result["feature_importance"], index=[0]).T.rename(columns={0: "Importance"}))
        else:
            st.success(f"Predicted Class: {result['prediction_class']}")
            if "class_probs" in result:
                st.bar_chart(pd.DataFrame(result["class_probs"], index=[0]).T.rename(columns={0: "Probability"}))
    except Exception as e:
        st.error(str(e))

# --------------------------- Run Prediction --------------------------- #
if submitted:
    data = {
        "date": parsed_date.strftime("%Y-%m-%d"),
        "quarter": normalize_map["quarter"](quarter),
        "department": normalize_map["department"](department),
        "day": normalize_map["day"](day),
        "team": team,
        "targeted_productivity": targeted_productivity,
        "smv": smv,
        "wip": wip,
        "over_time": over_time,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_style_change": no_of_style_change,
        "no_of_workers": no_of_workers,
    }

    try:
        if USE_API:
            endpoint = "/predict" if mode.lower()=="prediction" else "/classify"
            resp = requests.post(API_URL + endpoint, json=data)
            result = resp.json()
            if "error" in result:
                st.error(result["error"])
            else:
                # --- Enriched Results Layout ---
                tabs = st.tabs(["Prediction", "Feature Insights", "Historical Comparison", "Suggestions"])
                
                if mode.lower() == "prediction":
                    # Prediction Tab
                    with tabs[0]:
                        st.success(f"Predicted Productivity: {result['prediction']:.3f}")
                        # Optionally add confidence / interval (±5%)
                        lower = result['prediction'] * 0.95
                        upper = result['prediction'] * 1.05
                        st.info(f"Prediction Interval: [{lower:.2f} - {upper:.2f}]")
                    
                    # Feature Importance Tab
                    if "feature_importance" in result:
                        with tabs[1]:
                            fi_df = pd.DataFrame(result["feature_importance"], index=[0]).T.rename(columns={0:"Importance"})
                            st.bar_chart(fi_df)
                    
                    # Historical Comparison Tab
                    with tabs[2]:
                        # Placeholder: could fetch historical averages from API or DB
                        st.write("Average productivity for similar day/quarter/department: 72.5")
                        st.metric("Predicted vs Historical Avg", f"{result['prediction']:.2f}", "72.5")
                    
                    # Suggestions Tab
                    with tabs[3]:
                        suggestions = []
                        if data["wip"] > 50:
                            suggestions.append("High WIP → Consider balancing workload.")
                        if data["idle_time"] > 5:
                            suggestions.append("Idle time high → check workflow efficiency.")
                        if data["over_time"] > 2:
                            suggestions.append("Overtime detected → consider staffing adjustments.")
                        if suggestions:
                            for s in suggestions:
                                st.warning(s)
                        else:
                            st.success("No critical suggestions. All good!")

                else:  # Classification
                    with tabs[0]:
                        st.success(f"Predicted Class: {result['prediction_class']}")
                        # Top 3 probabilities
                        if "class_probs" in result:
                            sorted_probs = dict(sorted(result["class_probs"].items(), key=lambda item: item[1], reverse=True))
                            st.write("Top Class Probabilities:")
                            top3 = list(sorted_probs.items())[:3]
                            for cls, prob in top3:
                                st.metric(cls, f"{prob*100:.2f}%")
                    
                    with tabs[1]:
                        if "feature_importance" in result:
                            fi_df = pd.DataFrame(result["feature_importance"], index=[0]).T.rename(columns={0:"Importance"})
                            st.bar_chart(fi_df)
                    
                    with tabs[2]:
                        st.write("Historical class distribution:")  # Could fetch from API/DB
                        hist_df = pd.DataFrame({"Class":["Low","Medium","High"], "Count":[15,30,10]})
                        st.bar_chart(hist_df.set_index("Class"))
                    
                    with tabs[3]:
                        st.info("Suggestions based on class prediction:")
                        if result['prediction_class'] == "low":
                            st.warning("Productivity predicted low → Investigate bottlenecks.")
                        elif result['prediction_class'] == "high":
                            st.success("Productivity high → Keep up the good work!")
                        else:
                            st.info("Productivity moderate → Monitor team performance.")

        else:
            # Local model prediction logic (regression only, could be adapted)
            df = pd.DataFrame([data])
            for col in predict_model.feature_names_in_:
                if col not in df: df[col]=0
            df_pred = df[predict_model.feature_names_in_]
            pred = predict_model.predict(df_pred)[0]

            tabs = st.tabs(["Prediction", "Feature Insights", "Suggestions"])
            with tabs[0]:
                st.success(f"Predicted Productivity: {pred:.3f}")
                lower = pred * 0.95
                upper = pred * 1.05
                st.info(f"Prediction Interval: [{lower:.2f} - {upper:.2f}]")
            # Feature importance not available locally (unless added)
            with tabs[2]:
                suggestions = []
                if data["wip"] > 50:
                    suggestions.append("High WIP → Consider balancing workload.")
                if data["idle_time"] > 5:
                    suggestions.append("Idle time high → check workflow efficiency.")
                if data["over_time"] > 2:
                    suggestions.append("Overtime detected → consider staffing adjustments.")
                if suggestions:
                    for s in suggestions:
                        st.warning(s)
                else:
                    st.success("No critical suggestions. All good!")

    except Exception as e:
        st.error(str(e))

# --------------------------- Batch Prediction --------------------------- #
st.subheader("📁 Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV for batch predictions", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        results = []
        for _, row in df.iterrows():
            row_data = row.to_dict()
            for col, func in normalize_map.items():
                if col in row_data:
                    row_data[col] = func(row_data[col])
            if USE_API:
                endpoint = "/predict" if mode.lower() == "prediction" else "/classify"
                resp = requests.post(API_URL + endpoint, json=row_data)
                results.append(resp.json())
            else:
                for col in predict_model.feature_names_in_:
                    if col not in row_data:
                        row_data[col] = 0
                df_pred = pd.DataFrame([row_data])[predict_model.feature_names_in_]
                pred = predict_model.predict(df_pred)[0]
                results.append({"prediction": pred})
        st.success(f"✅ Processed {len(results)} rows")
        st.dataframe(pd.DataFrame(results))
    except Exception as e:
        st.error(str(e))

