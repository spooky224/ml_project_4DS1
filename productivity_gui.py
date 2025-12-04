import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import requests

USE_API = True
API_URL = "http://0.0.0.0:8000"  # Make sure your FastAPI server is running

# =========================================
# Load Models
# =========================================

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
    print("Warning: Classification model not found or failed to load.")


# =========================================
# Dropdown Values (Human-friendly)
# =========================================
quarters_display = ["Quarter1", "Quarter2", "Quarter3", "Quarter4"]
departments_display = ["Sweing", "Finishing"]
days_display = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Mapping GUI input → model expected lowercase labels
normalize_map = {
    "quarter": lambda x: x.lower(),             # Quarter3 → quarter3
    "department": lambda x: x.lower(),         # Sweing → sweing
    "day": lambda x: x.lower(),                # Thursday → thursday
}


# =========================================
# GUI Window
# =========================================
root = tk.Tk()
root.title("Productivity Prediction & Classification")
root.geometry("720x850")
root.configure(bg="#eef2f5")

title_label = tk.Label(
    root,
    text="Productivity Prediction / Classification",
    font=("Arial", 18, "bold"),
    bg="#eef2f5"
)
title_label.pack(pady=20)


# =========================================
# MODE SELECTION
# =========================================
mode_var = tk.StringVar(value="prediction")

mode_frame = tk.Frame(root, bg="#eef2f5")
mode_frame.pack(pady=10)

tk.Label(mode_frame, text="Select Mode:", font=("Arial", 12), bg="#eef2f5").pack(side="left", padx=10)

ttk.Radiobutton(mode_frame, text="Prediction", variable=mode_var, value="prediction").pack(side="left", padx=5)
ttk.Radiobutton(mode_frame, text="Classification", variable=mode_var, value="classification").pack(side="left", padx=5)


# =========================================
# FEATURE INPUT FORM
# =========================================
fields_frame = tk.Frame(root, bg="#eef2f5")
fields_frame.pack(pady=20)

fields = {
    "date": tk.Entry,
    "quarter": ttk.Combobox,
    "department": ttk.Combobox,
    "day": ttk.Combobox,
    "team": tk.Entry,
    "targeted_productivity": tk.Entry,
    "smv": tk.Entry,
    "wip": tk.Entry,
    "over_time": tk.Entry,
    "incentive": tk.Entry,
    "idle_time": tk.Entry,
    "idle_men": tk.Entry,
    "no_of_style_change": tk.Entry,
    "no_of_workers": tk.Entry,
}

entries = {}

for f, widget in fields.items():
    frame = tk.Frame(fields_frame, bg="#eef2f5")
    frame.pack(pady=6, padx=20, fill="x")

    tk.Label(frame, text=f.capitalize().replace("_", " "), width=20, anchor="w", font=("Arial", 10), bg="#eef2f5").pack(side="left")
    
    if widget == ttk.Combobox:
        values = (
            quarters_display if f == "quarter"
            else departments_display if f == "department"
            else days_display
        )
        cb = ttk.Combobox(frame, values=values, state="readonly")
        cb.current(0)
        cb.pack(side="left", fill="x", expand=True)
        entries[f] = cb
    else:
        e = widget(frame)
        e.pack(side="left", fill="x", expand=True)
        entries[f] = e


# =========================================
# RUN BUTTON ACTION
# =========================================

def run():
    try:
        data = {f: entries[f].get() for f in fields}

        # Validate empty fields
        for key, value in data.items():
            if value == "":
                messagebox.showerror("Input Error", f"Field '{key}' cannot be empty.")
                return

        # Normalize categorical values
        for col, func in normalize_map.items():
            data[col] = func(data[col])

        # Convert numeric fields
        numeric_fields = [
            "team", "targeted_productivity", "smv", "wip", "over_time",
            "incentive", "idle_time", "idle_men", "no_of_style_change", "no_of_workers"
        ]
        for f in numeric_fields:
            try:
                data[f] = float(data[f])
            except ValueError:
                messagebox.showerror("Input Error", f"'{f}' must be a number.")
                return

        if USE_API:
            # Call API
            endpoint = "/predict" if mode_var.get() == "prediction" else "/classify"
            response = requests.post(API_URL + endpoint, json=data)
            result = response.json()
            
            if "error" in result:
                messagebox.showerror("API Error", result["error"])
                return

            if mode_var.get() == "prediction":
                messagebox.showinfo("Prediction Result", f"Predicted Productivity: {result['prediction']:.3f}")
            else:
                messagebox.showinfo("Classification Result", f"Predicted Class: {result['prediction_class']}")
        
        else:
            # Existing local prediction logic
            df = pd.DataFrame([data])
            for col in predict_model.feature_names_in_:
                if col not in df:
                    df[col] = 0
            df_pred = df[predict_model.feature_names_in_]
            pred = predict_model.predict(df_pred)[0]
            messagebox.showinfo("Prediction Result", f"Predicted Productivity: {pred:.3f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# =========================================
# RUN BUTTON WITH HOVER EFFECT
# =========================================
def button_hover(e):
    run_btn.config(bg="#4CAF50", fg="white")

def button_leave(e):
    run_btn.config(bg="#e0e0e0", fg="black")

run_btn = tk.Button(root, text="Run", command=run, font=("Arial", 13, "bold"), bg="#e0e0e0", width=22)
run_btn.pack(pady=25)

run_btn.bind("<Enter>", button_hover)
run_btn.bind("<Leave>", button_leave)

# Start GUI
root.mainloop()

