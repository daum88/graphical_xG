import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import joblib
import numpy as np
import pandas as pd

# Load trained model, features, and scaler
xg_model = joblib.load("xg_model.joblib")
features = joblib.load("features.joblib")
scaler = joblib.load("scaler.joblib")

# Initialize GUI
root = tk.Tk()
root.title("Enhanced xG Calculator")

pitch_img = Image.open("football_pitch.jpg").resize((450, 270), Image.LANCZOS)
pitch_tk = ImageTk.PhotoImage(pitch_img)

canvas = tk.Canvas(root, width=450, height=270)
canvas.create_image(0, 0, anchor="nw", image=pitch_tk)
canvas.pack()

shot_x, shot_y = None, None

def set_shot_location(event):
    global shot_x, shot_y
    if shot_type.get() == "Penalty":
        shot_x, shot_y = 101 * (450/120), 40 * (270/80)
    else:
        shot_x, shot_y = event.x, event.y
    canvas.delete("shot_marker")
    canvas.create_oval(shot_x-5, shot_y-5, shot_x+5, shot_y+5, fill="red", tags="shot_marker")

canvas.bind("<Button-1>", set_shot_location)

# Body Part first
body_part_label = tk.Label(root, text="Body Part:")
body_part_label.pack()
body_part = ttk.Combobox(root, values=["Right Foot", "Left Foot", "Head", "Other"])
body_part.current(0)
body_part.pack()

# Shot Type based on Body Part
shot_type_label = tk.Label(root, text="Shot Type")
shot_type_label.pack()
shot_type = ttk.Combobox(root)
shot_type.pack()

def update_shot_type_options(event):
    if body_part.get() == "Head":
        shot_type['values'] = ["Open Play"]
        shot_type.current(0)
    else:
        shot_type['values'] = ["Open Play", "Free Kick", "Penalty", "Volley", "Half Volley"]
        shot_type.current(0)

body_part.bind("<<ComboboxSelected>>", update_shot_type_options)
update_shot_type_options(None)

pressure_var = tk.IntVar()
pressure_check = tk.Checkbutton(root, text="Under Pressure", variable=pressure_var)
pressure_check.pack()

result_label = tk.Label(root, text="Select location and input features, then calculate xG.")
result_label.pack()

# Corrected calculate xG function
def calculate_xg():
    global shot_x, shot_y
    if shot_x is None or shot_y is None:
        result_label.config(text="Please select shot location!")
        return

    pitch_x = shot_x * (120 / 450)
    pitch_y = shot_y * (80 / 270)
    distance = np.sqrt((120 - pitch_x)**2 + (40 - pitch_y)**2)
    angle = np.arctan2(abs(40 - pitch_y), 120 - pitch_x)

    input_dict = {feature: 0 for feature in features}

    input_dict.update({
        "x": pitch_x,
        "y": pitch_y,
        "distance_to_goal": distance,
        "angle_to_goal": angle,
    })

    shot_type_feature = f"shot_type_{shot_type.get()}"
    body_part_feature = f"shot_body_part_{body_part.get()}"
    
    if shot_type_feature in input_dict:
        input_dict[shot_type_feature] = 1
    if body_part_feature in input_dict:
        input_dict[body_part_feature] = 1
    
    # Set pressure feature (handle multiple under_pressure columns)
    if pressure_var.get():
        for feature in features:
            if "under_pressure" in feature:
                input_dict[feature] = 1
    
    # Set default values for additional features
    if "time_remaining" in input_dict:
        input_dict["time_remaining"] = 45  # Default: 45 minutes remaining
    if "rebound" in input_dict:
        input_dict["rebound"] = 0  # Default: not a rebound

    input_df = pd.DataFrame([input_dict])

    input_df[features] = scaler.transform(input_df[features])

    xg = xg_model.predict(input_df[features])[0]
    result_label.config(text=f"Expected Goals (xG): {xg:.3f}")

calculate_button = tk.Button(root, text="Calculate xG", command=calculate_xg)
calculate_button.pack()

root.mainloop()
