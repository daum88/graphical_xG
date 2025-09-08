"""
Utility functions for the xG calculator
"""
import numpy as np
import pandas as pd
import logging

# Constants
GOAL_X = 120
GOAL_Y = 40
CANVAS_WIDTH = 450
CANVAS_HEIGHT = 270
PITCH_WIDTH = 120
PITCH_HEIGHT = 80

logger = logging.getLogger(__name__)

def calculate_distance_to_goal(x, y):
    """Calculate distance from shot location to goal"""
    return np.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2)

def calculate_angle_to_goal(x, y):
    """Calculate angle from shot location to goal"""
    return np.arctan2(abs(GOAL_Y - y), GOAL_X - x)

def gui_to_pitch_coordinates(gui_x, gui_y):
    """Convert GUI coordinates to pitch coordinates"""
    pitch_x = gui_x * (PITCH_WIDTH / CANVAS_WIDTH)
    pitch_y = gui_y * (PITCH_HEIGHT / CANVAS_HEIGHT)
    return pitch_x, pitch_y

def pitch_to_gui_coordinates(pitch_x, pitch_y):
    """Convert pitch coordinates to GUI coordinates"""
    gui_x = pitch_x * (CANVAS_WIDTH / PITCH_WIDTH)
    gui_y = pitch_y * (CANVAS_HEIGHT / PITCH_HEIGHT)
    return gui_x, gui_y

def validate_coordinates(x, y, coord_type="pitch"):
    """Validate coordinates are within bounds"""
    if coord_type == "pitch":
        return 0 <= x <= PITCH_WIDTH and 0 <= y <= PITCH_HEIGHT
    elif coord_type == "gui":
        return 0 <= x <= CANVAS_WIDTH and 0 <= y <= CANVAS_HEIGHT
    return False

def create_feature_vector(features_list, x, y, shot_type, body_part, under_pressure=False):
    """Create feature vector for model prediction"""
    # Initialize with zeros
    feature_dict = {feature: 0 for feature in features_list}
    
    # Calculate derived features
    distance = calculate_distance_to_goal(x, y)
    angle = calculate_angle_to_goal(x, y)
    
    # Set basic features
    if "x" in feature_dict:
        feature_dict["x"] = x
    if "y" in feature_dict:
        feature_dict["y"] = y
    if "distance_to_goal" in feature_dict:
        feature_dict["distance_to_goal"] = distance
    if "angle_to_goal" in feature_dict:
        feature_dict["angle_to_goal"] = angle
    
    # Set time_remaining to default value (assume 45 minutes remaining)
    if "time_remaining" in feature_dict:
        feature_dict["time_remaining"] = 45
    
    # Set pressure features (handle duplicate under_pressure columns)
    if under_pressure:
        for feature in features_list:
            if "under_pressure" in feature:
                feature_dict[feature] = 1
    
    # Set rebound feature to default (not a rebound)
    if "rebound" in feature_dict:
        feature_dict["rebound"] = 0
    
    # Set categorical features
    shot_type_feature = f"shot_type_{shot_type}"
    if shot_type_feature in feature_dict:
        feature_dict[shot_type_feature] = 1
    
    body_part_feature = f"shot_body_part_{body_part}"
    if body_part_feature in feature_dict:
        feature_dict[body_part_feature] = 1
    
    return feature_dict

def format_xg_result(xg_value, distance, angle):
    """Format xG result for display"""
    angle_degrees = np.degrees(angle)
    return f"xG: {xg_value:.3f} | Distance: {distance:.1f}m | Angle: {angle_degrees:.1f}°"
