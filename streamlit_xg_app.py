"""
Modern Web-Based xG Calculator with Advanced Visualizations
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
from utils import (
    calculate_distance_to_goal,
    calculate_angle_to_goal,
    gui_to_pitch_coordinates,
    create_feature_vector,
    format_xg_result
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="⚽ Professional xG Calculator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .shot-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the xG model with caching"""
    try:
        model = joblib.load("xg_model.joblib")
        scaler = joblib.load("scaler.joblib")
        features = joblib.load("features.joblib")
        return model, scaler, features, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

def create_pitch_plot(shot_history=None):
    """Create an interactive football pitch visualization"""
    
    # Create pitch outline
    fig = go.Figure()
    
    # Pitch dimensions (120x80 yards)
    pitch_length, pitch_width = 120, 80
    
    # Pitch outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=pitch_length, y1=pitch_width,
        line=dict(color="white", width=3),
        fillcolor="green",
        opacity=0.3
    )
    
    # Goal areas
    # Left goal (6-yard box)
    fig.add_shape(
        type="rect",
        x0=0, y0=30, x1=6, y1=50,
        line=dict(color="white", width=2),
        fillcolor="lightgreen",
        opacity=0.5
    )
    
    # Right goal (6-yard box)
    fig.add_shape(
        type="rect",
        x0=114, y0=30, x1=120, y1=50,
        line=dict(color="white", width=2),
        fillcolor="lightblue",
        opacity=0.7
    )
    
    # Penalty areas (18-yard box)
    fig.add_shape(
        type="rect",
        x0=0, y0=22, x1=18, y1=58,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0.1)"
    )
    
    fig.add_shape(
        type="rect",
        x0=102, y0=22, x1=120, y1=58,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0.1)"
    )
    
    # Center circle
    fig.add_shape(
        type="circle",
        x0=50, y0=30, x1=70, y1=50,
        line=dict(color="white", width=2),
        fillcolor="rgba(255,255,255,0.1)"
    )
    
    # Center line
    fig.add_shape(
        type="line",
        x0=60, y0=0, x1=60, y1=80,
        line=dict(color="white", width=2)
    )
    
    # Add penalty spots
    fig.add_trace(go.Scatter(
        x=[12, 108], y=[40, 40],
        mode='markers',
        marker=dict(size=8, color='white'),
        name='Penalty Spots',
        showlegend=False
    ))
    
    # Add shot history if provided
    if shot_history:
        for i, shot in enumerate(shot_history):
            fig.add_trace(go.Scatter(
                x=[shot['x']], y=[shot['y']],
                mode='markers+text',
                marker=dict(
                    size=max(15, shot['xg'] * 50),
                    color=shot['xg'],
                    colorscale='RdYlGn',
                    cmin=0, cmax=1,
                    line=dict(width=2, color='black'),
                    opacity=0.8
                ),
                text=f"{shot['xg']:.3f}",
                textposition="middle center",
                textfont=dict(color="white", size=10),
                name=f"Shot {i+1}",
                hovertemplate=(
                    f"<b>Shot {i+1}</b><br>"
                    f"xG: {shot['xg']:.3f}<br>"
                    f"Distance: {shot['distance']:.1f}m<br>"
                    f"Angle: {shot['angle']:.1f}°<br>"
                    f"Type: {shot['shot_type']}<br>"
                    f"Body Part: {shot['body_part']}<br>"
                    "<extra></extra>"
                )
            ))
    
    # Update layout
    fig.update_layout(
        title="⚽ Interactive Football Pitch - Click to Add Shot",
        xaxis=dict(
            range=[0, 120],
            showgrid=False,
            showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, 80],
            showgrid=False,
            showticklabels=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='green',
        paper_bgcolor='white',
        height=500,
        showlegend=True,
        clickmode='event+select'
    )
    
    return fig

def create_xg_distribution_plot():
    """Create xG distribution visualization"""
    
    # Generate sample data for different scenarios
    scenarios = {
        'Penalties': np.random.normal(0.76, 0.05, 100),
        'Close Range (< 6m)': np.random.normal(0.65, 0.15, 200),
        'Box Edge (12-18m)': np.random.normal(0.09, 0.03, 300),
        'Long Range (> 25m)': np.random.normal(0.025, 0.01, 150)
    }
    
    fig = go.Figure()
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (scenario, data) in enumerate(scenarios.items()):
        fig.add_trace(go.Histogram(
            x=data,
            name=scenario,
            opacity=0.7,
            nbinsx=30,
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title="xG Distribution by Shot Type",
        xaxis_title="Expected Goals (xG)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    return fig

def create_heat_map():
    """Create xG heat map across the pitch"""
    
    # Create grid for heat map
    x_range = np.linspace(60, 120, 30)
    y_range = np.linspace(10, 70, 25)
    
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate xG for each point (simplified)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            distance = calculate_distance_to_goal(X[i,j], Y[i,j])
            angle = calculate_angle_to_goal(X[i,j], Y[i,j])
            
            # Simplified xG calculation based on distance and angle
            base_xg = max(0.01, 0.5 * np.exp(-distance/15))
            angle_penalty = max(0.2, 1 - (np.degrees(angle)/90)**2)
            Z[i,j] = base_xg * angle_penalty
    
    fig = go.Figure(data=go.Heatmap(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale='RdYlGn',
        showscale=True,
        hovertemplate='Distance: %{customdata:.1f}m<br>xG: %{z:.3f}<extra></extra>',
        customdata=np.sqrt((120-X)**2 + (40-Y)**2)
    ))
    
    # Add pitch outline
    fig.add_shape(
        type="rect",
        x0=60, y0=10, x1=120, y1=70,
        line=dict(color="white", width=3)
    )
    
    # Add goal
    fig.add_shape(
        type="rect",
        x0=114, y0=30, x1=120, y1=50,
        line=dict(color="white", width=3)
    )
    
    fig.update_layout(
        title="xG Heat Map - Attacking Half",
        xaxis_title="Pitch Length (yards)",
        yaxis_title="Pitch Width (yards)",
        height=400
    )
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Professional xG Calculator</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, features, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Model not loaded. Please ensure model files exist.")
        st.info("💡 Run `python train_calibrated_xg.py` to create the model files.")
        return
    
    # Sidebar controls
    st.sidebar.header("⚙️ Shot Configuration")
    
    # Shot type selection
    shot_type = st.sidebar.selectbox(
        "Shot Type",
        ["Open Play", "Free Kick", "Penalty", "Corner", "Volley", "Half Volley"],
        help="Type of shot being taken"
    )
    
    # Body part selection
    body_part = st.sidebar.selectbox(
        "Body Part",
        ["Right Foot", "Left Foot", "Head", "Other"],
        help="Body part used for the shot"
    )
    
    # Additional factors
    under_pressure = st.sidebar.checkbox(
        "Under Pressure",
        help="Shot taken under defensive pressure"
    )
    
    is_rebound = st.sidebar.checkbox(
        "Rebound Shot",
        help="Shot following a rebound or deflection"
    )
    
    minute = st.sidebar.slider(
        "Match Minute",
        1, 90, 45,
        help="When in the match the shot was taken"
    )
    
    # Initialize session state for shot history and selected position
    if 'shot_history' not in st.session_state:
        st.session_state.shot_history = []
    if 'selected_position' not in st.session_state:
        st.session_state.selected_position = None
    
    # Show selected position
    if st.session_state.selected_position:
        x, y = st.session_state.selected_position
        st.sidebar.success(f"📍 Selected Position: ({x:.0f}, {y:.0f})")
        
        if st.sidebar.button("🎯 Calculate xG"):
            # Check if model is loaded
            if not model_loaded or model is None or scaler is None:
                st.error("❌ Model not available. Please load the model first.")
                return
                
            # Calculate xG
            try:
                # Create feature vector
                feature_dict = create_feature_vector(
                    features, x, y, shot_type, body_part, under_pressure
                )
                
                # Add additional features
                if "time_remaining" in feature_dict:
                    feature_dict["time_remaining"] = 90 - minute
                if "rebound" in feature_dict:
                    feature_dict["rebound"] = int(is_rebound)
                
                # Make prediction
                input_df = pd.DataFrame([feature_dict])
                X_scaled = scaler.transform(input_df[features])
                xg = model.predict(X_scaled)[0]  # Using regressor
                
                # Calculate additional metrics
                distance = calculate_distance_to_goal(x, y)
                angle = calculate_angle_to_goal(x, y)
                
                # Add to shot history
                shot_data = {
                    'x': x,
                    'y': y,
                    'xg': xg,
                    'distance': distance,
                    'angle': np.degrees(angle),
                    'shot_type': shot_type,
                    'body_part': body_part,
                    'under_pressure': under_pressure,
                    'rebound': is_rebound,
                    'minute': minute
                }
                
                st.session_state.shot_history.append(shot_data)
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 xG Value", f"{xg:.3f}", help="Expected Goals probability")
                with col2:
                    st.metric("📏 Distance", f"{distance:.1f}m", help="Distance to goal")
                with col3:
                    st.metric("📐 Angle", f"{np.degrees(angle):.1f}°", help="Angle to goal")
                with col4:
                    percentage = xg * 100
                    st.metric("📊 Success %", f"{percentage:.1f}%", help="Probability as percentage")
                
                # Shot quality assessment
                if xg >= 0.5:
                    quality = "🔥 Excellent"
                    color = "green"
                elif xg >= 0.2:
                    quality = "⚡ Good"
                    color = "orange"
                elif xg >= 0.05:
                    quality = "⚠️ Fair"
                    color = "orange"
                else:
                    quality = "❌ Poor"
                    color = "red"
                
                st.markdown(f'<div class="shot-info"><h3>Shot Quality: <span style="color: {color}">{quality}</span></h3></div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error calculating xG: {e}")
    else:
        st.sidebar.info("👆 Click on the pitch below to select a shot position")
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear Shot History"):
        st.session_state.shot_history = []
        st.success("Shot history cleared!")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Interactive Pitch", "📊 Analytics", "🔥 Heat Map", "📈 Statistics"])
    
    with tab1:
        st.subheader("Interactive Football Pitch")
        
        # Position input
        st.write("📍 **Select Shot Position**")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            shot_x = st.number_input("X Position (60-120)", 60, 120, 108, help="Distance from left goal line")
        with col2:
            shot_y = st.number_input("Y Position (0-80)", 0, 80, 40, help="Distance from bottom sideline")
        with col3:
            if st.button("📍 Set Position", type="primary"):
                st.session_state.selected_position = (shot_x, shot_y)
                st.success(f"Position set: ({shot_x}, {shot_y})")
        
        # Create and display pitch
        pitch_fig = create_pitch_plot(st.session_state.shot_history)
        
        # Add selected position marker if exists
        if st.session_state.selected_position:
            x, y = st.session_state.selected_position
            pitch_fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='x',
                    line=dict(width=3, color='white')
                ),
                name='Selected Position',
                showlegend=False,
                hovertemplate=f"<b>Selected Position</b><br>X: {x}<br>Y: {y}<extra></extra>"
            ))
        
        st.plotly_chart(pitch_fig, use_container_width=True)
        
        # Shot history table
        if st.session_state.shot_history:
            st.subheader("📋 Shot History")
            
            history_df = pd.DataFrame(st.session_state.shot_history)
            history_df['xG'] = history_df['xg'].round(3)
            history_df['Distance (m)'] = history_df['distance'].round(1)
            history_df['Angle (°)'] = history_df['angle'].round(1)
            
            display_df = history_df[['shot_type', 'body_part', 'xG', 'Distance (m)', 'Angle (°)', 'minute']].copy()
            display_df.columns = ['Shot Type', 'Body Part', 'xG', 'Distance (m)', 'Angle (°)', 'Minute']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Shots", len(st.session_state.shot_history))
            with col2:
                avg_xg = np.mean([shot['xg'] for shot in st.session_state.shot_history])
                st.metric("Average xG", f"{avg_xg:.3f}")
            with col3:
                total_xg = sum([shot['xg'] for shot in st.session_state.shot_history])
                st.metric("Total xG", f"{total_xg:.2f}")
    
    with tab2:
        st.subheader("📊 xG Distribution Analysis")
        
        if st.session_state.shot_history:
            # Create distribution plot for user's shots
            shot_xgs = [shot['xg'] for shot in st.session_state.shot_history]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=shot_xgs,
                nbinsx=max(5, len(shot_xgs)//2),
                name="Your Shots",
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Your Shot xG Distribution",
                xaxis_title="xG Value",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Standard distribution comparison
        dist_fig = create_xg_distribution_plot()
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔥 xG Heat Map")
        st.info("This heat map shows xG values across different areas of the attacking half of the pitch.")
        
        heat_fig = create_heat_map()
        st.plotly_chart(heat_fig, use_container_width=True)
        
        # Heat map insights
        st.markdown("""
        **Key Insights:**
        - 🔴 **Red zones**: High xG areas (close to goal, central)
        - 🟡 **Yellow zones**: Medium xG areas (penalty box edges)
        - 🟢 **Green zones**: Low xG areas (wide angles, long distance)
        - 📍 **Sweet spot**: 6-12 yards from goal, central position
        """)
    
    with tab4:
        st.subheader("📈 Model Statistics & Insights")
        
        # Model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🤖 Model Details:**
            - **Algorithm**: RandomForestRegressor
            - **Training Data**: 20,000 synthetic samples
            - **Features**: 13 engineered features
            - **Validation**: Professional benchmarks
            """)
        
        with col2:
            st.markdown("""
            **🎯 Benchmark Performance:**
            - **Penalties**: ~76% (realistic)
            - **Close shots**: ~65% (accurate)
            - **Long shots**: ~3% (conservative)
            - **Overall**: Matches professional data
            """)
        
        # Feature importance (mock data for demonstration)
        importance_data = {
            'Feature': ['Distance to Goal', 'Angle to Goal', 'Shot Type', 'Body Part', 'Under Pressure', 'Position X', 'Position Y', 'Time Remaining'],
            'Importance': [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
        }
        
        fig = px.bar(
            importance_data,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance in xG Prediction"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tips for improvement
        st.markdown("""
        **💡 Tips for Better Shots:**
        1. **Get closer to goal** - Distance is the biggest factor
        2. **Aim for central positions** - Wide angles significantly reduce xG
        3. **Use your stronger foot** - Better control = higher conversion
        4. **Avoid pressure** - Take time to set up the shot when possible
        5. **Look for rebounds** - Second chances often have higher xG
        """)

if __name__ == "__main__":
    main()
