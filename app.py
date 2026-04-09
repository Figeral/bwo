import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
from functions import BenchmarkFunctions
from bwo import BlackWidowOptimization

# Page Configuration
st.set_page_config(page_title="Black Widow Optimization", layout="wide", page_icon="🕷️")

st.title("🕷️ Black Widow Optimization (BWO) Algorithm Visualization")
st.markdown("""
This application visualizes the **Black Widow Optimization Algorithm** solving mathematical optimization benchmarks.
Watch the spiders (population) navigate the 3D terrain to find the global minimum!
""")

# Sidebar settings
st.sidebar.header("⚙️ Algorithm Settings")
selected_func_name = st.sidebar.selectbox("Benchmark Function", ["Sphere", "Rastrigin", "Rosenbrock"])
pop_size = st.sidebar.slider("Population Size", min_value=10, max_value=200, value=30, step=10)
max_iter = st.sidebar.slider("Max Iterations", min_value=10, max_value=300, value=50, step=10)

st.sidebar.subheader("BWO Parameters")
procreating_rate = st.sidebar.slider("Procreating Rate (PR)", 0.1, 1.0, 0.6, 0.05)
cannibalism_rate = st.sidebar.slider("Cannibalism Rate (CR)", 0.1, 0.9, 0.44, 0.05)
mutation_rate = st.sidebar.slider("Mutation Rate (MR)", 0.01, 0.9, 0.4, 0.05)

# Session State for optimization
if 'bwo' not in st.session_state:
    st.session_state.bwo = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'history_best' not in st.session_state:
    st.session_state.history_best = []
    
# Layout definition
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🕸️ Optimization Terrain")
    plot_placeholder = st.empty()

with col2:
    st.subheader("Convergence Curve")
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    st.markdown("### Controls")
    
    # Buttons
    cols = st.columns(3)
    start_btn = cols[0].button("▶️ Start / Reset")
    pause_btn = cols[1].button("⏸️ Pause")
    step_btn = cols[2].button("⏭️ Step")

# Logic
func = BenchmarkFunctions.get_function(selected_func_name)
bounds = BenchmarkFunctions.get_bounds(selected_func_name)

# Precompute terrain for smooth rendering
@st.cache_data
def get_terrain_data(func_name, bz):
    f = BenchmarkFunctions.get_function(func_name)
    return BenchmarkFunctions.get_surface_data(f, bz, res=80)

X, Y, Z = get_terrain_data(selected_func_name, bounds)

def draw_plots(bwo_algo, history, X, Y, Z):
    state = bwo_algo.get_state()
    pop = state['pop']
    
    # 3D Surface Figure
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7)])
    pop_z = np.apply_along_axis(func, 1, pop)
    
    # Scatter points - the Spiders
    fig.add_trace(go.Scatter3d(
        x=pop[:, 0], y=pop[:, 1], z=pop_z, mode='markers',
        marker=dict(size=5, color='red', symbol='circle',
                    line=dict(color='black', width=1)),
        name="Spiders"
    ))
    
    # Best solution marked
    fig.add_trace(go.Scatter3d(
        x=[state['best_solution'][0]], 
        y=[state['best_solution'][1]], 
        z=[state['best_fitness']],
        mode='markers',
        marker=dict(size=8, color='gold', symbol='diamond', line=dict(color='black', width=2)),
        name="Global Best"
    ))
    
    fig.update_layout(
        title=f"Generation: {state['curr_iter']} / {bwo_algo.max_iter}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Fitness',
            xaxis=dict(range=[bounds[0], bounds[1]]),
            yaxis=dict(range=[bounds[0], bounds[1]])
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    
    plot_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Convergence Curve
    if len(history) > 0:
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=history, mode='lines+markers', name='Best Fitness', line=dict(color='blue')))
        fig_conv.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Best Fitness",
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        chart_placeholder.plotly_chart(fig_conv, use_container_width=True)
        
    metrics_placeholder.markdown(f"""
    **Current Status:**
    - **Iteration**: {state['curr_iter']}
    - **Best Fitness**: {state['best_fitness']:.6f}
    - **Best Position**: `[{state['best_solution'][0]:.4f}, {state['best_solution'][1]:.4f}]`
    """)

# Button Handlers
if start_btn:
    st.session_state.bwo = BlackWidowOptimization(
        func=func, bounds=bounds, dim=2, pop_size=pop_size, max_iter=max_iter,
        procreating_rate=procreating_rate, cannibalism_rate=cannibalism_rate, mutation_rate=mutation_rate
    )
    st.session_state.history_best = [st.session_state.bwo.best_fitness]
    st.session_state.is_running = True
    st.rerun()

if pause_btn:
    st.session_state.is_running = False

if step_btn and st.session_state.bwo:
    st.session_state.is_running = False
    if st.session_state.bwo.curr_iter < max_iter:
        st.session_state.bwo.step()
        st.session_state.history_best.append(st.session_state.bwo.best_fitness)

# Initial Draw
if st.session_state.bwo is None:
    # Initialize a dummy to show start terrain
    bwo_dummy = BlackWidowOptimization(
        func=func, bounds=bounds, dim=2, pop_size=pop_size, max_iter=max_iter
    )
    draw_plots(bwo_dummy, [bwo_dummy.best_fitness], X, Y, Z)

# Animation Loop
if st.session_state.is_running and st.session_state.bwo:
    if st.session_state.bwo.curr_iter < max_iter:
        draw_plots(st.session_state.bwo, st.session_state.history_best, X, Y, Z)
        
        # Advance via step
        has_next = st.session_state.bwo.step()
        if not has_next:
            st.session_state.is_running = False
            st.rerun()
            
        st.session_state.history_best.append(st.session_state.bwo.best_fitness)
        time.sleep(0.05) # Sleep to allow UI rendering
        st.rerun() # Rerun to update state properly

# Draw final state if finished naturally or paused
if st.session_state.bwo and not st.session_state.is_running:
    draw_plots(st.session_state.bwo, st.session_state.history_best, X, Y, Z)

st.sidebar.markdown("---")
if st.session_state.bwo and st.session_state.bwo.curr_iter >= max_iter:
    st.sidebar.success("Optimization Complete! 🎉")
    result_df = pd.DataFrame({"Iteration": range(len(st.session_state.history_best)), "Best Fitness": st.session_state.history_best})
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download Results CSV", data=csv, file_name="bwo_results.csv", mime="text/csv")
