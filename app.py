import streamlit as st
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Pure Python Quantum Playground", layout="wide")
st.title("🚀 Pure Python Quantum Computing Playground")

# Sidebar options
st.sidebar.header("Quantum Circuit Options")
algorithm = st.sidebar.selectbox(
    "Select Quantum Algorithm",
    ["Quantum Teleportation", "Superdense Coding", "Custom Circuit"]
)
qubits = st.sidebar.slider("Number of Qubits", 1, 2, 1)

# Basic quantum gates in matrix form
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

# Initialize state |0>
def init_state(n):
    state = np.zeros((2**n,), dtype=complex)
    state[0] = 1
    return state

# Apply single-qubit gate
def apply_gate(state, gate, qubit, n):
    full_gate = 1
    for i in range(n):
        full_gate = np.kron(full_gate, gate if i == qubit else np.eye(2))
    return full_gate @ state

# Bloch coordinates from single qubit state
def bloch_coords_single(state):
    alpha, beta = state[0], state[1]
    x = 2 * (alpha.conj()*beta).real
    y = 2 * (alpha.conj()*beta).imag
    z = abs(alpha)**2 - abs(beta)**2
    return x, y, z

# Animate Bloch sphere
def animate_bloch(states):
    frames = []
    for state in states:
        x, y, z = bloch_coords_single(state)
        frames.append(go.Frame(data=[go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', 
                                                  marker=dict(size=6, color='red'))]))
    # Sphere wireframe
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    
    fig = go.Figure(
        data=[go.Surface(x=xs, y=ys, z=zs, opacity=0.1, colorscale='Blues', showscale=False)],
        layout=go.Layout(
            scene=dict(xaxis=dict(showgrid=False, visible=False),
                       yaxis=dict(showgrid=False, visible=False),
                       zaxis=dict(showgrid=False, visible=False)),
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play", method="animate",
                                            args=[None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}]),
                                       dict(label="Pause", method="animate",
                                            args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
            margin=dict(l=0,r=0,b=0,t=0)
        ),
        frames=frames
    )
    return fig

# Simulate circuit
state = init_state(qubits)
states_list = [state.copy()]

if algorithm == "Custom Circuit":
    if st.button("Apply Hadamard"):
        state = apply_gate(state, H, 0, qubits)
        states_list.append(state.copy())
    if st.button("Apply X"):
        state = apply_gate(state, X, 0, qubits)
        states_list.append(state.copy())
elif algorithm == "Quantum Teleportation":
    if qubits != 2:
        st.warning("Quantum teleportation requires 2 qubits")
    else:
        # Simplified simulation of teleportation
        state = apply_gate(state, H, 0, qubits)
        states_list.append(state.copy())
        state = apply_gate(state, X, 1, qubits)
        states_list.append(state.copy())
elif algorithm == "Superdense Coding":
    if qubits != 2:
        st.warning("Superdense coding requires 2 qubits")
    else:
        state = apply_gate(state, H, 0, qubits)
        states_list.append(state.copy())
        state = apply_gate(state, Z, 0, qubits)
        states_list.append(state.copy())

# Display Bloch sphere animation
fig = animate_bloch(states_list)
st.plotly_chart(fig)

# Download final Bloch sphere frame as PNG
buffer = BytesIO()
fig.write_image(buffer, format='png')
st.download_button(
    label="Download Bloch Sphere Image",
    data=buffer,
    file_name="bloch_final.png",
    mime="image/png"
)
