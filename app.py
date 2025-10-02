import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
import plotly.graph_objects as go
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Quantum Teleportation Animation", layout="wide")
st.title("🚀 Multi-Qubit Animated Quantum Playground")

# Sidebar
st.sidebar.header("Quantum Circuit Options")
algorithm = st.sidebar.selectbox(
    "Select Quantum Algorithm",
    ["Quantum Teleportation", "Superdense Coding", "Quantum Fourier Transform", "Custom Circuit"]
)
qubits = st.sidebar.slider("Number of Qubits", 1, 3, 2)

# Bloch coordinates calculation
def bloch_coords(statevector):
    coords = []
    for amp in statevector.data:
        theta = 2 * np.arccos(np.abs(amp))
        phi = np.angle(amp)
        coords.append((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
    return coords

# Function to animate multiple qubits
def animate_multi_bloch(circuit):
    frames = []
    current_circuit = QuantumCircuit(circuit.num_qubits)
    
    for instr, qargs, cargs in circuit.data:
        current_circuit.append(instr, qargs, cargs)
        state = Statevector.from_instruction(current_circuit)
        x, y, z = zip(*bloch_coords(state))
        traces = []
        for xi, yi, zi in zip(x, y, z):
            traces.append(go.Scatter3d(x=[xi], y=[yi], z=[zi], mode='markers',
                                       marker=dict(size=6, color='red')))
        frames.append(go.Frame(data=traces))
    
    # Sphere wireframe
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    
    fig = go.Figure(
        data=[go.Surface(x=xs, y=ys, z=zs, opacity=0.1, colorscale='Blues', showscale=False)],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(showgrid=False, visible=False),
                zaxis=dict(showgrid=False, visible=False)
            ),
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

# Circuit definition
if algorithm == "Quantum Teleportation":
    qc = QuantumCircuit(3)
    st.subheader("Quantum Teleportation Circuit")
    qc.h(1)
    qc.cx(1,2)
    qc.cx(0,1)
    qc.h(0)
elif algorithm == "Superdense Coding":
    qc = QuantumCircuit(2)
    st.subheader("Superdense Coding Circuit")
    qc.h(0)
    qc.cx(0,1)
    qc.x(0)
    qc.z(0)
    qc.cx(0,1)
    qc.h(0)
elif algorithm == "Quantum Fourier Transform":
    qc = QuantumCircuit(qubits)
    st.subheader(f"Quantum Fourier Transform Circuit ({qubits} qubits)")
    for j in range(qubits):
        qc.h(j)
        for k in range(j+1, qubits):
            qc.cp(np.pi/2**(k-j), k, j)
else:
    qc = QuantumCircuit(qubits)
    st.subheader("Custom Circuit")
    if st.button("Apply Hadamard to all qubits"):
        for i in range(qubits):
            qc.h(i)
    if st.button("Apply CNOT (0->1)"):
        if qubits > 1:
            qc.cx(0,1)

# Display circuit diagram
st.pyplot(qc.draw('mpl'))

# Display animated Bloch spheres
st.plotly_chart(animate_multi_bloch(qc))

# Download final frame
final_state = Statevector.from_instruction(qc)
buffer = BytesIO()
fig = animate_multi_bloch(qc)
fig.write_image(buffer, format='png')
st.download_button(
    label="Download Final Bloch Sphere Image",
    data=buffer,
    file_name="multi_bloch_final.png",
    mime="image/png"
)
