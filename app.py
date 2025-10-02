import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import numpy as np
import io
import imageio

st.set_page_config(page_title="Quantum IDE Ultra-Pro Highlight GIF", layout="wide", page_icon="⚛️")
st.markdown("<h1 style='text-align:center; color:#f0f0f0;'>Quantum IDE Ultra-Pro with Highlighted GIF</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#a0a0a0;'>Interactive 3-Qubit Simulator + Animated Timeline GIF with Step Highlight</h4>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

# Initialize session
if "qc" not in st.session_state:
    st.session_state.qc = QuantumCircuit(3)
if "timeline" not in st.session_state:
    st.session_state.timeline = {0: [], 1: [], 2: []}
if "step_index" not in st.session_state:
    st.session_state.step_index = 0

qc = st.session_state.qc
timeline = st.session_state.timeline
step_index = st.session_state.step_index

# Sidebar demo
st.sidebar.header("Demos / Reset")
demo_choice = st.sidebar.selectbox("Select Demo", ["None", "Teleportation", "Superdense Coding", "Reset Circuit"])

if demo_choice == "Reset Circuit":
    st.session_state.qc = QuantumCircuit(3)
    st.session_state.timeline = {0: [], 1: [], 2: []}
    st.session_state.step_index = 0
    qc = st.session_state.qc
    timeline = st.session_state.timeline
    step_index = 0
    st.sidebar.success("Circuit reset!")
elif demo_choice == "Teleportation":
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1,2)
    qc.cx(0,1)
    qc.h(0)
    st.session_state.qc = qc
    st.session_state.timeline = {0:["H","CX"],1:["H","CX"],2:["CX"]}
    st.sidebar.success("Teleportation demo loaded!")
elif demo_choice == "Superdense Coding":
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0,1)
    st.session_state.qc = qc
    st.session_state.timeline = {0:["H","CX"],1:["CX"],2:[]}
    st.sidebar.success("Superdense Coding demo loaded!")

# Apply gates interactively
st.subheader("Apply Gates to Qubits (Drag-and-Drop)")
gate_options = ["H", "X", "Y", "Z", "CNOT"]
cols = st.columns(qc.num_qubits)
for i, col in enumerate(cols):
    col.markdown(f"**Qubit {i}**")
    for gate in gate_options:
        if col.button(f"{gate} Q{i}"):
            try:
                if gate == "H": qc.h(i)
                elif gate == "X": qc.x(i)
                elif gate == "Y": qc.y(i)
                elif gate == "Z": qc.z(i)
                elif gate == "CNOT":
                    target = st.number_input(f"Target Qubit for CNOT from Q{i}", 0, qc.num_qubits-1, 0)
                    if target != i:
                        qc.cx(i, target)
                        timeline[target].append(f"CX from Q{i}")
                    else:
                        st.error("Control and Target must be different!")
                timeline[i].append(gate)
                st.success(f"{gate} applied to Qubit {i}")
            except Exception as e:
                st.error(f"Error: {e}")

# Animated timeline step
st.subheader("Gate Timeline Animation")
max_steps = max([len(t) for t in timeline.values()])
step_index = st.slider("Animation Step", 0, max_steps, step_index)
st.session_state.step_index = step_index

# Build circuit up to current step
animated_qc = QuantumCircuit(qc.num_qubits)
for q, gates in timeline.items():
    for g in gates[:step_index]:
        if g == "H": animated_qc.h(q)
        elif g == "X": animated_qc.x(q)
        elif g == "Y": animated_qc.y(q)
        elif g == "Z": animated_qc.z(q)
        elif "CX" in g:
            target = int(g.split("Q")[1])
            animated_qc.cx(q, target)

# Display animated circuit
st.write(animated_qc.draw(output='mpl', fold=-1))

# Statevector simulation
sim = Aer.get_backend('statevector_simulator')
state = execute(animated_qc, sim).result().get_statevector()

# Live Qubit State Inspection
st.subheader("Live Qubit State Inspection")
selected_qubit = st.selectbox("Select Qubit to Inspect", list(range(animated_qc.num_qubits)))
alpha = state[selected_qubit] if selected_qubit < len(state) else 0
beta = state[(selected_qubit + 1) % len(state)] if selected_qubit+1 < len(state) else 0
st.write(f"**Qubit {selected_qubit} state amplitudes:**")
st.write(f"α = {np.round(np.real(alpha),4)} + {np.round(np.imag(alpha),4)}i")
st.write(f"β = {np.round(np.real(beta),4)} + {np.round(np.imag(beta),4)}i")

# Bloch Sphere Visualization
st.subheader("Bloch Sphere Visualization")
fig_bloch = plot_bloch_multivector(state)
st.pyplot(fig_bloch)

# Measurement Histogram
st.subheader("Measurement Histogram")
qc_measure = animated_qc.copy()
qc_measure.measure_all()
sim2 = Aer.get_backend('qasm_simulator')
counts = execute(qc_measure, sim2, shots=1024).result().get_counts()
fig_hist = plot_histogram(counts)
st.pyplot(fig_hist)

# GIF Export with Highlighted Current Gate
st.subheader("Export Animated Timeline GIF with Highlighted Current Gate")
if st.button("Generate Highlight GIF"):
    images = []
    colors = ['red','green','cyan']  # colors per qubit
    for step in range(1, max_steps+1):
        temp_qc = QuantumCircuit(qc.num_qubits)
        applied_gates = []
        last_gate_idx = None
        gate_counter = 0
        # apply gates stepwise
        for q, gates in timeline.items():
            for g in gates[:step]:
                if g == "H": temp_qc.h(q)
                elif g == "X": temp_qc.x(q)
                elif g == "Y": temp_qc.y(q)
                elif g == "Z": temp_qc.z(q)
                elif "CX" in g:
                    target = int(g.split("Q")[1])
                    temp_qc.cx(q,target)
                applied_gates.append((gate_counter, f"Q{q}: {g}", q))
                last_gate_idx = gate_counter
                gate_counter +=1
        temp_state = execute(temp_qc, sim).result().get_statevector()
        fig = plot_bloch_multivector(temp_state)
        # Multi-color labels with highlight
        for idx, label, qubit in applied_gates:
            color = 'yellow' if idx == last_gate_idx else colors[qubit % len(colors)]
            plt.text(0, -1.2-0.1*idx, label, fontsize=10, color=color)
        # Timeline bar
        plt.axhline(-1.8, xmin=0, xmax=1, color='white', linewidth=3)
        plt.text(0.02, -1.85, f"Step {step}/{max_steps}", color='yellow', fontsize=10)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        images.append(imageio.imread(buf))
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, images, duration=1)
    gif_buf.seek(0)
    st.download_button("Download Highlighted Timeline GIF", data=gif_buf, file_name="quantum_timeline_highlight.gif", mime="image/gif")
    st.success("Highlighted GIF generated successfully!")

st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)
st.write("Developed by: **AISNOTA | Quantum Computing Enthusiast**")
