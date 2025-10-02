# Dynamic Qiskit installation if missing
import sys
import subprocess

try:
    import qiskit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qiskit"])

import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# Streamlit config and styling
st.set_page_config(page_title="Quantum Playground", layout="wide")
st.markdown("<h1 style='text-align:center; color:#f0f0f0;'>🪐 Interactive Quantum Playground</h1>", unsafe_allow_html=True)
st.markdown("<style>body {background-color:#1e1e1e; color:#f0f0f0;}</style>", unsafe_allow_html=True)

# Initialize 2-qubit quantum circuit
num_qubits = 2
qc = QuantumCircuit(num_qubits)

# Sidebar: Quantum Gate Application
st.sidebar.markdown("### Apply Quantum Gates")
gate_options = ["None", "X", "Y", "Z", "H", "CX"]

for i in range(num_qubits):
    gate = st.sidebar.selectbox(f"Gate on Qubit {i}", gate_options, key=f"gate{i}")
    target = None
    if gate == "CX":
        target = st.sidebar.selectbox(f"Target qubit for CX from qubit {i}", [q for q in range(num_qubits) if q != i], key=f"target{i}")
    if gate != "None":
        if gate == "X": qc.x(i)
        elif gate == "Y": qc.y(i)
        elif gate == "Z": qc.z(i)
        elif gate == "H": qc.h(i)
        elif gate == "CX" and target is not None: qc.cx(i, target)

# Teleportation Demo
st.sidebar.markdown("### Quantum Teleportation Demo")
if st.sidebar.button("Run Teleportation"):
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier()
    qc.measure_all()
    st.success("Teleportation circuit applied with 3 qubits!")

# Display Quantum Circuit
st.subheader("Quantum Circuit")
st.text(qc.draw(output='text'))

# Quantum State Visualization
state = Statevector.from_instruction(qc)
st.subheader("Bloch Sphere Visualization")
fig_bloch = plot_bloch_multivector(state)
st.pyplot(fig_bloch)

# Measurement Probabilities
st.subheader("Measurement Probabilities")
qc_measure = qc.copy()
qc_measure.measure_all()
backend = Aer.get_backend('qasm_simulator')
result = execute(qc_measure, backend, shots=1024).result()
counts = result.get_counts()
fig_hist = plot_histogram(counts)
st.pyplot(fig_hist)

# Notes
st.subheader("🔹 Notes")
st.markdown("""
- Use the sidebar to apply quantum gates.
- Bloch Sphere shows qubit superposition states.
- Measurement histogram shows probabilities after measuring qubits.
- Teleportation demo uses a simple 3-qubit circuit.
- Try different combinations of gates and observe results!
""")
