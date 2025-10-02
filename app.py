import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Quantum Playground", layout="wide")
st.title("🪐 Interactive Quantum Playground")

# Sidebar for qubit selection
num_qubits = st.sidebar.slider("Select number of qubits", 1, 3, 2)
qc = QuantumCircuit(num_qubits)

st.sidebar.markdown("### Apply Quantum Gates")

# Function to apply selected gate
def apply_gate(gate, qubit, target=None):
    if gate == "X":
        qc.x(qubit)
    elif gate == "Y":
        qc.y(qubit)
    elif gate == "Z":
        qc.z(qubit)
    elif gate == "H":
        qc.h(qubit)
    elif gate == "CX" and target is not None:
        qc.cx(qubit, target)

# Gate application interface
for i in range(num_qubits):
    st.sidebar.markdown(f"**Qubit {i}**")
    gate = st.sidebar.selectbox(f"Gate for qubit {i}", ["None", "X", "Y", "Z", "H", "CX"], key=f"gate{i}")
    if gate != "None":
        target = None
        if gate == "CX":
            possible_targets = [q for q in range(num_qubits) if q != i]
            target = st.sidebar.selectbox(f"Target qubit for CX from qubit {i}", possible_targets, key=f"target{i}")
        apply_gate(gate, i, target)

st.subheader("Quantum Circuit")
st.text(qc.draw(output='text'))

# Simulation
state = Statevector.from_instruction(qc)

# Bloch Sphere
st.subheader("Bloch Sphere Visualization")
fig_bloch = plot_bloch_multivector(state)
st.pyplot(fig_bloch)

# Measurement
st.subheader("Measurement Probabilities")
qc_measure = qc.copy()
qc_measure.measure_all()
backend = Aer.get_backend('qasm_simulator')
result = execute(qc_measure, backend, shots=1024).result()
counts = result.get_counts()

fig_hist = plot_histogram(counts)
st.pyplot(fig_hist)

st.subheader("🔹 Notes")
st.markdown("""
- Apply gates using the sidebar.
- See how quantum states evolve on the Bloch sphere.
- Measure qubits to see probabilities.
- Try creating entanglement, superposition, or teleportation circuits!
""")
