import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantum Playground", layout="wide")
st.title("Interactive Quantum Playground")

# Define basic gates
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
I = np.eye(2, dtype=complex)
CX = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,0,1],
               [0,0,1,0]], dtype=complex)

# Initialize 2-qubit state |00>
state = np.zeros(4, dtype=complex)
state[0] = 1.0

# Sidebar for gate selection
st.sidebar.markdown("### Apply Gates")
gate_options = ["None", "X", "Y", "Z", "H", "CX"]
gate1 = st.sidebar.selectbox("Gate on Qubit 0", gate_options)
gate2 = st.sidebar.selectbox("Gate on Qubit 1", gate_options)

def apply_gate(state, gate_name, qubit):
    if gate_name == "X":
        gate = X
    elif gate_name == "Y":
        gate = Y
    elif gate_name == "Z":
        gate = Z
    elif gate_name == "H":
        gate = H
    else:
        return state
    # Apply gate to the correct qubit
    if qubit == 0:
        gate_full = np.kron(gate, I)
    else:
        gate_full = np.kron(I, gate)
    return gate_full @ state

# Apply single-qubit gates
state = apply_gate(state, gate1, 0)
state = apply_gate(state, gate2, 1)

# Apply CNOT if selected
if gate1 == "CX":
    state = CX @ state
if gate2 == "CX":
    state = CX @ state

# Display state vector
st.subheader("Quantum State Vector |ψ⟩")
st.text(np.round(state,3))

# Measurement probabilities
st.subheader("Measurement Probabilities")
probs = np.abs(state)**2
for i, p in enumerate(probs):
    st.write(f"|{i:02b}⟩ : {p:.3f}")

# Bloch sphere (2D projection)
st.subheader("Bloch Sphere (2D projection)")
fig, ax = plt.subplots()
x = [np.real(state[0]), np.real(state[1])]
y = [np.imag(state[0]), np.imag(state[1])]
ax.quiver(0,0,x[0],y[0], angles='xy', scale_units='xy', scale=1, color='cyan', label='Qubit 0')
ax.quiver(0,0,x[1],y[1], angles='xy', scale_units='xy', scale=1, color='magenta', label='Qubit 1')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_xlabel('Real')
ax.set_ylabel('Imag')
ax.set_title('2D Bloch Sphere Projection')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Teleportation Demo
st.sidebar.markdown("### Teleportation Demo")
if st.sidebar.button("Run Teleportation"):
    # Simple 2-qubit entanglement demonstration
    teleport_state = np.zeros(4, dtype=complex)
    teleport_state[0] = 1/np.sqrt(2)
    teleport_state[3] = 1/np.sqrt(2)
    st.success("Teleportation-like entangled state created!")
    st.write(np.round(teleport_state,3))
