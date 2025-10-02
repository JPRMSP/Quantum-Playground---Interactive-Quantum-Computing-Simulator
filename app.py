import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------
# ⚛️ Quantum Playground – Interactive Quantum Simulator
# ---------------------------------------------------------
# Built from scratch using only NumPy and Matplotlib.
# Features:
#  - Qubit simulator + 3D Bloch sphere visualization
#  - Quantum teleportation simulator
#  - Grover’s search demo
#  - Theory tabs for learning
# ---------------------------------------------------------

# Streamlit Page Config
st.set_page_config(page_title="Quantum Playground", page_icon="⚛️", layout="wide")

# Apply dark mode background
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #f5f5f5; }
    .stMarkdown, .stRadio, .stSelectbox, .stMultiSelect, .stButton, .stSlider { color: #f5f5f5; }
    h1, h2, h3, h4 { color: #00e6a8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚛️ Quantum Playground – Interactive Quantum Computing Simulator")
st.markdown("Explore quantum computing concepts visually and interactively.")

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def ket0():
    return np.array([[1], [0]], dtype=complex)

def ket1():
    return np.array([[0], [1]], dtype=complex)

# Basic gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def tensor(*args):
    """Kronecker product for multi-qubit states"""
    result = args[0]
    for a in args[1:]:
        result = np.kron(result, a)
    return result

def measure(state):
    """Simulate quantum measurement"""
    probs = np.abs(state.flatten())**2
    return np.random.choice(len(state), p=probs)

# ---------------------------------------------------------
# Bloch Sphere Visualization
# ---------------------------------------------------------

def plot_bloch(state):
    """3D Bloch sphere for a single qubit state"""
    # Extract alpha, beta
    alpha = state[0, 0]
    beta = state[1, 0]

    # Convert to Bloch sphere coordinates
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")

    # Draw Bloch sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='cyan', alpha=0.1, edgecolor='gray')

    # Draw state vector
    ax.quiver(0, 0, 0, x, y, z, color='red', linewidth=3, arrow_length_ratio=0.15)

    # Axes styling
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=25, azim=45)
    return fig

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tabs = st.tabs(["🧬 Qubit Simulator", "📡 Quantum Teleportation", "🔍 Grover’s Search", "📚 Theory"])

# ---------------------------------------------------------
# Tab 1: Qubit Simulator
# ---------------------------------------------------------
with tabs[0]:
    st.header("🧬 Qubit State Simulator + 3D Bloch Sphere")

    # Choose initial state
    qubit_choice = st.radio("Choose initial qubit state:", ["|0⟩", "|1⟩"])
    state = ket0() if qubit_choice == "|0⟩" else ket1()

    # Apply gates
    gate_choice = st.multiselect("Apply gates (in order):", ["X", "Y", "Z", "H"])
    for g in gate_choice:
        if g == "X":
            state = X @ state
        elif g == "Y":
            state = Y @ state
        elif g == "Z":
            state = Z @ state
        elif g == "H":
            state = H @ state

    # Show final state
    st.subheader("📊 Final State Vector:")
    st.code(state)

    # Measurement simulation
    meas = measure(state)
    st.success(f"📏 Measurement result: |{meas}⟩")

    # Plot Bloch sphere
    st.subheader("🌐 3D Bloch Sphere Visualization")
    st.pyplot(plot_bloch(state))

# ---------------------------------------------------------
# Tab 2: Quantum Teleportation
# ---------------------------------------------------------
with tabs[1]:
    st.header("📡 Quantum Teleportation Simulator")
    st.markdown("This simulation demonstrates how an unknown quantum state can be transferred from Alice to Bob using entanglement and classical communication.")

    if st.button("Run Teleportation Simulation"):
        # Prepare |ψ> state (unknown state)
        psi = (1/np.sqrt(2)) * (ket0() + ket1())
        alice = ket0()
        bob = ket0()

        # Full 3-qubit state: |ψ>|0>|0>
        state_total = tensor(psi, alice, bob)

        # Create Bell pair between qubits 2 and 3
        state_total = tensor(I, H, I) @ state_total

        # CNOT(2->3)
        CNOT_23 = np.kron(I, np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,0,1],
                                       [0,0,1,0]], dtype=complex))
        state_total = CNOT_23 @ state_total

        # Alice performs CNOT(1->2) and H on qubit 1
        CNOT_12 = np.kron(np.array([[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,0,1],
                                    [0,0,1,0]], dtype=complex), I)
        state_total = CNOT_12 @ state_total
        state_total = tensor(H, I, I) @ state_total

        st.success("✅ Teleportation complete! The unknown state has been transferred to Bob.")
        st.code(state_total)

# ---------------------------------------------------------
# Tab 3: Grover’s Search
# ---------------------------------------------------------
with tabs[2]:
    st.header("🔍 Grover’s Search Algorithm Demo")
    st.markdown("Grover's algorithm searches an unsorted database faster than classical algorithms.")

    oracle_target = st.selectbox("Choose the secret state to search for:", ["|00⟩", "|01⟩", "|10⟩", "|11⟩"])
    oracle_index = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"].index(oracle_target)

    if st.button("Run Grover Search"):
        psi = (1/2) * np.array([[1], [1], [1], [1]], dtype=complex)

        # Oracle
        O = np.eye(4, dtype=complex)
        O[oracle_index, oracle_index] = -1

        # Diffusion
        D = 2 * (1/4 * np.ones((4, 4), dtype=complex)) - np.eye(4)

        psi = D @ (O @ psi)

        st.subheader("📈 Final state vector:")
        st.code(psi)

        probs = np.abs(psi.flatten())**2
        found_state = np.argmax(probs)
        st.success(f"✅ Grover found: {['|00⟩', '|01⟩', '|10⟩', '|11⟩'][found_state]} with probability {probs[found_state]:.2f}")

# ---------------------------------------------------------
# Tab 4: Theory
# ---------------------------------------------------------
with tabs[3]:
    st.header("📚 Quantum Computing Concepts – Quick Theory")
    st.markdown("""
    ### 🧬 Qubits & Superposition
    A qubit is the fundamental unit of quantum information. It can exist as |0⟩, |1⟩, or any linear combination:  
    $$|ψ⟩ = α|0⟩ + β|1⟩$$

    ### 📡 Quantum Teleportation
    Quantum teleportation uses entanglement and classical communication to transfer an unknown quantum state without physically moving the particle.

    ### 🔍 Grover’s Algorithm
    Grover’s algorithm searches an unsorted database of N items in roughly √N steps — a quadratic speed-up over classical search.
    """)

st.markdown("---")
st.caption("⚛️ Built with ❤️ using NumPy, Matplotlib, and Streamlit – 2025")
