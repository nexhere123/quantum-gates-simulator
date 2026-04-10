import streamlit as st
import numpy as np
import plotly.graph_objects as go

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Quantum Gates Simulator",
    page_icon="⚛️",
    layout="wide"
)

# =========================
# Quantum Gates
# =========================
def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def hadamard():
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# =========================
# Utility functions
# =========================
def normalize(a, b):
    norm = np.sqrt(abs(a)**2 + abs(b)**2)
    if norm == 0:
        return 1+0j, 0+0j
    return a/norm, b/norm

def bloch_coords(a, b):
    x = 2*np.real(np.conj(a)*b)
    y = 2*np.imag(np.conj(a)*b)
    z = abs(a)**2 - abs(b)**2
    return x, y, z

def plot_bloch(x, y, z):
    fig = go.Figure()

    # Sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig.add_surface(x=xs, y=ys, z=zs, opacity=0.2)

    # State vector
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines+markers'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    return fig

# =========================
# HEADER
# =========================
st.title("⚛️ Quantum Gates Simulator")

st.write("Visualize how quantum gates affect qubit states.")

# =========================
# CONTROLS (NOW IN MAIN SCREEN)
# =========================
st.subheader("🎛️ Controls")

gate_type = st.selectbox(
    "Select Quantum Gate:",
    ["Pauli-X", "Pauli-Y", "Pauli-Z", "Hadamard"]
)

col1, col2 = st.columns(2)

with col1:
    st.write("### Initial State")

    alpha_real = st.slider("α real", -2.0, 2.0, 1.0)
    alpha_imag = st.slider("α imag", -2.0, 2.0, 0.0)
    beta_real = st.slider("β real", -2.0, 2.0, 0.0)
    beta_imag = st.slider("β imag", -2.0, 2.0, 0.0)

# =========================
# STATE CALCULATION
# =========================
a, b = normalize(alpha_real + 1j*alpha_imag,
                 beta_real + 1j*beta_imag)

state = np.array([a, b])

# Select gate
if gate_type == "Pauli-X":
    U = pauli_x()
elif gate_type == "Pauli-Y":
    U = pauli_y()
elif gate_type == "Pauli-Z":
    U = pauli_z()
elif gate_type == "Hadamard":
    U = hadamard()

new_state = U @ state

# =========================
# OUTPUT
# =========================
with col2:
    st.write("### After Gate")

    st.write(f"|ψ⟩ = ({new_state[0]:.3f})|0⟩ + ({new_state[1]:.3f})|1⟩")

    p0 = abs(new_state[0])**2
    p1 = abs(new_state[1])**2

    st.write(f"P(|0⟩) = {p0:.3f}")
    st.write(f"P(|1⟩) = {p1:.3f}")

# =========================
# BLOCH SPHERE
# =========================
x, y, z = bloch_coords(new_state[0], new_state[1])
st.plotly_chart(plot_bloch(x, y, z), use_container_width=True)

# =========================
# CREDITS (VISIBLE NOW)
# =========================
st.markdown("---")
if st.button("Show Credits"):
    st.write("""
    Developed by:
    - AJ Kaarthick  
    - Prince Deshmukh  
    - Priyanshu  
    - Raj Sharma  
    - Dhyey Raiyani  
    - Yash Sharma  
    - Faisal Firoz  
    - Hemchand Mouli  
    """)