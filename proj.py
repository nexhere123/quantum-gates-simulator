import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# =========================
# Math & utility functions
# =========================
def normalize_state(*amps):
    """Normalize 1-qubit (a,b) or N-qubit statevector (single array)."""
    if len(amps) == 1:
        vec = np.asarray(amps[0], dtype=complex)
        nrm = np.linalg.norm(vec)
        if nrm == 0:
            out = np.zeros_like(vec, dtype=complex)
            out[0] = 1 + 0j
            return out
        return vec / nrm
    a, b = amps
    nrm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
    if nrm == 0:
        return 1 + 0j, 0 + 0j
    return a / nrm, b / nrm


def pretty_complex(z, nd=3):
    r = np.round(np.real(z), nd)
    i = np.round(np.imag(z), nd)
    if np.isclose(i, 0.0):  return f"{r:.{nd}f}"
    if np.isclose(r, 0.0):  return f"{i:.{nd}f}i"
    sign = "+" if i >= 0 else "-"
    return f"{r:.{nd}f} {sign} {abs(i):.{nd}f}i"


def state_to_latex_1q(a, b, nd=3):
    a_str = pretty_complex(a, nd)
    b_str = pretty_complex(b, nd)
    return rf"|\psi\rangle \;=\; ({a_str})\,|0\rangle \;+\; ({b_str})\,|1\rangle"


def prob_0_1(a, b, nd=3):
    p0 = np.round(np.abs(a) ** 2, nd)
    p1 = np.round(np.abs(b) ** 2, nd)
    return float(p0), float(p1)


def bloch_coords_from_state(a, b):
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return float(x), float(y), float(z)


def density_from_state(psi):
    psi = psi.reshape(-1, 1)
    return psi @ np.conj(psi.T)


def partial_trace_2q(rho, keep="A"):
    """
    Partial trace of a 2-qubit density matrix over B (keep="A") or over A (keep="B").
    Basis order: |00>, |01>, |10>, |11> with q0 as LEFT bit (most significant).
    """
    rho = rho.reshape(2, 2, 2, 2)  # (i, j, i', j')  == (q0, q1, q0', q1')
    if keep.upper() == "A":  # trace over q1  (j = j')
        return np.einsum('ijkj->ik', rho)
    else:  # trace over q0  (i = i')
        return np.einsum('ijil->jl', rho)


def bloch_from_density(rho1):
    """Given 2x2 density matrix, return Bloch vector via Tr(rho σ)."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    x = np.real(np.trace(rho1 @ sx))
    y = np.real(np.trace(rho1 @ sy))
    z = np.real(np.trace(rho1 @ sz))
    return float(x), float(y), float(z)


# =========================
# Gates & application
# =========================
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)


def apply_on_qubits(state, U, targets, n_qubits):
    """
    Apply a k-qubit gate U to given targets (indices) inside an n-qubit statevector.
    Qubit 0 is leftmost (most significant) in basis |q0 q1 ...>.
    """
    k = len(targets)
    assert U.shape == (2 ** k, 2 ** k)
    dims = [2] * n_qubits
    tensor = state.reshape(dims)

    all_axes = list(range(n_qubits))
    other_axes = [ax for ax in all_axes if ax not in targets]
    new_order = other_axes + targets
    inv_order = np.argsort(new_order)

    tensor_perm = np.transpose(tensor, axes=new_order)
    flat_other = 2 ** (n_qubits - k)
    flat_target = 2 ** k
    tensor_flat = tensor_perm.reshape(flat_other, flat_target)

    out_flat = tensor_flat @ U.T
    out_perm = out_flat.reshape(*(dims[ax] for ax in new_order))
    out = np.transpose(out_perm, axes=inv_order).reshape(2 ** n_qubits)
    return out


def twoq_unitary_for(control, target, gate2x2_on_target):
    """
    Build a 4x4 unitary acting on [control, target] (in that ORDER),
    for controlled-U: |0><0|⊗I + |1><1|⊗U(target).
    """
    return np.kron(P0, I) + np.kron(P1, gate2x2_on_target)


def CNOT(control, target):
    assert control != target
    return twoq_unitary_for(control, target, X), [control, target]


# Set page configuration
st.set_page_config(
    page_title="Quantum Gates Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header1{
        background: linear-gradient(180deg,rgb(0,159,255),rgb(30,47,75) );
        font-size: 75px;
        color: Violet;
        text-align: center;
        box-shadow: 5px 5px 10px hsl(0, 0%, 0%);
        border-radius: 10px;
        margin-left: 110px;
        min-width: 300px;
        text-shadow: 5px 5px 5px hsl(0, 0%, 0%);
    }
    .main-header {
        border-radius: 10px;
        align-items: center;
        margin-bottom: 5px;
        background: linear-gradient(180deg,rgb(14,17,23),rgb(38,39,48));
        font-size: 3.5rem;
        margin-left: 110px;
        min-width: 300px;
        border-width: 10px;
        border-color: rgb(132, 0, 255);
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 5px 5px 5px hsl(0, 0%, 0%);
    }
    .gate-info {
        background-color: #0F0F0F;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .qubit-state {
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        background-color: #0F0F0F;
        padding: 0.5rem;
        border-radius: 0.3rem;
        box-shadow: 5px 5px 10px hsl(0, 0%, 0%);
    }
</style>
""", unsafe_allow_html=True)


# Quantum gate definitions
class QuantumGates:
    @staticmethod
    def pauli_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)

    @staticmethod
    def pauli_y():
        return np.array([[0, -1j], [1j, 0]], dtype=complex)

    @staticmethod
    def pauli_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def hadamard():
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def identity():
        return np.array([[1, 0], [0, 1]], dtype=complex)


# New function to plot Bloch sphere (from cnot.py)
def plot_bloch(x, y, z, height=320):
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_surface(x=xs, y=ys, z=zs, opacity=0.15, showscale=False)
    axis_len = 1.1
    fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode="lines", name="X"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode="lines", name="Y"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode="lines", name="Z"))
    fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z],
                               mode="lines+markers", marker=dict(size=4), name="|ψ⟩"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], title="X"),
            yaxis=dict(range=[-1.2, 1.2], title="Y"),
            zaxis=dict(range=[-1.2, 1.2], title="Z"),
            aspectmode="cube",
        ),
        showlegend=False,
        height=height
    )
    return fig


def display_state_info(state, label="State"):
    """Display quantum state information for single qubit"""
    alpha, beta = state[0], state[1]

    st.write(f"**{label}:**")

    # Display state vector
    alpha_str = f"{alpha.real:.3f}" + (
        f" + {alpha.imag:.3f}i" if alpha.imag >= 0 else f" - {abs(alpha.imag):.3f}i") if alpha.imag != 0 else f"{alpha.real:.3f}"
    beta_str = f"{beta.real:.3f}" + (
        f" + {beta.imag:.3f}i" if beta.imag >= 0 else f" - {abs(beta.imag):.3f}i") if beta.imag != 0 else f"{beta.real:.3f}"

    st.markdown(f"""
    <div class="qubit-state">
    |ψ⟩ = ({alpha_str})|0⟩ + ({beta_str})|1⟩
    </div>
    """, unsafe_allow_html=True)

    # Display probabilities
    prob_0 = abs(alpha) ** 2
    prob_1 = abs(beta) ** 2

    st.write(f"- Probability of measuring |0⟩: {prob_0:.3f}")
    st.write(f"- Probability of measuring |1⟩: {prob_1:.3f}")


def main():
    # Main header

    if 'show_credits' not in st.session_state:
        st.session_state.show_credits = False

    if st.sidebar.button("Show Credits"):
        st.session_state.show_credits = not st.session_state.show_credits

    if st.session_state.show_credits:
        with st.sidebar.expander("Credits", expanded=False):
            st.markdown("""
            **Developed by:**
                    TEAM PIKACHU
            - NEEL BADLANI (25BRS1310)
            - HUMAIRA AISHA. U (25BRS1319)		
            - SHREYA SUMAN PATY (25BRS1407)
            - A GOPIKA SRI (25BAI1729)         
            - OVIYA PUGAZHENDI (25BAI1530)
            """)

    st.markdown('<h2 class="main-header">⚛️ Quantum Gates Simulator</h1>', unsafe_allow_html=True)

    st.markdown("""
    This simulator demonstrates how various quantum gates affect qubit states. 
    You can visualize the effects on the Bloch sphere and see the mathematical transformations.
    """)

    # Sidebar for gate selection and parameters
    st.sidebar.header("🎛️ Gate Controls")

    gate_type = st.sidebar.selectbox(
        "Select Quantum Gate:",
        ["Pauli-X", "Pauli-Y", "Pauli-Z", "Hadamard", "CNOT"]
    )

    # --- START CNOT GATE SECTION ---
    if gate_type == "CNOT":
        st.title("CNOT Gate Simulation")
        st.markdown(
            """
            The **Controlled-NOT (CNOT) gate** is a two-qubit quantum gate.  
            - It uses one qubit as the **control** and the other as the **target**.  
            - If the control qubit is in state $|1\\rangle$, the target qubit is flipped (X gate applied).  
            - If the control is $|0\\rangle$, the target remains unchanged.  

            In this simulator:  
            - You can choose an initial 2-qubit state from the menu.  
            - The control qubit is fixed to **q0** (the left qubit), and the target is **q1** (the right qubit).  
            - The basis order is $|q0 q1\\rangle$, with **q0 as the left bit**.  
            """
        )

        st.sidebar.subheader("CNOT Gate")

        preset2 = st.sidebar.selectbox(
            "Initial 2-qubit state",
            ["|00⟩", "|01⟩", "|10⟩", "|11⟩", "|Φ⁺⟩", "|Φ⁻⟩", "|Ψ⁺⟩", "|Ψ⁻⟩"],
            index=0
        )

        twoq_states = {
            "|00⟩": np.array([1, 0, 0, 0], dtype=complex),
            "|01⟩": np.array([0, 1, 0, 0], dtype=complex),
            "|10⟩": np.array([0, 0, 1, 0], dtype=complex),
            "|11⟩": np.array([0, 0, 0, 1], dtype=complex),
            "|Φ⁺⟩": (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=complex),
            "|Φ⁻⟩": (1 / np.sqrt(2)) * np.array([1, 0, 0, -1], dtype=complex),
            "|Ψ⁺⟩": (1 / np.sqrt(2)) * np.array([0, 1, 1, 0], dtype=complex),
            "|Ψ⁻⟩": (1 / np.sqrt(2)) * np.array([0, 1, -1, 0], dtype=complex),
        }
        psi0 = normalize_state(twoq_states[preset2])

        control = 0
        target = 1

        U2, targets = CNOT(control, target)
        psi1 = apply_on_qubits(psi0, U2, targets, n_qubits=2)
        psi1 = normalize_state(psi1)

        def reduced_bloch_pair(rho2, control_idx):
            rhoA = partial_trace_2q(rho2, keep="A")
            rhoB = partial_trace_2q(rho2, keep="B")
            if control_idx == 0:
                bC = bloch_from_density(rhoA)
                bT = bloch_from_density(rhoB)
            else:
                bC = bloch_from_density(rhoB)
                bT = bloch_from_density(rhoA)
            return bC, bT

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Before CNOT")
            p = np.abs(psi0) ** 2
            st.markdown(
                f"- **P(|00⟩)** `{p[0]:.3f}` • **P(|01⟩)** `{p[1]:.3f}` • "
                f"**P(|10⟩)** `{p[2]:.3f}` • **P(|11⟩)** `{p[3]:.3f}`"
            )

            rho0 = density_from_state(psi0)
            (xC0, yC0, zC0), (xT0, yT0, zT0) = reduced_bloch_pair(rho0, control)

            st.markdown("**Reduced Bloch Spheres**")
            bb1, bb2 = st.columns(2)
            with bb1:
                st.caption(f"Control qubit: q{control}")
                st.plotly_chart(plot_bloch(xC0, yC0, zC0, height=300), use_container_width=True,
                                key="bloch_before_control")
            with bb2:
                st.caption(f"Target qubit: q{target}")
                st.plotly_chart(plot_bloch(xT0, yT0, zT0, height=300), use_container_width=True,
                                key="bloch_before_target")

        with c2:
            st.subheader("After CNOT")
            p1 = np.abs(psi1) ** 2
            st.markdown(
                f"- **P(|00⟩)** `{p1[0]:.3f}` • **P(|01⟩)** `{p1[1]:.3f}` • "
                f"**P(|10⟩)** `{p1[2]:.3f}` • **P(|11⟩)** `{p1[3]:.3f}`"
            )

            rho1 = density_from_state(psi1)
            (xC1, yC1, zC1), (xT1, yT1, zT1) = reduced_bloch_pair(rho1, control)

            st.markdown("**Reduced Bloch Spheres**")
            ba1, ba2 = st.columns(2)
            with ba1:
                st.caption(f"Control qubit: q{control}")
                st.plotly_chart(plot_bloch(xC1, yC1, zC1, height=300), use_container_width=True,
                                key="bloch_after_control")
            with ba2:
                st.caption(f"Target qubit: q{target}")
                st.plotly_chart(plot_bloch(xT1, yT1, zT1, height=300), use_container_width=True,
                                key="bloch_after_target")

    # --- END CNOT GATE SECTION ---

    # --- START SINGLE QUBIT GATES SECTION ---
    else:
        # Initialize session state for qubit parameters
        if 'alpha_real' not in st.session_state:
            st.session_state.alpha_real = 1.0
        if 'alpha_imag' not in st.session_state:
            st.session_state.alpha_imag = 0.0
        if 'beta_real' not in st.session_state:
            st.session_state.beta_real = 0.0
        if 'beta_imag' not in st.session_state:
            st.session_state.beta_imag = 0.0

        st.sidebar.subheader("Initial Qubit State")

        # Preset states
        preset = st.sidebar.selectbox(
            "Choose preset state:",
            ["Custom", "|0⟩", "|1⟩", "|+⟩", "|-⟩", "|+i⟩", "|-i⟩"]
        )

        if preset == "|0⟩":
            st.session_state.alpha_real, st.session_state.alpha_imag = 1.0, 0.0
            st.session_state.beta_real, st.session_state.beta_imag = 0.0, 0.0
        elif preset == "|1⟩":
            st.session_state.alpha_real, st.session_state.alpha_imag = 0.0, 0.0
            st.session_state.beta_real, st.session_state.beta_imag = 1.0, 0.0
        elif preset == "|+⟩":
            st.session_state.alpha_real, st.session_state.alpha_imag = 1 / np.sqrt(2), 0.0
            st.session_state.beta_real, st.session_state.beta_imag = 1 / np.sqrt(2), 0.0
        elif preset == "|-⟩":
            st.session_state.alpha_real, st.session_state.alpha_imag = 1 / np.sqrt(2), 0.0
            st.session_state.beta_real, st.session_state.beta_imag = -1 / np.sqrt(2), 0.0
        elif preset == "|+i⟩":
            st.session_state.alpha_real, st.session_state.alpha_imag = 1 / np.sqrt(2), 0.0
            st.session_state.beta_real, st.session_state.beta_imag = 0.0, 1 / np.sqrt(2)
        elif preset == "|-i⟩":
            st.session_state.alpha_real, st.session_state.alpha_imag = 1 / np.sqrt(2), 0.0
            st.session_state.beta_real, st.session_state.beta_imag = 0.0, -1 / np.sqrt(2)

        if preset == "Custom":
            st.sidebar.write("α (amplitude for |0⟩):")
            alpha_real = st.sidebar.slider("α real part", -2.0, 2.0, st.session_state.alpha_real, 0.1)
            alpha_imag = st.sidebar.slider("α imaginary part", -2.0, 2.0, st.session_state.alpha_imag, 0.1)

            st.sidebar.write("β (amplitude for |1⟩):")
            beta_real = st.sidebar.slider("β real part", -2.0, 2.0, st.session_state.beta_real, 0.1)
            beta_imag = st.sidebar.slider("β imaginary part", -2.0, 2.0, st.session_state.beta_imag, 0.1)
        else:
            alpha_real = st.session_state.alpha_real
            alpha_imag = st.session_state.alpha_imag
            beta_real = st.session_state.beta_real
            beta_imag = st.session_state.beta_imag

        # Create initial state
        initial_state = normalize_state(
            alpha_real + 1j * alpha_imag,
            beta_real + 1j * beta_imag
        )

        # Apply selected gate
        gates = QuantumGates()
        if gate_type == "Pauli-X":
            gate_matrix = gates.pauli_x()
        elif gate_type == "Pauli-Y":
            gate_matrix = gates.pauli_y()
        elif gate_type == "Pauli-Z":
            gate_matrix = gates.pauli_z()
        elif gate_type == "Hadamard":
            gate_matrix = gates.hadamard()

        final_state = gate_matrix @ initial_state

        # Main content area for single qubit gates
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Before Gate Application")
            display_state_info(initial_state, "Initial State")

            x, y, z = bloch_coords_from_state(initial_state[0], initial_state[1])
            fig_initial = plot_bloch(x, y, z, height=400)
            st.plotly_chart(fig_initial, use_container_width=True, key=f"single_bloch_initial_{gate_type}")

        with col2:
            st.subheader("🎯 After Gate Application")
            display_state_info(final_state, f"After {gate_type}")

            x, y, z = bloch_coords_from_state(final_state[0], final_state[1])
            fig_final = plot_bloch(x, y, z, height=400)
            st.plotly_chart(fig_final, use_container_width=True, key=f"single_bloch_final_{gate_type}")
    # --- END SINGLE QUBIT GATES SECTION ---

    # Gate information section
    st.subheader("🔬 Gate Information")

    gate_info = {
        "Pauli-X": {
            "description": "The Pauli-X gate (bit-flip gate) acts like a NOT gate, flipping |0⟩ ↔ |1⟩",
            "matrix": "[[0, 1], [1, 0]]",
            "effect": "|0⟩ → |1⟩, |1⟩ → |0⟩"
        },
        "Pauli-Y": {
            "description": "The Pauli-Y gate combines bit-flip and phase-flip operations",
            "matrix": "[[0, -i], [i, 0]]",
            "effect": "|0⟩ → i|1⟩, |1⟩ → -i|0⟩"
        },
        "Pauli-Z": {
            "description": "The Pauli-Z gate (phase-flip gate) flips the phase of |1⟩",
            "matrix": "[[1, 0], [0, -1]]",
            "effect": "|0⟩ → |0⟩, |1⟩ → -|1⟩"
        },
        "Hadamard": {
            "description": "The Hadamard gate creates superposition states",
            "matrix": "(1/√2)[[1, 1], [1, -1]]",
            "effect": "|0⟩ → (|0⟩ + |1⟩)/√2, |1⟩ → (|0⟩ - |1⟩)/√2"
        },
        "CNOT": {
            "description": "The CNOT gate flips the target qubit if the control qubit is |1⟩",
            "matrix": "[[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]",
            "effect": "|00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩"
        }
    }

    info = gate_info[gate_type]
    st.markdown(f"""
    <div class="gate-info">
    <h4>{gate_type} Gate</h4>
    <p><strong>Description:</strong> {info['description']}</p>
    <p><strong>Matrix:</strong> {info['matrix']}</p>
    <p><strong>Effect:</strong> {info['effect']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Additional features
    with st.expander("📚 Learn More About Quantum Gates"):
        st.write("""
        ### Understanding Quantum Gates

        Quantum gates are the building blocks of quantum circuits. Unlike classical logic gates, 
        quantum gates are reversible and can create quantum superposition and entanglement.

        **Key Concepts:**
        - **Superposition**: A qubit can exist in a combination of $|0\\rangle$ and $|1\\rangle$ states
        - **Bloch Sphere**: A geometric representation of qubit states
        - **Quantum Interference**: Quantum amplitudes can interfere constructively or destructively
        - **Entanglement**: CNOT gate can create correlations between qubits

        **The Bloch Sphere:**
        - North pole: $|0\\rangle$ state
        - South pole: $|1\\rangle$ state  
        - Equator: superposition states with equal probability
        """)


if __name__ == "__main__":
    main()
