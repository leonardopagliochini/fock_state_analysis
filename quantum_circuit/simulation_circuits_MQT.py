#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/02/2025

@author: Maurizio Ferrari Dacrema
"""

import os

from quantum_circuit.qudit_gates import displacement_gate, SNAP_gate
from mqt.qudits.quantum_circuit import QuantumRegister, QuantumCircuit
import numpy as np

def SNAP_Displacement_Circut_MQT(n_qudits, d, n_bumpers, n_blocks, gate_parameters_dict):
    """
    Constructs a quantum circuit with multiple layers of SNAP-Displacement operations.

    Parameters:
    - n_qudits (int): Number of qudits.
    - d (int): Qudit dimension.
    - n_bumpers (int): Number of bumper states.
    - n_blocks (int): Number of SNAP-displacement blocks.
    - parameter_dict: dictionary of the parameters label:value

    Returns:
    - QuantumCircuit: The constructed quantum circuit.
    """

    circuit = QuantumCircuit()
    qR = QuantumRegister("qR", size=n_qudits, dims=[d + n_bumpers] * n_qudits)
    circuit.append(qR)

    for block in range(n_blocks + 1):
        for q in range(n_qudits):
            complex_alpha = gate_parameters_dict[f"alpha_block{block}_q{q}"]
            D = displacement_gate(complex_alpha, d + n_bumpers)
            circuit.cu_one(qR[q], D)
            
            if block < n_blocks:
                theta_list = [gate_parameters_dict[f"theta_block{block}_q{q}_level{level}"] for level in range(d)]
                SNAP = SNAP_gate(theta_list, d, n_bumpers = n_bumpers)
                circuit.cu_one(qR[q], SNAP)
        
    return circuit


# import torch
# from quantum_circuit.qudit_gates import displacement_gate_torch, SNAP_gate_torch
#
# def SNAP_Displacement_Circuit_torch(
#     d: int,
#     n_layers: int,
#     parameter_dict: dict,
#     initial_state: torch.Tensor,
#     device: str = 'cpu'
#     ) -> torch.Tensor:
#
#     """
#     Builds and applies a sequence of Displacement + SNAP gates to an initial state in a d-dimensional Hilbert space,
#     using the parameters stored in 'parameter_dict'.
#
#     The circuit is parameterized by:
#       - alpha_real[i], alpha_imag[i] for each step i (0 <= i < n_steps)
#       - thetas[i, :] for each step i
#
#     Implementation:
#       - For step i, we construct alpha = alpha_real[i] + 1j * alpha_imag[i]
#         and then the Displacement gate D(alpha).
#       - We also construct the SNAP gate from thetas[i, :].
#       - We multiply them to form step_op = S_i * D_i.
#       - We accumulate these step operators into a total circuit operator
#         or directly apply them to the state in sequence (whichever approach you prefer).
#
#     Parameters
#     ----------
#     initial_state : torch.Tensor
#         A (d, 1) or (d,) complex tensor representing the initial state.
#     parameter_dict : dict
#         Must contain 'alpha_real', 'alpha_imag', and 'thetas' as nn.Parameters.
#         alpha_real, alpha_imag: shape [n_steps]
#         thetas: shape [n_steps, d]
#     d : int
#         Truncation dimension of the Hilbert space.
#     dtype : torch.dtype, optional
#         Torch dtype to use (e.g., torch.complex128). Defaults to torch.complex128.
#     device : str, optional
#         Device on which to allocate tensors. Defaults to 'cpu'.
#
#     Returns
#     -------
#     final_state : torch.Tensor
#         The final state after applying all Displacement + SNAP gates in sequence.
#         Shape (d, 1) or (d,).
#     """
#
#     # Ensure the initial_state is a column vector
#     if initial_state.ndim == 1:
#         initial_state = initial_state.reshape(d, 1)
#
#     final_state = initial_state.clone()
#     #circuit_op = torch.eye(d, d, device = device).to(torch.complex64)
#
#     for layer in range(n_layers+1):  # Apply multiple layers
#         for q in [0]:
#             complex_alpha = parameter_dict[f"alpha_layer{layer}_q{q}"]
#             D = displacement_gate_torch(complex_alpha, d, device = device)
#             #circuit_op = torch.matmul(D, circuit_op)
#             final_state = torch.matmul(D, final_state)
#             final_state = final_state / torch.linalg.vector_norm(final_state, ord=2) # The truncated displacement is not exactly unitary
#
#             if layer < n_layers:
#                 theta_list = [parameter_dict[f"theta_layer{layer}_q{q}_level{level}"] for level in range(d)]
#                 S = SNAP_gate_torch(theta_list, d, device = device)
#                 #circuit_op = torch.matmul(S, circuit_op)
#                 final_state = torch.matmul(S, final_state)
#
#     # Apply final operator to initial_state
#     #final_state = torch.matmul(circuit_op, initial_state)
#
#     return final_state


# Define the Cost Function (Fidelity-Based)
def compute_fidelity(state1, state2):
    """
    Computes the fidelity between two quantum states.

    Fidelity is defined as:
        F(ψ, φ) = |⟨ψ | φ⟩|^2

    Parameters:
    - state1 (np.ndarray): First quantum state (must be normalized).
    - state2 (np.ndarray): Second quantum state (must be normalized).

    Returns:
    - float: Fidelity value in [0,1], where 1 means identical states.
    """
    # Ensure states are column vectors
    state1 = state1.flatten()
    state2 = state2.flatten()

    # Compute fidelity
    return np.abs(np.vdot(state1, state2)) ** 2


def run_simulation_MQT(encoded_params, n_qudits, d, n_blocks, backend, n_bumpers = 0):

    circuit = SNAP_Displacement_Circut_MQT(n_qudits=n_qudits, d=d, n_bumpers = n_bumpers, n_blocks = n_blocks, gate_parameters_dict = encoded_params)

    job = backend.run(circuit)
    result = job.result()
    prepared_state = result.get_state_vector()

    return prepared_state

