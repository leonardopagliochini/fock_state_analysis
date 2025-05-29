#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/03/2025

@author: Maurizio Ferrari Dacrema
"""

from quantum_circuit.qudit_gates import displacement_gate, SNAP_gate, displacement_gate_torch, SNAP_gate_torch
import torch

class SNAP_Displacement(object):
    """
    
    Parameter Mapping for SNAP and Displacement Gates with Real-Imaginary Encoding.
    Each RL action input is assumed to be **bounded between [-1, 1]** and then mapped accordingly.

    - **Displacement Gate Parameters:**
      - Encoded using **two separate real numbers**: one for the **real part** and another for the **imaginary part**.

    - **SNAP Gate Parameters:**
      - Encoded as **angles** directly within the range **[-π, +π]**.
    """
    
    def __init__(self, d, n_bumpers, n_qudits, n_blocks):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param d: Energy levels (dimensionality) of each qudit.
        :param n_bumpers: Total bumper states in the system.
        :param n_qudits: Number of qudits in the quantum system.
        :param n_blocks: Number of layers in the quantum circuit.
        """
        super(SNAP_Displacement, self).__init__()
        self.parameter_types = {}

        self.d = d
        self.n_bumpers = n_bumpers
        self.n_qudits = n_qudits
        self.n_blocks = n_blocks
        
        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                # Displacement gate parameters (Real and Imaginary parts)
                label_alpha = f"alpha_block{block}_q{q}"
                self.parameter_types[label_alpha] = "complex"
                
                if block < self.n_blocks:
                    # SNAP gate phase parameters (one per energy level d)
                    for level in range(self.d):
                        label_theta = f"theta_block{block}_q{q}_level{level}"
                        self.parameter_types[label_theta] = "angle"
    
    def get_parameter_types(self):
        return self.parameter_types.copy()
    
    def run_state(self,
            parameter_dict: dict,
            initial_state: torch.Tensor,
            device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Builds and applies a sequence of Displacement + SNAP gates to an initial state in a d-dimensional Hilbert space,
        using the parameters stored in 'parameter_dict'.
    
        The circuit is parameterized by:
          - alpha_real[i], alpha_imag[i] for each step i (0 <= i < n_steps)
          - thetas[i, :] for each step i
    
        Implementation:
          - For step i, we construct alpha = alpha_real[i] + 1j * alpha_imag[i]
            and then the Displacement gate D(alpha).
          - We also construct the SNAP gate from thetas[i, :].
          - We multiply them to form step_op = S_i * D_i.
          - We accumulate these step operators into a total circuit operator
            or directly apply them to the state in sequence (whichever approach you prefer).
    
        Parameters
        ----------
        initial_state : torch.Tensor
            A (d, 1) or (d,) complex tensor representing the initial state.
        parameter_dict : dict
            Must contain 'alpha_real', 'alpha_imag', and 'thetas' as nn.Parameters.
            alpha_real, alpha_imag: shape [n_steps]
            thetas: shape [n_steps, d]
        d : int
            Truncation dimension of the Hilbert space.
        dtype : torch.dtype, optional
            Torch dtype to use (e.g., torch.complex128). Defaults to torch.complex128.
        device : str, optional
            Device on which to allocate tensors. Defaults to 'cpu'.
    
        Returns
        -------
        final_state : torch.Tensor
            The final state after applying all Displacement + SNAP gates in sequence.
            Shape (d, 1) or (d,).
        """
        
        assert initial_state.shape[0] == self.d + self.n_bumpers, f"Initial states must be the same size of d + n_bumpers, expected size was {self.d + self.n_bumpers} actual size was {initial_state.shape[0]}"
        
        # Ensure the initial_state is a column vector
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(self.d + self.n_bumpers, 1)
        
        final_state = initial_state.clone()
        # circuit_op = torch.eye(d, d, device = device).to(torch.complex64)
        
        for block in range(self.n_blocks + 1):  # Apply multiple layers
            for q in range(self.n_qudits):
                alpha = parameter_dict[f"alpha_block{block}_q{q}"]
                D = displacement_gate_torch(alpha, self.d + self.n_bumpers, device = device)
                # circuit_op = torch.matmul(D, circuit_op)
                final_state = torch.matmul(D, final_state)
                # The truncated displacement is not exactly unitary, renormalize
                final_state = final_state / torch.linalg.vector_norm(final_state, ord = 2)
                
                if block < self.n_blocks:
                    theta_list = [parameter_dict[f"theta_block{block}_q{q}_level{level}"] for level in range(self.d)]
                    S = SNAP_gate_torch(theta_list, self.d, n_bumpers = self.n_bumpers, device = device)
                    # circuit_op = torch.matmul(S, circuit_op)
                    final_state = torch.matmul(S, final_state)
        
        # Apply final operator to initial_state
        # final_state = torch.matmul(circuit_op, initial_state)
        
        return final_state.squeeze()






class SNAP1_Displacement(SNAP_Displacement):
    """

    Parameter Mapping for SNAP and Displacement Gates with Real-Imaginary Encoding.
    Each RL action input is assumed to be **bounded between [-1, 1]** and then mapped accordingly.

    - **Displacement Gate Parameters:**
      - Encoded using **two separate real numbers**: one for the **real part** and another for the **imaginary part**.

    - **SNAP Gate Parameters:**
      - Encoded as **angles** directly within the range **[-π, +π]**.
    """

    def __init__(self, d, n_bumpers, n_qudits, n_blocks):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param d: Energy levels (dimensionality) of each qudit.
        :param n_bumpers: Total bumper states in the system.
        :param n_qudits: Number of qudits in the quantum system.
        :param n_blocks: Number of layers in the quantum circuit.
        """
        super(SNAP1_Displacement, self).__init__(d, n_bumpers, n_qudits, n_blocks)
        self.parameter_types = {}

        self.d = d
        self.n_bumpers = n_bumpers
        self.n_qudits = n_qudits
        self.n_blocks = n_blocks

        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                # Displacement gate parameters (Real and Imaginary parts)
                label_alpha = f"alpha_block{block}_q{q}"
                self.parameter_types[label_alpha] = "real"

                if block < self.n_blocks:
                    # SNAP gate phase parameters (one per block with the corresponding energy level d)
                    label_theta = f"theta_block{block}_q{q}_level{block}"
                    self.parameter_types[label_theta] = "angle"


    def run_state(self,
            parameter_dict: dict,
            initial_state: torch.Tensor,
            device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Builds and applies a sequence of Displacement + SNAP gates to an initial state in a d-dimensional Hilbert space,
        using the parameters stored in 'parameter_dict'.

        The circuit is parameterized by:
          - alpha_real[i], alpha_imag[i] for each step i (0 <= i < n_steps)
          - thetas[i, :] for each step i

        Implementation:
          - For step i, we construct alpha = alpha_real[i] + 1j * alpha_imag[i]
            and then the Displacement gate D(alpha).
          - We also construct the SNAP gate from thetas[i, :].
          - We multiply them to form step_op = S_i * D_i.
          - We accumulate these step operators into a total circuit operator
            or directly apply them to the state in sequence (whichever approach you prefer).

        Parameters
        ----------
        initial_state : torch.Tensor
            A (d, 1) or (d,) complex tensor representing the initial state.
        parameter_dict : dict
            Must contain 'alpha_real', 'alpha_imag', and 'thetas' as nn.Parameters.
            alpha_real, alpha_imag: shape [n_steps]
            thetas: shape [n_steps, d]
        d : int
            Truncation dimension of the Hilbert space.
        dtype : torch.dtype, optional
            Torch dtype to use (e.g., torch.complex128). Defaults to torch.complex128.
        device : str, optional
            Device on which to allocate tensors. Defaults to 'cpu'.

        Returns
        -------
        final_state : torch.Tensor
            The final state after applying all Displacement + SNAP gates in sequence.
            Shape (d, 1) or (d,).
        """

        parameter_dict = parameter_dict.copy()

        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):

                if block < self.n_blocks:
                    # Add missing SNAP gate phase parameters (one per every energy level d != block)
                    for level in range(self.d):
                        if level != block:
                            label_theta = f"theta_block{block}_q{q}_level{level}"

                            if label_theta in parameter_dict:
                                print(f"Warning: parameter {label_theta} should not be present and will be ignored.")
                            parameter_dict[label_theta] = torch.zeros(1)

        final_state = super(SNAP1_Displacement, self).run_state(parameter_dict = parameter_dict,
                                                                initial_state = initial_state,
                                                                device = device)
        return final_state
