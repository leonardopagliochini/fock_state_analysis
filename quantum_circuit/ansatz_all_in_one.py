#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/03/2025

@author: Maurizio Ferrari Dacrema
"""

from quantum_circuit.qudit_gates import displacement_gate, SNAP_gate, displacement_gate_torch, SNAP_gate_torch, photon_loss_kraus_operators
import torch, math
import torch.nn as nn
from utils.parameter_encoding import complex_to_polar, polar_to_complex
import torch.nn.functional as F
import matplotlib.pyplot as plt
from quantum_circuit.qudit_gates import fourier_gate_torch


class SNAP_Displacement(object):
    """
    SNAP Displacement protocol in the form D^+ S D.

    The parameters are encoded as follows:
    - Displacement: the complex parameter is encoded using the polar-phase representation. The radius is a positive
                    number and the phase is an angle in [-π, +π], obtained with a Tanh.

    - SNAP: the phases are angles in [-π, +π], obtained with a Tanh.

    Note: the use of a Tanh to transform an unbounded parameter in [-1, +1] for the angles may slow learning
    if very large angles are requires and the learning needs to push in the asymptotic region of the function.
    """

    def __init__(self, d, n_bumpers, n_blocks, device):
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
        self.n_qudits = 1
        self.n_blocks = n_blocks
        self.device = device
        self._init_parameters()

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


    def _init_parameters(self):
        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype = torch.float32, device = self.device))
        self.S_params = nn.Parameter(torch.rand((self.n_blocks, self.d), dtype = torch.float32, device = self.device))


    def get_parameter_types(self):
        return self.parameter_types.copy()

    def clone_torch_parameters(self):
        return {key:value.detach().cpu().numpy() for key, value in self.get_torch_parameters().items()}

    def _get_gate_params(self):

        D_radii = nn.ReLU()(self.D_params[:,0])
        D_angles = nn.Tanh()(self.D_params[:,1]) * torch.pi
        D_gate = polar_to_complex(D_radii, D_angles)

        S_gate = nn.Tanh()(self.S_params) * torch.pi

        return D_gate, S_gate


    def get_n_parameters(self):
        return self.D_params.numel() + self.S_params.numel()


    def get_n_gates(self):
        return self.d * 2 + 1


    def get_torch_parameters(self):
        return {"D_params": self.D_params, "S_params": self.S_params}

    def set_torch_parameters(self, param_dict):
        with torch.no_grad():
            for name in ["D_params", "S_params"]:
                attribute_value = getattr(self, name)
                param_tensor = torch.tensor(param_dict[name], dtype=attribute_value.dtype, device=attribute_value.device)
                attribute_value.data.copy_(param_tensor)


    def encoded_to_gate_params(self):

        D_gate, S_gate = self._get_gate_params()
        gate_parameters = {}

        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                # Displacement gate parameters (Real and Imaginary parts)
                label_alpha = f"alpha_block{block}_q{q}"
                gate_parameters[label_alpha] = D_gate[block].detach()

                if block < self.n_blocks:
                    # SNAP gate phase parameters (one per energy level d)
                    for level in range(self.d):
                        label_theta = f"theta_block{block}_q{q}_level{level}"
                        gate_parameters[label_theta] = S_gate[block, level].detach()

        return gate_parameters


    def run_state(self,
            initial_state: torch.Tensor,
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
            initial_state = initial_state.reshape(-1, 1)

        final_state = initial_state.clone()
        # circuit_op = torch.eye(d, d, device = device).to(torch.complex64)

        D_gate, S_gate = self._get_gate_params()
        self._cached_gate_params = (D_gate, S_gate)

        for block in range(self.n_blocks + 1):  # Apply multiple layers
            for q in range(self.n_qudits):
                alpha = D_gate[block].unsqueeze(0)  # Compilation require this to be a (1,) tensor
                D = displacement_gate_torch(alpha, self.d + self.n_bumpers, device = self.device)
                final_state = torch.matmul(D, final_state)

                # The truncated displacement is not exactly unitary, renormalize
                final_state = final_state / torch.linalg.vector_norm(final_state, ord = 2)

                if block < self.n_blocks:
                    theta_list = S_gate[block, :]
                    S = SNAP_gate_torch_AllInOne(theta_list, n_bumpers = self.n_bumpers)
                    final_state = torch.matmul(S, final_state)

        return final_state.squeeze()
    
    def run_state_from_params(self, input_state, D_params, S_params):
        self.set_parameters_from_tensors(D_params, S_params)
        return self.run_state(input_state)

        
    def set_parameters_from_tensors(self, D_tensor, S_tensor):
        self.D_params.data = D_tensor.clone().detach().to(self.device).requires_grad_()
        self.S_params.data = S_tensor.clone().detach().to(self.device).requires_grad_()




    def run_unitary(self) -> torch.Tensor:
        """
        Constructs the full unitary matrix corresponding to the SNAP-displacement sequence.
        It starts from the identity and sequentially applies the gates: D_0, S_0, D_1, ..., S_{n-1}, D_n.

        Returns
        -------
        total_unitary : torch.Tensor
            The full unitary operator as a (d + n_bumpers, d + n_bumpers) complex matrix.
        """

        dim = self.d + self.n_bumpers
        total_unitary = torch.eye(dim, dtype=torch.complex64, device=self.device)

        D_gate, S_gate = self._get_gate_params()
        self._cached_gate_params = (D_gate, S_gate)

        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                alpha = D_gate[block].unsqueeze(0)
                D = displacement_gate_torch(alpha, dim, device=self.device)
                total_unitary = torch.matmul(D, total_unitary)

                # Apply SNAP if not final block
                if block < self.n_blocks:
                    theta_list = S_gate[block, :]
                    S = SNAP_gate_torch_AllInOne(theta_list, n_bumpers=self.n_bumpers)
                    total_unitary = torch.matmul(S, total_unitary)

        return total_unitary



    def run_density(
            self,
            initial_state: torch.Tensor,
            eta=0.9,
            ell_max=4,
    ):
        """
        Evolve an initial state under D-S-D ansatz with photon loss (Kraus ops).

        Parameters
        ----------
        initial_state : (d,) complex tensor (pure state)
        D_gate : list of complex displacement amplitudes (torch tensors)
        S_gate : list of real phase vectors per block
        phase_indices : list of lists of indices (one per block)
        eta : float
            Photon survival probability (e.g., eta=0.98 means 2% loss)
        ell_max : int
            Max number of lost photons to include in the Kraus sum
        """

        assert initial_state.shape[0] == self.d + self.n_bumpers, f"Initial states must be the same size of d + n_bumpers, expected size was {self.d + self.n_bumpers} actual size was {initial_state.shape[0]}"

        # Ensure the initial_state is a column vector
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(-1, 1)

        d = initial_state.shape[0]
        rho = initial_state.view(-1, 1) @ initial_state.conj().view(1, -1)

        kraus_ops = photon_loss_kraus_operators(d, eta, ell_max, self.device)

        D_gate, S_gate = self._get_gate_params()
        self._cached_gate_params = (D_gate, S_gate)

        # Main evolution
        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                alpha = D_gate[block]
                D = displacement_gate_torch(alpha, self.d + self.n_bumpers, device = self.device)
                rho = D @ rho @ D.conj().T

                if block < self.n_blocks:
                    theta_list = S_gate[block, :]
                    S = SNAP_gate_torch_AllInOne(theta_list, n_bumpers = self.n_bumpers)
                    rho = S @ rho @ S.conj().T

                if eta < 1.0:
                    rho_new = torch.zeros_like(rho)
                    for K in kraus_ops:
                        rho_new += K @ rho @ K.conj().T
                    rho = rho_new

                tr = torch.trace(rho).real
                if not (0.99 < tr < 1.01):
                    raise RuntimeError(f"[!] Large non-unitarity detected at block {block}: Tr(ρ) = {tr.item():.6f}")
                elif torch.abs(tr - 1.0) > 1e-4:
                    print(f"[Warning] Trace deviation at block {block}: Tr(ρ) = {tr.item():.6f}")

                rho = rho / tr  # always renormalize

        return rho



def SNAP_gate_torch_AllInOne(theta_list, n_bumpers = 0):
    """
    Creates the SNAP gate for a qudit system of dimension d, implemented in PyTorch.

    The SNAP gate is a diagonal operator with phases exp(i * theta_k) on each Fock basis state |k>:
        SNAP(theta_list) = diag(exp(i * theta_0), exp(i * theta_1), ..., exp(i * theta_{d-1}))

    Parameters
    ----------
    theta_list : list or array-like
        List of phases theta_k, must be length d.
    d : int
        Dimension of the Hilbert space (truncation).
    dtype : torch.dtype, optional
        Data type to use for the operator. Defaults to torch.complex128.
    device : str, optional
        Device on which to create the operator (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns
    -------
    snap_operator : torch.Tensor
        The diagonal SNAP operator, shape (d, d), as a complex PyTorch tensor.
    """
    # assert len(theta_list) == d, "Theta list length must match qudit dimension d."
    # assert n_bumpers >= 0, "Number of bumper states must be >= 0"

    # Convert phases to a torch tensor and exponentiate
    #theta_list = torch.stack(theta_list, dim = 0).squeeze(-1)

    # This is a simple r = e^(i*theta)
    # torch.exp(1j * theta_list)
    # It is written like this because it is compilable by Triton
    phases = torch.exp(torch.complex(torch.zeros(theta_list.shape[0], device=theta_list.device), theta_list))

    # if n_bumpers > 0: IF statements break computational graph, if n_bumpers=0 should create empty tensor
    bumper = torch.ones(n_bumpers, dtype = phases.dtype, device = phases.device)
    phases = torch.cat([phases, bumper], dim = 0)

    # Construct diagonal matrix
    return torch.diag(phases)





class SNAPGumbel_Displacement(SNAP_Displacement):
    """
    SNAP Displacement protocol in the form D^+ S D.
    The SNAP gates are restricted as having only one (almost) non-one phase. The state this phase should be applied to
    is determined with a Gumbel softmax which can be used to perform one-hot encoding of a vector and hence select
    only one phase.

    The parameters are encoded as follows:
    - Displacement: the complex parameter is encoded using the polar-phase representation. The radius is a positive
                    number and the phase is an angle in [-π, +π], obtained with a Tanh.

    - SNAP: the phases are angles in [-π, +π], obtained with a Tanh.

    Note: the use of a Tanh to transform an unbounded parameter in [-1, +1] for the angles may slow learning
    if very large angles are requires and the learning needs to push in the asymptotic region of the function.
    """

    def __init__(self, d, n_bumpers, n_blocks, device, tau = 0.5):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param d: Energy levels (dimensionality) of each qudit.
        :param n_bumpers: Total bumper states in the system.
        :param n_qudits: Number of qudits in the quantum system.
        :param n_blocks: Number of layers in the quantum circuit.
        """
        super(SNAPGumbel_Displacement, self).__init__(d, n_bumpers, n_blocks, device)
        self.tau = tau

    def _init_parameters(self):
        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype = torch.float32, device = self.device))
        self.S_params = nn.Parameter(torch.rand((self.n_blocks, self.d), dtype = torch.float32, device = self.device))
        self.S_logits = nn.Parameter(torch.rand((self.n_blocks, self.d), dtype = torch.float32, device = self.device))


    def get_torch_parameters(self):
        return {"D_params": self.D_params, "S_params": self.S_params, "S_logits": self.S_logits}

    def _get_gate_params(self):

        D_radii = nn.ReLU()(self.D_params[:,0])
        D_angles = nn.Tanh()(self.D_params[:,1]) * torch.pi
        D_gate = polar_to_complex(D_radii, D_angles)

        # Phase angle per block
        phase_angles = nn.Tanh()(self.S_params) * torch.pi

        # One-hot selection vector using Gumbel-softmax
        one_hot_mask = F.gumbel_softmax(self.S_logits, tau=self.tau, hard=True)

        # Multiply each block's angle with the selected level
        # Resulting shape: (n_blocks, d)
        S_gate = one_hot_mask * phase_angles

        return D_gate, S_gate


    def set_torch_parameters(self, param_dict):
        with torch.no_grad():
            for name in ["D_params", "S_params", "S_logits"]:
                attribute_value = getattr(self, name)
                param_tensor = torch.tensor(param_dict[name], dtype=attribute_value.dtype, device=attribute_value.device)
                attribute_value.data.copy_(param_tensor)



class SNAPSymmetricFactorized_Displacement(SNAP_Displacement):
    """
    SNAP Displacement protocol in the form D^+ S D.
    The SNAP gates are parametrized as the product of two low-dimensional matrices of dimension rank.
    Since the number of parameters is: 2*d (displacement) + d^2 SNAP, a reduction in the number of parameters
    for the SNAP can be achieved if d*rank + rank*d < d^2, hence rank < d/2

    The parameters are encoded as follows:
    - Displacement: the complex parameter is encoded using the polar-phase representation. The radius is a positive
                    number and the phase is an angle in [-π, +π], obtained with a Tanh.

    - SNAP: the phases are angles in [-π, +π], obtained with a Tanh.

    Note: the use of a Tanh to transform an unbounded parameter in [-1, +1] for the angles may slow learning
    if very large angles are requires and the learning needs to push in the asymptotic region of the function.
    """

    def __init__(self, d, n_bumpers, n_blocks, rank, device):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param d: Energy levels (dimensionality) of each qudit.
        :param n_bumpers: Total bumper states in the system.
        :param n_qudits: Number of qudits in the quantum system.
        :param n_blocks: Number of layers in the quantum circuit.
        """

        assert rank >= 1, f"SNAPSymmetricFactorized_Displacement: The rank must be >= 1, the value was {rank}"

        if rank >= d/2:
            print(f"SNAPFactorized_Displacement: the chosen rank {rank} is higher than d/2 {d/2:.1f}, which results in "
                  f"a number of SNAP parameters greater than the d^2 required by a non factorized model.")
        self.rank = rank
        super(SNAPSymmetricFactorized_Displacement, self).__init__(d, n_bumpers, n_blocks, device)

    def _init_parameters(self):
        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype = torch.float32, device = self.device))
        self.S_V_params = nn.Parameter(torch.rand((self.n_blocks, self.rank), dtype = torch.float32, device = self.device))


    def get_torch_parameters(self):
        return {"D_params": self.D_params, "S_V_params": self.S_V_params}


    def set_torch_parameters(self, param_dict):
        with torch.no_grad():
            for name in ["D_params", "S_V_params"]:
                attribute_value = getattr(self, name)
                param_tensor = torch.tensor(param_dict[name], dtype=attribute_value.dtype, device=attribute_value.device)
                attribute_value.data.copy_(param_tensor)


    def get_n_parameters(self):
        return self.D_params.numel() + self.S_V_params.numel()

    def _get_gate_params(self):

        D_radii = nn.ReLU()(self.D_params[:,0])
        D_angles = nn.Tanh()(self.D_params[:,1]) * torch.pi
        D_gate = polar_to_complex(D_radii, D_angles)

        S_reconstructed = self.S_V_params.mm(self.S_V_params.T)
        S_gate = nn.Tanh()(S_reconstructed) * torch.pi

        return D_gate, S_gate


class SNAPAsymmetricFactorized_Displacement(SNAP_Displacement):
    """
    SNAP Displacement protocol in the form D^+ S D.
    The SNAP gates are parametrized as the product of two low-dimensional matrices of dimension rank.
    Since the number of parameters is: 2*d (displacement) + d^2 SNAP, a reduction in the number of parameters
    for the SNAP can be achieved if d*rank + rank*d < d^2, hence rank < d/2

    The parameters are encoded as follows:
    - Displacement: the complex parameter is encoded using the polar-phase representation. The radius is a positive
                    number and the phase is an angle in [-π, +π], obtained with a Tanh.

    - SNAP: the phases are angles in [-π, +π], obtained with a Tanh.

    Note: the use of a Tanh to transform an unbounded parameter in [-1, +1] for the angles may slow learning
    if very large angles are requires and the learning needs to push in the asymptotic region of the function.
    """

    def __init__(self, d, n_bumpers, n_blocks, rank, device):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param d: Energy levels (dimensionality) of each qudit.
        :param n_bumpers: Total bumper states in the system.
        :param n_qudits: Number of qudits in the quantum system.
        :param n_blocks: Number of layers in the quantum circuit.
        """

        assert rank >= 1, f"SNAPSymmetricFactorized_Displacement: The rank must be >= 1, the value was {rank}"

        if rank >= d/2:
            print(f"SNAPFactorized_Displacement: the chosen rank {rank} is higher than d/2 {d/2:.1f}, which results in "
                  f"a number of SNAP parameters greater than the d^2 required by a non factorized model.")
        self.rank = rank
        super(SNAPAsymmetricFactorized_Displacement, self).__init__(d, n_bumpers, n_blocks, device)

    def _init_parameters(self):
        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype = torch.float32, device = self.device))
        self.S_V_params = nn.Parameter(torch.rand((self.n_blocks, self.rank), dtype = torch.float32, device = self.device))
        self.S_U_params = nn.Parameter(torch.rand((self.rank, self.d), dtype = torch.float32, device = self.device))


    def get_torch_parameters(self):
        return {"D_params": self.D_params, "S_V_params": self.S_V_params, "S_U_params": self.S_U_params}


    def set_torch_parameters(self, param_dict):
        with torch.no_grad():
            for name in ["D_params", "S_V_params", "S_U_params"]:
                attribute_value = getattr(self, name)
                param_tensor = torch.tensor(param_dict[name], dtype=attribute_value.dtype, device=attribute_value.device)
                attribute_value.data.copy_(param_tensor)


    def get_n_parameters(self):
        return self.D_params.numel() + self.S_V_params.numel() + self.S_U_params.numel()

    def _get_gate_params(self):

        D_radii = nn.ReLU()(self.D_params[:,0])
        D_angles = nn.Tanh()(self.D_params[:,1]) * torch.pi
        D_gate = polar_to_complex(D_radii, D_angles)

        S_reconstructed = self.S_V_params.mm(self.S_U_params)
        S_gate = nn.Tanh()(S_reconstructed) * torch.pi

        return D_gate, S_gate




class SNAP1_Displacement(SNAP_Displacement):
    """
    SNAP Displacement protocol in the form D^+ S D.
    The SNAP gates are restricted as having only one non-one phase, the one corresponding to the block index.

    The parameters are encoded as follows:
    - Displacement: the complex parameter is encoded using the polar-phase representation. The radius is a positive
                    number and the phase is an angle in [-π, +π], obtained with a Tanh.

    - SNAP: the phases are angles in [-π, +π], obtained with a Tanh.

    Note: the use of a Tanh to transform an unbounded parameter in [-1, +1] for the angles may slow learning
    if very large angles are requires and the learning needs to push in the asymptotic region of the function.
    """

    def __init__(self, d, n_bumpers, n_blocks, device):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param d: Energy levels (dimensionality) of each qudit.
        :param n_bumpers: Total bumper states in the system.
        :param n_qudits: Number of qudits in the quantum system.
        :param n_blocks: Number of layers in the quantum circuit.
        """
        super(SNAP1_Displacement, self).__init__(d, n_bumpers, n_blocks, device)
        self.parameter_types = {}

        self.d = d
        self.n_bumpers = n_bumpers
        self.n_qudits = 1
        self.n_blocks = n_blocks
        self.device = device
        self._init_parameters()


        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                # Displacement gate parameters (Real and Imaginary parts)
                label_alpha = f"alpha_block{block}_q{q}"
                self.parameter_types[label_alpha] = "complex"

                if block < self.n_blocks:
                    # SNAP gate phase parameters (one per block with the corresponding energy level d)
                    label_theta = f"theta_block{block}_q{q}_level{block}"
                    self.parameter_types[label_theta] = "angle"


    def _init_parameters(self):
        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype = torch.float32, device = self.device))
        self.S_params = nn.Parameter(torch.rand((self.n_blocks, ), dtype = torch.float32, device = self.device))


    def get_parameter_types(self):
        return self.parameter_types.copy()


    def encoded_to_gate_params(self):

        D_gate, S_gate = self._get_gate_params()
        gate_parameters = {}

        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                # Displacement gate parameters (Real and Imaginary parts)
                label_alpha = f"alpha_block{block}_q{q}"
                gate_parameters[label_alpha] = D_gate[block].detach()

                if block < self.n_blocks:
                    # SNAP gate phase parameters (one per energy level d)
                    for level in range(self.d):
                        label_theta = f"theta_block{block}_q{q}_level{level}"
                        if level == block:
                            gate_parameters[label_theta] = S_gate[block].detach()
                        else:
                            gate_parameters[label_theta] = torch.zeros(1, dtype = torch.float32, device=self.device)

        return gate_parameters


    def run_state(self,
            initial_state: torch.Tensor,
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
            initial_state = initial_state.reshape(-1, 1)

        final_state = initial_state.clone()
        # circuit_op = torch.eye(d, d, device = device).to(torch.complex64)

        D_gate, S_gate = self._get_gate_params()
        self._cached_gate_params = (D_gate, S_gate)

        for block in range(self.n_blocks + 1):  # Apply multiple layers
            for q in range(self.n_qudits):
                #alpha = parameter_dict[f"alpha_block{block}_q{q}"]
                alpha = D_gate[block].unsqueeze(0)  # Compilation require this to be a (1,) tensor
                D = displacement_gate_torch(alpha, self.d + self.n_bumpers, device = self.device)
                # circuit_op = torch.matmul(D, circuit_op)
                final_state = torch.matmul(D, final_state)
                # The truncated displacement is not exactly unitary, renormalize
                final_state = final_state / torch.linalg.vector_norm(final_state, ord = 2)

                if block < self.n_blocks:
                    #theta_list = [parameter_dict[f"theta_block{block}_q{q}_level{level}"] for level in range(self.d)]
                    theta_list = torch.zeros(self.d, dtype = torch.float32, device=self.device)
                    theta_list[block] = S_gate[block]
                    S = SNAP_gate_torch_AllInOne(theta_list, n_bumpers = self.n_bumpers)
                    # circuit_op = torch.matmul(S, circuit_op)
                    final_state = torch.matmul(S, final_state)

        # Apply final operator to initial_state
        # final_state = torch.matmul(circuit_op, initial_state)

        return final_state.squeeze()





class SNAPn_Displacement(SNAP_Displacement):
    """
    SNAP Displacement protocol in the form D^+ S D.
    The SNAP gates are restricted as having only n contiguous non-one phases, starting from the block index.

    The parameters are encoded as follows:
    - Displacement: the complex parameter is encoded using the polar-phase representation. The radius is a positive
                    number and the phase is an angle in [-π, +π], obtained with a Tanh.

    - SNAP: the phases are angles in [-π, +π], obtained with a Tanh.

    Note: the use of a Tanh to transform an unbounded parameter in [-1, +1] for the angles may slow learning
    if very large angles are requires and the learning needs to push in the asymptotic region of the function.
    """

    def __init__(self, d, n_bumpers, n_blocks, width, device):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param d: Energy levels (dimensionality) of each qudit.
        :param n_bumpers: Total bumper states in the system.
        :param n_qudits: Number of qudits in the quantum system.
        :param n_blocks: Number of layers in the quantum circuit.
        """

        self.d = d
        self.n_bumpers = n_bumpers
        self.n_qudits = 1
        self.n_blocks = n_blocks
        self.device = device
        self.width = width
        self._learnable_levels_per_block = [self._get_learnable_levels(block) for block in range(n_blocks)]
        super(SNAPn_Displacement, self).__init__(d, n_bumpers, n_blocks, device)

        self.parameter_types = {}
        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                # Displacement gate parameters (Real and Imaginary parts)
                label_alpha = f"alpha_block{block}_q{q}"
                self.parameter_types[label_alpha] = "complex"

                if block < self.n_blocks:
                    # SNAP gate phase parameters (n_learnable_phases per block starting from the corresponding energy level d)
                    indices = self._learnable_levels_per_block[block]
                    for level in indices:
                        label_theta = f"theta_block{block}_q{q}_level{level}"
                        self.parameter_types[label_theta] = "angle"
           
                        

    def _init_parameters(self):

        max_phases = max([len(self._learnable_levels_per_block[block]) for block in range(self.n_blocks)])

        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype = torch.float32, device = self.device))
        self.S_params = nn.Parameter(torch.rand((self.n_blocks, max_phases), dtype = torch.float32, device = self.device))


    def get_parameter_types(self):
        return self.parameter_types.copy()


    def encoded_to_gate_params(self):

        D_gate, S_gate = self._get_gate_params()
        gate_parameters = {}

        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                # Displacement gate parameters (Real and Imaginary parts)
                label_alpha = f"alpha_block{block}_q{q}"
                gate_parameters[label_alpha] = D_gate[block].detach()

                if block < self.n_blocks:
                    # SNAP gate phase parameters (n_learnable_phases per block starting from the corresponding energy level d)
                    # Initialize all phases to zero
                    for level in range(self.d):
                        label_theta = f"theta_block{block}_q{q}_level{level}"
                        gate_parameters[label_theta] = torch.zeros(1, dtype = torch.float32, device = self.device)
                    
                    # Now assign only the learnable phases
                    learnable_levels = self._learnable_levels_per_block[block]
                    
                    for idx, level in enumerate(learnable_levels):
                        label_theta = f"theta_block{block}_q{q}_level{level}"
                        gate_parameters[label_theta] = S_gate[block, idx].detach()
                        
        return gate_parameters


    def _get_learnable_levels(self, block):
        return [(block + i) % self.d for i in range(self.width)]


    def plot_learned_phase_locations(self, save_path):
        """
        Visualize which (fock level, block) positions are controlled by trainable phases.
        Optionally save the plot to file.
        """
        grid = torch.zeros((self.d, self.n_blocks), dtype=torch.int32)
        for block in range(self.n_blocks):
            indices = self._learnable_levels_per_block[block]
            for k in indices:
                grid[k, block] = 1

        plt.figure(figsize=(10, 6))
        plt.imshow(grid.numpy(), aspect='auto', cmap='Greys', interpolation='none')
        plt.xlabel("Block index b")
        plt.ylabel("Fock state index k")
        plt.title("Learned Phase Locations")
        plt.colorbar(label="Phase learned (1 = yes)")
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(save_path)


    def run_state(self,
            initial_state: torch.Tensor,
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
            initial_state = initial_state.reshape(-1, 1)

        final_state = initial_state.clone()

        D_gate, S_gate = self._get_gate_params()
        self._cached_gate_params = (D_gate, S_gate)

        for block in range(self.n_blocks + 1):  # Apply multiple layers
            for q in range(self.n_qudits):
                alpha = D_gate[block].unsqueeze(0)  # Compilation require this to be a (1,) tensor
                D = displacement_gate_torch(alpha, self.d + self.n_bumpers, device = self.device)
                final_state = torch.matmul(D, final_state)

                # The truncated displacement is not exactly unitary, renormalize
                final_state = final_state / torch.linalg.vector_norm(final_state, ord = 2)

                if block < self.n_blocks:
                    theta_list = torch.zeros(self.d, dtype = torch.float32, device = self.device)
                    
                    learnable_levels = self._learnable_levels_per_block[block]
                    theta_list[learnable_levels] = S_gate[block, :len(learnable_levels)]        #Different blocks may have different numbers of learnable phases
                    
                    S = SNAP_gate_torch_AllInOne(theta_list, n_bumpers = self.n_bumpers)

                    final_state = torch.matmul(S, final_state)

        return final_state.squeeze()


    def run_unitary(self) -> torch.Tensor:
        """
        Constructs the full unitary matrix corresponding to the SNAP-displacement sequence.
        It starts from the identity and sequentially applies the gates: D_0, S_0, D_1, ..., S_{n-1}, D_n.

        Returns
        -------
        total_unitary : torch.Tensor
            The full unitary operator as a (d + n_bumpers, d + n_bumpers) complex matrix.
        """

        dim = self.d + self.n_bumpers
        total_unitary = torch.eye(dim, dtype=torch.complex64, device=self.device)

        D_gate, S_gate = self._get_gate_params()

        for block in range(self.n_blocks + 1):
            for q in range(self.n_qudits):
                alpha = D_gate[block].unsqueeze(0)
                D = displacement_gate_torch(alpha, dim, device=self.device)
                total_unitary = torch.matmul(D, total_unitary)

                # Apply SNAP if not final block
                if block < self.n_blocks:
                    theta_list = torch.zeros(self.d, dtype = torch.float32, device = self.device)

                    learnable_levels = self._learnable_levels_per_block[block]
                    theta_list[learnable_levels] = S_gate[block, :len(learnable_levels)]        #Different blocks may have different numbers of learnable phases

                    S = SNAP_gate_torch_AllInOne(theta_list, n_bumpers = self.n_bumpers)
                    total_unitary = torch.matmul(S, total_unitary)

        return total_unitary




class SNAPnEquispaced_Displacement(SNAPn_Displacement):
    """
    SNAP Displacement protocol where n trainable phases are equally spaced across d Fock states.
    The positions shift as the block index increases.
    Inherits from SNAPn_Displacement but overrides the indexing logic.
    """

    def __init__(self, d, n_bumpers, n_blocks, n_learnable_phases, device):
        assert n_learnable_phases >= 1, f"SNAPnEquispaced_Displacement: The number of phases must be >= 1, the value was {n_learnable_phases}"
        self.n_learnable_phases = n_learnable_phases
        super(SNAPnEquispaced_Displacement, self).__init__(d, n_bumpers, n_blocks, None, device)

    def _get_learnable_levels(self, block):
        """
        Compute n equispaced indices over d, shifted by block index.
        """
        step = self.d / self.n_learnable_phases
        shifted_indices = [int(block + i * step) % self.d for i in range(self.n_learnable_phases)]
        return shifted_indices



class SNAPnInterpolated_Displacement(SNAPn_Displacement):
    """
    SNAP Displacement protocol where n trainable phases are
        Block 0: Learnable SNAP phases at equispaced positions.
        Block N-1 (last): Learnable phases at contiguous positions centered in the Fock space.
        Intermediate blocks: Phase positions that linearly interpolate from the equispaced layout to the centered contiguous one.

    Inherits from SNAPn_Displacement but overrides the indexing logic.
    """

    def __init__(self, d, n_bumpers, n_blocks, n_learnable_phases, device):
        assert n_learnable_phases >= 2, f"SNAPnInterpolated_Displacement: The number of phases must be >= 2, the value was {n_learnable_phases}"
        self.n_learnable_phases = n_learnable_phases
        self.precomputed_learnable_levels = None
        super(SNAPnInterpolated_Displacement, self).__init__(d, n_bumpers, n_blocks, n_learnable_phases, device)

    def _get_learnable_levels(self, block):

        if self.precomputed_learnable_levels is None:
            self.precomputed_learnable_levels = self._compute_interpolated_indices()

        return self.precomputed_learnable_levels[block]

    def _compute_interpolated_indices(self):
        """
        Computes interpolated SNAP indices from equispaced to centered contiguous.
        Returns a list of lists: indices[block] = [pos0, pos1, ..., pos{n-1}]
        """
        indices_per_block = []

        # Start: equispaced, first in 0 and last in d-1
        equispaced = [round(i * (self.d - 1) / (self.n_learnable_phases - 1)) for i in range(self.n_learnable_phases)]

        # End: centered
        center = self.d // 2
        contiguous = [(center - self.n_learnable_phases // 2 + i) % self.d for i in range(self.n_learnable_phases)]

        for b in range(self.n_blocks):
            frac = b / (self.n_blocks - 1) if self.n_blocks > 1 else 0
            interpolated = [
                int(round((1 - frac) * start + frac * end)) % self.d
                for start, end in zip(equispaced, contiguous)
            ]
            indices_per_block.append(interpolated)

        return indices_per_block





class SNAPnRotatedGrid_Displacement(SNAPn_Displacement):
    """
    SNAP Displacement protocol where n trainable phases are in grid along the diagonals
    Inherits from SNAPn_Displacement but overrides the indexing logic.
    """

    def __init__(self, d, n_bumpers, n_blocks, n_learnable_phases, device):
        assert n_learnable_phases >= 1, f"SNAPnDiagonalGrid_Displacement: The number of phases must be >=1, the value was {n_learnable_phases}"

        self.n_learnable_phases = n_learnable_phases
        self.d = d
        self.n_blocks = n_blocks
        self.precomputed_learnable_levels = self._compute_rotated_grid_indices()

        super(SNAPnRotatedGrid_Displacement, self).__init__(d, n_bumpers, n_blocks, None, device)

    def _get_learnable_levels(self, block):

        if self.precomputed_learnable_levels is None:
            self.precomputed_learnable_levels = self._compute_rotated_grid_indices()

        return self.precomputed_learnable_levels[block]

    def _compute_rotated_grid_indices(self):
        """
        Returns a list of phase indices per block following a 45-degree rotated grid.
        Spacing is computed from d and n_learnable_phases to control density.
        """
        max_candidates = 2 * self.d  # Rough upper bound on number of candidate positions per block
        spacing = max(1, int(max_candidates / self.n_learnable_phases))
        
        phase_indices = []
        for b in range(self.n_blocks):
            indices = []
            for k in range(self.d):
                if (k - b) % spacing == 0 or (k + b) % spacing == 0:
                    indices.append(k)
            indices = sorted(indices)
            phase_indices.append(indices)
        
        return phase_indices





############################################################################################################################################
##########################################
##########################################      New cleaned ansatze
##########################################
############################################################################################################################################


class SNAPAdaptiveDiagonal_Displacement(SNAPn_Displacement):
    """
    SNAP Displacement ansatz where each block controls one phase aligned along the logical diagonal
    from (0, 0) to (d, n_blocks). This ensures exactly one learnable phase per block, mapped such that:

        k_b = round(b * d / n_blocks)

    where k_b is the Fock level controlled by block b. Works even when d ≠ n_blocks.
    """

    def __init__(self, d, n_bumpers, n_blocks, width, device):
        assert width >= 1, f"SNAPAdaptiveDiagonal_Displacement: The diagonal width must be >=1, the value was {width}"
        super(SNAPAdaptiveDiagonal_Displacement, self).__init__(d, n_bumpers, n_blocks, width=width, device=device)

    def _get_learnable_levels(self, block):
        """
        Return the single Fock level for the block on the diagonal path.
        Defined as: k = round(block * d / n_blocks)
        """
        center = round(block * self.d / self.n_blocks)
        center = min(self.d - 1, max(0, center))
        half_width = self.width // 2
        indices = [center + offset for offset in range(-half_width, half_width + 1) ]
        return [i for i in indices if 0 <= i < self.d]



class SNAPFixedDiagonal_Displacement(SNAPn_Displacement):
    """
    SNAP Displacement ansatz where each block controls exactly one phase at the fixed Fock level equal to the block index.
    If the block index exceeds the Fock dimension d, the index is wrapped via modulo d:

        k_b = block % d

    This implementation is independent of the ratio between n_blocks and d.
    """

    def __init__(self, d, n_bumpers, n_blocks, width, device):
        assert width >= 1, f"SNAPFixedDiagonal_Displacement: The diagonal width must be >=1, the value was {width}"
        super(SNAPFixedDiagonal_Displacement, self).__init__(d, n_bumpers, n_blocks, width=width, device=device)

    def _get_learnable_levels(self, block):
        """
        Return the single Fock level for the block index itself, modulo d if necessary.
        """
        center = block % self.d
        half_width = self.width // 2
        indices = [center + offset for offset in range(-half_width, half_width + 1)]
        return [i for i in indices if 0 <= i < self.d]



class SNAPAdaptiveMultiDiagonal_Displacement(SNAPAdaptiveDiagonal_Displacement):
    """
    SNAPAdaptiveDiagonal variant with multiple equispaced diagonal bands.
    Each diagonal uses the same central logic from the parent class, shifted by a fixed spacing.
    """

    def __init__(self, d, n_bumpers, n_blocks, width, n_diagonals, device):
        self.n_diagonals = n_diagonals
        super(SNAPAdaptiveMultiDiagonal_Displacement, self).__init__(d, n_bumpers, n_blocks, width=width, device=device)

    def _get_learnable_levels(self, block):
        learnable_indices = set()
        slope = self.d / self.n_blocks

        for i in range(self.n_diagonals):
            # Use midpoint spacing along anti-diagonal
            t = (i + 0.5) / self.n_diagonals
            b_i = t * self.n_blocks
            k_i = (1 - t) * self.d

            center = round(k_i + (block - b_i) * slope)
            half_width = self.width // 2

            for offset in range(-half_width, half_width + 1):
                k = center + offset
                if 0 <= k < self.d:
                    learnable_indices.add(k)

        return sorted(learnable_indices)



class SNAPFixedMultiDiagonal_Displacement(SNAPFixedDiagonal_Displacement):
    """
    SNAPFixedDiagonal variant with multiple equispaced diagonal bands.
    Each diagonal uses the same fixed-modulo mapping from the parent class, shifted by fixed spacing.
    """

    def __init__(self, d, n_bumpers, n_blocks, width, n_diagonals, device):
        self.n_diagonals = n_diagonals
        super(SNAPFixedMultiDiagonal_Displacement, self).__init__(d, n_bumpers, n_blocks, width=width, device=device)

    def _get_learnable_levels(self, block):
        learnable_indices = set()
        slope = 1.0

        for i in range(self.n_diagonals):
            # Use midpoint spacing along anti-diagonal
            t = (i + 0.5) / self.n_diagonals
            b_i = t * self.n_blocks
            k_i = (1 - t) * self.d

            center = round(k_i + (block - b_i) * slope)
            half_width = self.width // 2

            for offset in range(-half_width, half_width + 1):
                k = center + offset

                k = k % self.d  # wrap around for fixed grid
                learnable_indices.add(k)

        return sorted(learnable_indices)



class SNAPAdaptiveGrid_Displacement(SNAPAdaptiveDiagonal_Displacement):
    """
    SNAPAdaptiveDiagonal variant with multiple equispaced diagonal bands.
    Each diagonal uses the same central logic from the parent class, shifted by a fixed spacing.
    """

    def __init__(self, d, n_bumpers, n_blocks, width, n_diagonals, device):
        self.n_diagonals = n_diagonals
        super(SNAPAdaptiveGrid_Displacement, self).__init__(d, n_bumpers, n_blocks, width=width, device=device)

    def _get_learnable_levels(self, block):
        learnable_indices = set()
        m = self.d / self.n_blocks
        half_width = self.width // 2

        # ↘ diagonals (equispaced along anti-diagonal)
        for i in range(self.n_diagonals):
            t = (i + 0.5) / self.n_diagonals
            k_i = (1 - t) * self.d
            b_i = t * self.n_blocks
            center = round(k_i + (block - b_i) * m)
            for offset in range(-half_width, half_width + 1):
                k = center + offset
                if 0 <= k < self.d:
                    learnable_indices.add(k)

        # ↗ off-diagonals (equispaced along main diagonal)
        for i in range(self.n_diagonals):
            t = (i + 0.5) / self.n_diagonals
            k_i = t * self.d
            b_i = t * self.n_blocks
            center = round(k_i - (block - b_i) * m)
            for offset in range(-half_width, half_width + 1):
                k = center + offset
                if 0 <= k < self.d:
                    learnable_indices.add(k)

        return sorted(learnable_indices)



class SNAPFixedGrid_Displacement(SNAPFixedDiagonal_Displacement):
    """
    SNAPFixedDiagonal variant with multiple equispaced diagonal bands.
    Each diagonal uses the same fixed-modulo mapping from the parent class, shifted by fixed spacing.
    """

    def __init__(self, d, n_bumpers, n_blocks, width, n_diagonals, device):
        self.n_diagonals = n_diagonals
        super(SNAPFixedGrid_Displacement, self).__init__(d, n_bumpers, n_blocks, width=width, device=device)

    def _get_learnable_levels(self, block):
        learnable_indices = set()
        m = 1.0
        half_width = self.width // 2

        # ↘ diagonals (equispaced along anti-diagonal)
        for i in range(self.n_diagonals):
            t = (i + 0.5) / self.n_diagonals
            k_i = (1 - t) * self.d
            b_i = t * self.n_blocks
            center = round(k_i + (block - b_i) * m)
            for offset in range(-half_width, half_width + 1):
                k = center + offset

                k = k % self.d  # wrap around for fixed grid
                learnable_indices.add(k)

        # ↗ off-diagonals (equispaced along main diagonal)
        for i in range(self.n_diagonals):
            t = (i + 0.5) / self.n_diagonals
            k_i = t * self.d
            b_i = t * self.n_blocks
            center = round(k_i - (block - b_i) * m)
            for offset in range(-half_width, half_width + 1):
                k = center + offset

                k = k % self.d  # wrap around for fixed grid
                learnable_indices.add(k)

        return sorted(learnable_indices)




class SNAPPolynomial_Displacement(SNAP_Displacement):
    """
    SNAP-Displacement ansatz where each SNAP gate is parametrized by a low-degree polynomial in the Fock level index.
    The phase at Fock level n in block b is given by:
        phi_n^(b) = theta_0^(b) + theta_1^(b) * n + theta_2^(b) * n^2 + ...

    This reduces the number of learnable parameters per block from d to polynomial_order + 1.
    """

    def __init__(self, d, n_bumpers, n_blocks, polynomial_order, device):
        self.polynomial_order = polynomial_order
        super().__init__(d, n_bumpers, n_blocks, device)

    def _init_parameters(self):
        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype=torch.float32, device=self.device))
        self.poly_coeffs = nn.Parameter(
            torch.rand((self.n_blocks, self.polynomial_order + 1), dtype=torch.float32, device=self.device)
        )

    def get_torch_parameters(self):
        return {"D_params": self.D_params, "poly_coeffs": self.poly_coeffs}

    def set_torch_parameters(self, param_dict):
        with torch.no_grad():
            for name in ["D_params", "poly_coeffs"]:
                param_tensor = torch.tensor(param_dict[name], dtype=torch.float32, device=self.device)
                getattr(self, name).data.copy_(param_tensor)

    def get_n_parameters(self):
        return self.D_params.numel() + self.poly_coeffs.numel()

    def _get_gate_params(self):
        D_radii = nn.ReLU()(self.D_params[:, 0])
        D_angles = nn.Tanh()(self.D_params[:, 1]) * torch.pi
        D_gate = polar_to_complex(D_radii, D_angles)

        # Compute phase vector from polynomial coefficients
        n = torch.arange(self.d, device=self.device).float()  # shape (d,)
        powers = torch.stack([n ** k for k in range(self.polynomial_order + 1)], dim=0)  # shape (K+1, d)

        # (n_blocks, K+1) @ (K+1, d) -> (n_blocks, d)
        raw_phases = torch.matmul(self.poly_coeffs, powers)
        S_gate = nn.Tanh()(raw_phases) * torch.pi

        return D_gate, S_gate


class SNAPSinusoidal_Displacement(SNAP_Displacement):
    """
    SNAP-Displacement ansatz where each SNAP gate applies a sum of sinusoidal phase profiles
    over Fock levels with learnable frequency, phase shift, and amplitude for each component.

    Phase pattern per block:
        phi_n^(b) = pi * sum_k A_k^(b) * sin(omega_k^(b) * n + theta_k^(b))

    The number of sinusoidal components (frequencies) is a hyperparameter.
    """

    def __init__(self, d, n_bumpers, n_blocks, n_frequencies, device):
        self.n_frequencies = n_frequencies
        super().__init__(d, n_bumpers, n_blocks, device)

    def _init_parameters(self):
        self.D_params = nn.Parameter(torch.rand((self.n_blocks + 1, 2), dtype=torch.float32, device=self.device))
        self.freq_params = nn.Parameter(torch.rand((self.n_blocks, self.n_frequencies), dtype=torch.float32, device=self.device))
        self.phase_params = nn.Parameter(torch.rand((self.n_blocks, self.n_frequencies), dtype=torch.float32, device=self.device))
        self.amplitude_params = nn.Parameter(torch.rand((self.n_blocks, self.n_frequencies), dtype=torch.float32, device=self.device))

    def get_torch_parameters(self):
        return {
            "D_params": self.D_params,
            "freq_params": self.freq_params,
            "phase_params": self.phase_params,
            "amplitude_params": self.amplitude_params
        }

    def set_torch_parameters(self, param_dict):
        with torch.no_grad():
            for name in ["D_params", "freq_params", "phase_params", "amplitude_params"]:
                param_tensor = torch.tensor(param_dict[name], dtype=torch.float32, device=self.device)
                getattr(self, name).data.copy_(param_tensor)

    def get_n_parameters(self):
        return (
            self.D_params.numel() +
            self.freq_params.numel() +
            self.phase_params.numel() +
            self.amplitude_params.numel()
        )

    def _get_gate_params(self):
        D_radii = nn.ReLU()(self.D_params[:, 0])
        D_angles = nn.Tanh()(self.D_params[:, 1]) * torch.pi
        D_gate = polar_to_complex(D_radii, D_angles)

        n = torch.arange(self.d, device=self.device).float().view(1, 1, -1)  # shape (1, 1, d)
        omegas = nn.Tanh()(self.freq_params).view(self.n_blocks, self.n_frequencies, 1) * math.pi
        phases = nn.Tanh()(self.phase_params).view(self.n_blocks, self.n_frequencies, 1) * math.pi
        amplitudes = nn.Sigmoid()(self.amplitude_params).view(self.n_blocks, self.n_frequencies, 1)

        # phi_n^(b) = sum_k A_k * sin(omega_k * n + theta_k)
        raw_phases = (amplitudes * torch.sin(omegas * n + phases)).sum(dim=1)  # shape (n_blocks, d)
        S_gate = nn.Tanh()(raw_phases) * math.pi

        return D_gate, S_gate



if __name__ == "__main__":
    # Example usage
    d = 16
    n_blocks = 16
    n_bumpers = 8
    width = 3
    device = "cpu"

    ansatz = SNAPFixedDiagonal_Displacement(d, n_bumpers, n_blocks, width=width, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPFixedDiagonal_Displacement.png")

    ansatz = SNAPAdaptiveDiagonal_Displacement(d, n_bumpers, n_blocks, width=width, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPAdaptiveDiagonal_Displacement.png")

    ansatz = SNAPAdaptiveMultiDiagonal_Displacement(d, n_bumpers, n_blocks, width=width, n_diagonals=4, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPAdaptiveMultiDiagonal_Displacement.png")

    ansatz = SNAPFixedMultiDiagonal_Displacement(d, n_bumpers, n_blocks, width=width, n_diagonals=4, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPFixedMultiDiagonal_Displacement.png")

    ansatz = SNAPAdaptiveGrid_Displacement(d, n_bumpers, n_blocks, width=width, n_diagonals=4, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPAdaptiveGrid_Displacement.png")

    ansatz = SNAPFixedGrid_Displacement(d, n_bumpers, n_blocks, width=width, n_diagonals=4, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPFixedGrid_Displacement.png")


    ansatz = SNAPFixedDiagonal_Displacement(d, n_bumpers, n_blocks*2, width=width, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPFixedDiagonal_Displacement_2.png")

    ansatz = SNAPAdaptiveDiagonal_Displacement(d, n_bumpers, n_blocks*2, width=width, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPAdaptiveDiagonal_Displacement_2.png")

    ansatz = SNAPAdaptiveMultiDiagonal_Displacement(d, n_bumpers, n_blocks*2, width=width, n_diagonals=3, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPAdaptiveMultiDiagonal_Displacement_2.png")

    ansatz = SNAPFixedMultiDiagonal_Displacement(d, n_bumpers, n_blocks*2, width=width, n_diagonals=3, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPFixedMultiDiagonal_Displacement_2.png")

    ansatz = SNAPAdaptiveGrid_Displacement(d, n_bumpers, n_blocks*2, width=width, n_diagonals=4, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPAdaptiveGrid_Displacement_2.png")

    ansatz = SNAPFixedGrid_Displacement(d, n_bumpers, n_blocks*2, width=width, n_diagonals=4, device=device)
    ansatz.plot_learned_phase_locations(save_path="SNAPFixedGrid_Displacement_2.png")








    ansatz = SNAPPolynomial_Displacement(d, n_bumpers, n_blocks, polynomial_order=4, device=device)

    F = fourier_gate_torch(d+n_bumpers, dtype = torch.complex64, device = device)

    initial_state = torch.zeros(d+n_bumpers, dtype = torch.complex64, device = device)
    initial_state[0] = 1.0
    target_state = F @ initial_state

    ansatz.run_state(initial_state)