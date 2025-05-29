#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/02/2025

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
from scipy.linalg import expm

def X_qudit_gate(d):
    """Creates the cyclic shift (mod d) operator for dimension d."""
    X = np.zeros((d, d))
    for i in range(d):
        X[(i + 1) % d, i] = 1  # The notation is: [destination state, input state]
    return X


def csum_gate(d):
    """Creates the CSUM gate for a qudit system of dimension d using the correct X operator."""
    CSUM = np.zeros((d**2, d**2))

    X = X_qudit_gate(d)  # X shift operator

    for m in range(d):
        control_matrix = np.zeros((d, d))
        control_matrix[m, m] = 1  # Projector on the control qudit

        block = np.linalg.matrix_power(X, m)  # Apply X^m on the target qudit
        CSUM += np.kron(control_matrix, block)

    return CSUM


def hadamard_qudit_gate(d):
    """Creates the generalized Hadamard (Fourier) gate for a qudit system of dimension d."""
    H = np.zeros((d, d), dtype=complex)

    for m in range(d):
        for n in range(d):
            H[m, n] = np.exp(2j * m * n * np.pi / d) / np.sqrt(d)

    return H


def Z_qudit_gate(d):
    """Creates the generalized Z (phase) gate for a qudit system of dimension d."""
    Z = np.zeros((d, d), dtype=complex)

    for n in range(d):
        Z[n, n] = np.exp(2j * n * np.pi / d)  # Apply phase based on state index

    return Z


def Y_qudit_gate(d):
    """Creates the generalized Y gate for a qudit system of dimension d."""
    X = X_qudit_gate(d)
    Z = Z_qudit_gate(d)

    # Generalized Y gate: Y = i * X * Z
    Y = 1j * X.dot(Z)

    return Y


def annihilation_operator(d):
    """Creates the annihilation operator a for a finite-dimensional qudit system."""
    a = np.zeros((d, d))

    for n in range(1, d):
        a[n-1, n] = np.sqrt(n)

    return a


import sympy as sp

def symbolic_annihilation_operator(d):
    """Construct symbolic creation (a†) and annihilation (a) operators in dimension d."""
    # Define symbolic matrix elements
    a = sp.zeros(d, d)

    for n in range(1, d):
        a[n-1, n] = sp.sqrt(n)  # Annihilation operator

    return a

def symbolic_displacement_gate(alpha, d):
    """
    Constructs a symbolic displacement gate D(alpha) for a qudit system of dimension d.

    Parameters:
        alpha (sp.Basic): A symbolic complex parameter (SymPy expression).
        d (int): Hilbert space dimension.

    Returns:
        sp.Matrix: Symbolic displacement gate as a matrix.
    """
    if not isinstance(alpha, (sp.Expr, sp.Symbol)):
        raise TypeError("alpha must be a SymPy symbolic expression, e.g., sp.Symbol('alpha', complex=True)")
    
    a = symbolic_annihilation_operator(d)
    
    adag = a.H  # Hermitian conjugate
    exponent = alpha * adag - sp.conjugate(alpha) * a

    return exponent.exp()


def symbolic_SNAP_gate(theta_vector, d, n_bumpers=0):
    """
    Constructs a symbolic SNAP gate as a diagonal matrix with phases exp(i * theta_k),
    vectorized and bumper-aware.
    """
    if not isinstance(theta_vector, sp.Matrix):
        raise TypeError("theta_vector must be a SymPy Matrix (column vector)")

    if theta_vector.shape != (d, 1):
        raise ValueError(f"theta_vector must be a column vector of shape ({d}, 1)")

    if not all(isinstance(theta_vector[n, 0], (sp.Expr, sp.Symbol)) for n in range(d)):
        raise TypeError("Each entry in theta_vector must be a SymPy symbolic expression")

    # Vectorized symbolic computation of e^{i theta}
    diag_entries = (sp.I * theta_vector).applyfunc(sp.exp)

    if n_bumpers > 0:
        bumper_matrix = sp.ones(n_bumpers, 1)
        diag_entries = diag_entries.col_join(bumper_matrix)

    return sp.diag(*diag_entries)




def displacement_gate(alpha, d):
    """Creates the displacement gate D(alpha) for a qudit system of dimension d."""
    a = annihilation_operator(d)

    return expm(alpha * a.T.conj() - np.conjugate(alpha) * a)  # Exponentiation


def SNAP_gate(phases, d, n_bumpers = 0):
    """Creates the SNAP gate for a qudit system of dimension d."""
    assert len(phases) == d, "Theta list length must match qudit dimension d."
    
    phases = np.exp(1j * np.array(phases))
    if n_bumpers > 0:
        phases = np.concatenate([phases, np.ones(n_bumpers)])

    return np.diag(phases)  # Diagonal phase shifts


import torch

# def annihilation_operator_torch(d, dtype=torch.complex32, device='cpu'):
#     """
#     Creates the annihilation operator 'a' for a finite-dimensional qudit system,
#     implemented in PyTorch.
#
#     Parameters
#     ----------
#     d : int
#         Dimension of the Hilbert space (truncation).
#     dtype : torch.dtype, optional
#         Data type to use for the operator. Defaults to torch.complex128.
#     device : str, optional
#         Device on which to create the operator (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
#
#     Returns
#     -------
#     a : torch.Tensor
#         The annihilation operator of shape (d, d), stored as a complex tensor.
#     """
#     a = torch.zeros((d, d), dtype=dtype, device=device)
#     # Fill the subdiagonal with sqrt(n)
#     for n in range(1, d):
#         a[n-1, n] = torch.sqrt(torch.tensor(float(n), dtype=dtype, device=device))
#     return a


def annihilation_operator_torch(d, dtype=torch.complex64, device='cpu'):
    """
    Creates the annihilation operator 'a' as a strictly lower diagonal matrix
    with sqrt(n) on the (n-1, n) positions.
    """
    # Prepare indices
    rows = torch.arange(0, d-1, device=device)
    cols = torch.arange(1, d, device=device)
    values = torch.sqrt(torch.arange(1, d, device=device)).to(dtype)

    a = torch.zeros((d, d), dtype=dtype, device=device)
    # Simpler but does not compile correctly: a[rows, cols] = values
    a = a.index_put((rows, cols), values, accumulate=False) # Compile-safe

    return a


def displacement_gate_torch(alpha, d, dtype=torch.complex64, device='cpu'):
    """
    Creates the displacement gate D(alpha) for a qudit system of dimension d, implemented in PyTorch.

    D(alpha) = exp(alpha * a^† - alpha^* * a)

    Parameters
    ----------
    alpha : complex
        Displacement parameter (can be real or complex).
    d : int
        Dimension of the Hilbert space (truncation).
    dtype : torch.dtype, optional
        Data type to use for the operator. Defaults to torch.complex128.
    device : str, optional
        Device on which to create the operator (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns
    -------
    D : torch.Tensor
        The displacement operator, shape (d, d), as a complex PyTorch tensor.
    """
    # Convert alpha into a torch complex tensor
    #alpha = torch.tensor(alpha, dtype=dtype, device=device)

    # Construct annihilation operator 'a'
    a = annihilation_operator_torch(d, dtype=dtype, device=device)
    a_dag = a.T.conj()

    # Create alpha^*, unfortunately the simple alpha.conj() is not compilable
    alpha_conj = torch.complex(alpha.real, -alpha.imag)

    # Build the matrix in the exponent:  alpha * a^† - (alpha^*) * a
    exponent_matrix = alpha * a_dag - alpha_conj * a
    
    # Exponentiate the matrix using PyTorch
    D = torch.matrix_exp(exponent_matrix)

    return D

#
# def displacement_gate_torch_factorized(alpha, d, device='cpu', dtype=torch.complex64):
#     """
#     More stable factorization for e^(alpha a^\dagger - alpha^* a).
#     D(alpha) = e^{-|alpha|^2/2} e^{alpha a^\dagger} e^{-alpha^* a}
#     in truncated dimension d.
#     Baker–Campbell–Hausdorff (BCH) Identity
#     """
#     # a is the annihilation operator
#     a = annihilation_operator_torch(d, dtype=dtype, device=device)
#     adag = a.conj().T
#
#     # Compute e^(alpha a^dagger) via a direct series approach or torch.matrix_exp.
#     # But since alpha*a^dagger is nilpotent, a direct series can be simpler and more stable for large alpha.
#
#     # Pre-factor:
#     prefactor = torch.exp(-0.5 * torch.abs(alpha)**2)
#
#     # Then exponentiate alpha a^dagger
#     A = alpha * adag
#     # Then exponentiate - alpha^*. conj(a)
#     B = -alpha.conj() * a
#
#     # Use torch.matrix_exp for both small A, B
#     # but watch for partial sums. A direct series approach is often used.
#     EA = torch.matrix_exp(A)
#     EB = torch.matrix_exp(B)
#
#     D = prefactor * EA.multiply(EB)
#
#     return D



def SNAP_gate_torch(theta_list, d, n_bumpers = 0, device='cpu'):
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
    assert len(theta_list) == d, "Theta list length must match qudit dimension d."
    assert n_bumpers >= 0, "Number of bumper states must be >= 0"
    
    # Convert phases to a torch tensor and exponentiate
    theta_list = torch.stack(theta_list, dim = 0).squeeze(-1)

    # This is a simple r = e^(i*theta)
    # torch.exp(1j * theta_list)
    # It is written like this because it is compilable by Triton
    phases = torch.exp(torch.complex(torch.zeros(theta_list.shape[0], device=theta_list.device), theta_list))
    
    if n_bumpers > 0:
        bumper = torch.ones(n_bumpers, dtype = phases.dtype, device = phases.device)
        phases = torch.cat([phases, bumper], dim = 0)
    
    # Construct diagonal matrix
    return torch.diag(phases)



def fourier_gate_torch(d, dtype=torch.complex64, device='cpu'):
    """
    Creates the generalized Fourier (Hadamard-like) gate for a qudit system of dimension d,
    implemented in PyTorch.

    The Fourier gate is defined as:
        F_{mn} = exp(2πi * m * n / d) / sqrt(d)

    Parameters
    ----------
    d : int
        Dimension of the Hilbert space.
    dtype : torch.dtype
        Data type of the returned tensor.
    device : str
        Torch device ('cpu' or 'cuda').

    Returns
    -------
    F : torch.Tensor of shape (d, d)
        The Fourier unitary operator.
    """
    m = torch.arange(d, device=device).reshape(-1, 1)  # column
    n = torch.arange(d, device=device).reshape(1, -1)  # row

    phase_argument = 2 * torch.pi * m * n / d
    real = torch.cos(phase_argument)
    imag = torch.sin(phase_argument)

    F = torch.complex(real, imag) / torch.sqrt(torch.tensor(d, dtype=torch.float32, device=device))
    return F.to(dtype)














import math

def photon_loss_kraus_operators(d, eta, ell_max, device='cpu', threshold=1e-7):
    """Construct Kraus operators for photon loss channel with binomial formulation."""
    a = annihilation_operator_torch(d, device=device)

    if eta < 1.0:
        eta_n_half = torch.diag(torch.pow(torch.tensor(eta, device=device), 0.5 * torch.arange(d, device=device))).to(torch.complex64)
        kraus_ops = []
        for ell in range(ell_max + 1):
            coeff = torch.sqrt(torch.tensor(((1 - eta) ** ell) / math.factorial(ell)))
            a_pow = torch.linalg.matrix_power(a, ell)
            K = coeff * eta_n_half @ a_pow

            # Filter by trace(K†K)
            contribution = torch.trace(K.conj().T @ K).real
            if contribution >= threshold:
                kraus_ops.append(K)

    else:
        kraus_ops = []

    return kraus_ops