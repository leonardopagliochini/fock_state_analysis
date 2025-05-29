#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/02/2025

@author: Maurizio Ferrari Dacrema
"""

import unittest, torch
import numpy as np
from quantum_circuit.qudit_gates import X_qudit_gate, csum_gate, hadamard_qudit_gate, Z_qudit_gate, Y_qudit_gate, annihilation_operator, symbolic_annihilation_operator, displacement_gate, SNAP_gate, symbolic_displacement_gate
from sympy import matrix2numpy

class TestQuditShiftOperatorStates(unittest.TestCase):

    def test_unitary(self):
        """Test the CSUM gate by applying it to different basis states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            print(f"Testing d={d}")
            X = X_qudit_gate(d)

            # Test unitarity: X * X^dagger = I
            identity = np.eye(d)
            np.testing.assert_array_almost_equal(
                X.dot(X.conj().T), identity, decimal=6, err_msg=f"Unitarity check failed for d={d}"
            )

    def test_X_qudit_gate(self):
        """Test the generalized X (shift) gate for various dimensions."""

        for d in range(2, 11):  # Test for different qudit dimensions
            print(f"Testing d={d}")
            X = X_qudit_gate(d)

            # Test its application to different basis states |n⟩
            for n in range(d):
                state = np.zeros((d, 1))
                state[n, 0] = 1  # |n⟩ state

                # Expected output: |(n+1) mod d⟩
                expected_state = np.zeros((d, 1))
                expected_state[(n + 1) % d, 0] = 1

                print(f"X checking transformation for d={d}, |{n}⟩ -> |{(n+1) % d}⟩")

                result_state = X.dot(state)

                np.testing.assert_array_almost_equal(
                    result_state, expected_state, decimal=6,
                    err_msg=f"Failed state transformation for d={d}, |{n}⟩ -> |{(n+1) % d}⟩"
                )


class TestCSUMGate(unittest.TestCase):

    def test_unitary(self):
        """Test the CSUM gate by applying it to different basis states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            print(f"Testing d={d}")
            CSUM = csum_gate(d)

            # Test unitarity: CSUM * CSUM^dagger = I
            identity = np.eye(d**2)
            np.testing.assert_array_almost_equal(
                CSUM.dot(CSUM.conj().T), identity, decimal=6, err_msg=f"Unitarity check failed for d={d}"
            )

    def test_csum_gate(self):
        """Test the CSUM gate by applying it to different basis states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            print(f"Testing d={d}")
            CSUM = csum_gate(d)

            # Test its application to different basis states CSUM|c,t⟩
            for c in range(d):
                for t in range(d):
                    # Create separate control and target qudit states
                    control_state = np.zeros((d, 1), dtype=complex)
                    target_state = np.zeros((d, 1), dtype=complex)

                    control_state[c, 0] = 1  # |c⟩ for control qudit
                    target_state[t, 0] = 1  # |t⟩ for target qudit

                    # Create full two-qudit state as tensor product |c⟩ ⊗ |t⟩
                    state = np.kron(control_state, target_state)

                    # Expected output: |c⟩ ⊗ |(c+t) mod d⟩
                    expected_target_state = np.zeros((d, 1), dtype=complex)
                    expected_target_state[(c + t) % d, 0] = 1  # Modular addition

                    expected_state = np.kron(control_state, expected_target_state)

                    result_state = CSUM.dot(state)

                    print(f"CSUM checking transformation for d={d}, |{c},{t}⟩ -> |{c},{(c+t) % d}⟩")

                    np.testing.assert_array_almost_equal(
                        result_state, expected_state, decimal=6,
                        err_msg=f"CSUM transformation failed for d={d}, |{c},{t}⟩ -> |{c},{(c+t) % d}⟩"
                    )



class TestHadamardGate(unittest.TestCase):

    def test_unitary(self):
        """Test the CSUM gate by applying it to different basis states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            print(f"Testing d={d}")
            H = hadamard_qudit_gate(d)

            # Test unitarity: H * H^dagger = I
            identity = np.eye(d)
            np.testing.assert_array_almost_equal(
                H.dot(H.conj().T), identity, decimal=6, err_msg=f"Unitarity check failed for d={d}"
            )

    def test_hadamard_gate(self):
        """Test the generalized Hadamard gate for different dimensions."""

        for d in range(2, 11):  # Test for different qudit dimensions
            H = hadamard_qudit_gate(d)

            # Test action on different computational basis states |n⟩
            for n in range(d):
                state = np.zeros((d, 1), dtype=complex)
                state[n, 0] = 1  # |n⟩ state

                # Compute expected Fourier transform manually
                expected_state = np.array([np.exp(2j * np.pi * m * n / d) for m in range(d)]).reshape((d, 1)) / np.sqrt(d)

                result_state = H.dot(state)

                print(f"\nHadamard Gate (d={d}) applied to |{n}⟩:")
                print(result_state)

                self.assertTrue(
                    np.allclose(result_state, expected_state, atol=1e-6),
                    msg=f"Hadamard transformation on |{n}⟩ failed for d={d}"
                )


class TestZGate(unittest.TestCase):

    def test_unitary(self):
        """Test the CSUM gate by applying it to different basis states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            Z = Z_qudit_gate(d)

            # Test unitarity: Z * Z^dagger = I
            identity = np.eye(d)
            np.testing.assert_array_almost_equal(
                Z.dot(Z.conj().T), identity, decimal=6, err_msg=f"Unitarity check failed for d={d}"
            )

    def test_z_gate(self):
        """Test the generalized Z gate for different dimensions and print results."""

        for d in range(2, 11):  # Test for different qudit dimensions
            Z = Z_qudit_gate(d)

            # Test action on different computational basis states |n⟩
            for n in range(d):
                state = np.zeros((d, 1), dtype=complex)
                state[n, 0] = 1  # |n⟩ state

                # Expected output: e^(2πi n / d) |n⟩
                expected_state = np.exp(2j * np.pi * n / d) * state

                result_state = Z.dot(state)

                print(f"\nZ Gate (d={d}) applied to |{n}⟩:")
                print(result_state)

                self.assertTrue(
                    np.allclose(result_state, expected_state, atol=1e-6),
                    msg=f"Z transformation on |{n}⟩ failed for d={d}"
                )


class TestYGate(unittest.TestCase):

    def test_unitary(self):
        """Test the CSUM gate by applying it to different basis states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            Y = Y_qudit_gate(d)

            # Test unitarity: Y * Y^dagger = I
            identity = np.eye(d)
            np.testing.assert_array_almost_equal(
                Y.dot(Y.conj().T), identity, decimal=6, err_msg=f"Unitarity check failed for d={d}"
            )

    def test_y_gate(self):
        """Test the generalized Y gate for different dimensions and print results."""

        for d in range(2, 11):  # Test for different qudit dimensions
            Y = Y_qudit_gate(d)

            # Test action on different computational basis states |n⟩
            for n in range(d):
                state = np.zeros((d, 1), dtype=complex)
                state[n, 0] = 1  # |n⟩ state

                # Expected output: i * e^(2πi n / d) |(n+1) mod d⟩
                expected_state = np.zeros((d, 1), dtype=complex)
                expected_state[(n + 1) % d, 0] = 1j * np.exp(2j * np.pi * n / d)

                result_state = Y.dot(state)

                print(f"\nY Gate (d={d}) applied to |{n}⟩:")
                print(result_state)

                self.assertTrue(
                    np.allclose(result_state, expected_state, atol=1e-6),
                    msg=f"Y transformation on |{n}⟩ failed for d={d}"
                )


def number_operator(d):
    """
    Creates the number operator N directly using its definition.

    Parameters:
    d (int): Dimension of the Hilbert space (truncation of the Fock space).

    Returns:
    np.ndarray: The number operator matrix of size (d, d).
    """
    N = np.diag(np.arange(d))  # Construct diagonal matrix with elements 0, 1, 2, ..., d-1

    return N

import sympy as sp

class TestAnnihilationCreationOperators(unittest.TestCase):

    def test_annihilation_operator(self):
        """Test the annihilation operator by applying it to multiple states, including |0⟩."""

        for d in range(2, 11):  # Test for different qudit dimensions
            a = annihilation_operator(d)

            # Test |0⟩ state
            zero_state = np.zeros((d, 1))
            zero_state[0, 0] = 1  # |0⟩ state

            expected_zero_state = np.zeros((d, 1))  # a |0⟩ = 0

            result_zero_state = a.dot(zero_state)

            print(f"\nAnnihilation operator applied to |0⟩ (d={d}):")
            print(result_zero_state.flatten())

            self.assertTrue(
                np.allclose(result_zero_state, expected_zero_state, atol=1e-6),
                msg=f"Annihilation operator failed for |0⟩ in d={d}"
            )

            # Test other valid states (n >= 1)
            for n in range(1, d):
                state = np.zeros((d, 1))
                state[n, 0] = 1  # |n⟩ state

                expected_state = np.zeros((d, 1))
                expected_state[n-1, 0] = np.sqrt(n)  # a |n⟩ = sqrt(n) |n-1⟩

                result_state = a.dot(state)

                print(f"\nAnnihilation operator applied to |{n}⟩ (d={d}):")
                print(result_state.flatten())

                self.assertTrue(
                    np.allclose(result_state, expected_state, atol=1e-6),
                    msg=f"Annihilation operator failed for |{n}⟩ in d={d}"
                )


    def test_creation_operator(self):
        """Test the creation operator by applying it to multiple states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            a_dagger = annihilation_operator(d).T  # Creation operator is the Hermitian conjugate

            for n in range(d-1):  # Test only valid states (n < d-1)
                state = np.zeros((d, 1))
                state[n, 0] = 1  # |n⟩ state

                expected_state = np.zeros((d, 1))
                expected_state[n+1, 0] = np.sqrt(n+1)  # a^† |n⟩ = sqrt(n+1) |n+1⟩

                result_state = a_dagger.dot(state)

                print(f"\nCreation operator applied to |{n}⟩ (d={d}):")
                print(result_state.flatten())

                self.assertTrue(
                    np.allclose(result_state, expected_state, atol=1e-6),
                    msg=f"Creation operator failed for |{n}⟩ in d={d}"
                )


    def test_numerical_vs_symbolic(self):
        """Tests if the numerical and symbolic annihilation operators match for different dimensions."""
        for d in range(2, 11):
            a_numeric = annihilation_operator(d)
            a_symbolic = np.array(symbolic_annihilation_operator(d)).astype(np.float64)  # Convert symbolic to numeric

            # Check if they match within a small tolerance
            np.testing.assert_array_almost_equal(a_numeric, a_symbolic, decimal=10)
            np.testing.assert_array_almost_equal(a_numeric.T, a_symbolic.T, decimal=10)


    def test_commutation_relation(self):
        """Test that [a, a†] = I for large d."""
        for d in range(2, 11):  # Test for different qudit dimensions
            a = annihilation_operator(d)
            a_dagger = a.conj().T
            identity = np.eye(d, dtype=complex)
            commutator = a.dot(a_dagger) - a_dagger.dot(a)

            print(f"\nCommutation relation [a, a†] for d={d}:")
            print(commutator)

            # Check commutator holds for all elements except the last diagonal one
            expected_commutator = np.copy(identity)
            expected_commutator[d-1, d-1] = -(d-1)  # Correct the last element

            self.assertTrue(
                np.allclose(commutator, expected_commutator, atol=1e-6),
                msg=f"Commutation relation failed for d={d} (excluding last diagonal element)"
            )

    def test_commutation_relation_symbolic(self):
        """Tests the commutator [a, a†] = I for different dimensions using symbolic operators."""
        for d in range(2, 11):
            # Get symbolic annihilation and creation operators
            a_sym = symbolic_annihilation_operator(d)
            a_dagger_sym = a_sym.T  # The creation operator is the transpose of annihilation

            # Compute the commutator: [a, a†] = a a† - a† a
            commutator = a_sym * a_dagger_sym - a_dagger_sym * a_sym

            # Expected identity matrix
            identity_matrix = sp.eye(d)
            identity_matrix[d-1, d-1] = -(d-1)

            # Assert symbolic equality
            self.assertTrue(sp.simplify(commutator - identity_matrix) == sp.zeros(d, d),
                            f"Commutation relation failed for d={d}")


    def test_number_operator_construction(self):
        """Test that the number operator is correctly constructed as N = a† a."""

        for d in range(2, 11):  # Test for different qudit dimensions

            # Compute the number operator as a† a
            a = annihilation_operator(d)
            a_dagger = a.conj().T
            computed_N = a_dagger.dot(a)

            # Compute the expected number operator directly
            expected_N = number_operator(d)

            print("\nComputed Number Operator (N = a† a):")
            print(computed_N)

            print("\nExpected Number Operator:")
            print(expected_N)

            # Compare both matrices element-wise
            self.assertTrue(
                np.allclose(computed_N, expected_N, atol=1e-6),
                msg="Number operator computation failed: N != a† a"
            )











class TestDisplacementGate(unittest.TestCase):

    def test_unitary(self):
        """Test the Displacement gate by applying it to different basis states."""

        for d in range(2, 11):  # Test for different qudit dimensions
            for alpha in [0, 0.5, 1j, 1+1j, 5+5j]:  # Test different displacements, including large values
                D = displacement_gate(alpha, d)

                # Test unitarity: D * D^dagger = I
                identity = np.eye(d, dtype=complex)
                np.testing.assert_array_almost_equal(
                    D.dot(D.conj().T), identity, decimal=6, err_msg=f"Unitarity check failed for d={d}, alpha={alpha}"
                )

    def test_displacement_gate(self):
        """Test the displacement gate for different values of alpha and dimensions."""

        for d in range(2, 11):  # Test for different qudit dimensions
            for alpha in [0, 0.5, 1j, 1+1j, 5+5j]:  # Test different displacements, including large values
                D = displacement_gate(alpha, d)

                # Test action on vacuum state |0⟩
                vacuum_state = np.zeros((d, 1), dtype=complex)
                vacuum_state[0, 0] = 1  # |0⟩ state

                result_state = D.dot(vacuum_state)

                print(f"\nDisplacement Gate D({alpha}) for d={d} applied to |0⟩:")
                print(result_state.flatten())

                self.assertTrue(
                    np.linalg.norm(result_state) - 1 < 1e-6,
                    msg=f"State norm is not preserved for d={d}, alpha={alpha}"
                )

    def test_displacement_gate_amplitudes(self):
        """Test the displacement gate by explicitly checking the amplitudes of the resulting coherent state."""

        test_cases = [
            (1 + 1j, 20),  # The truncation works well only for large d
        ]

        for alpha, d in test_cases:
            D = displacement_gate(alpha, d)

            # Initialize vacuum state |0⟩
            vacuum_state = np.zeros((d, 1), dtype=complex)
            vacuum_state[0, 0] = 1  # |0⟩ state

            # Apply the displacement gate to the vacuum state
            computed_state = D.dot(vacuum_state)

            # Compute the expected amplitudes using the coherent state expansion
            expected_amplitudes = np.array([
                np.exp(-np.abs(alpha)**2 / 2) * (alpha**n / np.sqrt(np.math.factorial(n)))
                for n in range(d)
            ])

            print(f"\nTesting Displacement Gate with α={alpha}, d={d}:")
            print("Computed State Amplitudes:")
            print(computed_state.flatten())
            print("Expected State Amplitudes:")
            print(expected_amplitudes.flatten())

            # Compare absolute values to account for global phase differences
            self.assertTrue(
                np.allclose(np.abs(computed_state.flatten()), np.abs(expected_amplitudes), atol=1e-6),
                msg=f"Amplitude mismatch for α={alpha}, d={d}"
            )


    # def test_displacement_commutation(self):
    #     """Test that D†(alpha) a D(alpha) = a + alpha I for large d."""
    #     d = 100  # Increased dimension for better accuracy
    #     for alpha in [0.5, 1 + 1j, -0.3 + 0.8j]:
    #         a = annihilation_operator(d)
    #         D = displacement_gate(alpha, d)
    #         D_dagger = np.conjugate(D.T)
    #
    #         transformed_a = D_dagger.dot(a).dot(D)  # D† a D
    #         expected_a = a + alpha * np.eye(d, dtype=complex)
    #
    #         print(f"\nDisplacement commutation test for α={alpha}, d={d}:")
    #         print("Transformed a:")
    #         print(transformed_a[:5, :5])  # Print a small section for readability
    #         print("Expected a + αI:")
    #         print(expected_a[:5, :5])
    #
    #         self.assertTrue(
    #             np.allclose(transformed_a, expected_a, atol=1e-6),
    #             msg=f"Displacement operator commutation relation failed for α={alpha}, d={d}"
    #         )


    def test_coherent_state_preservation(self):
        """Test that a D(alpha) |0⟩ = alpha D(alpha) |0⟩ for large d."""
        d = 50  # Increased dimension for better accuracy
        for alpha in [0.5, 1 + 1j, -0.3 + 0.8j]:
            a = annihilation_operator(d)
            D = displacement_gate(alpha, d)
            vacuum_state = np.zeros((d, 1), dtype=complex)
            vacuum_state[0, 0] = 1  # |0⟩ state

            displaced_state = D.dot(vacuum_state)  # |α⟩ = D(α) |0⟩
            transformed_state = a.dot(displaced_state)  # a |α⟩
            expected_state = alpha * displaced_state  # α |α⟩

            print(f"\nCoherent state test for α={alpha}, d={d}:")
            print("Computed a |α⟩:")
            print(transformed_state.flatten()[:10])  # Print the first 10 elements
            print("Expected α |α⟩:")
            print(expected_state.flatten()[:10])

            self.assertTrue(
                np.allclose(transformed_state, expected_state, atol=1e-6),
                msg=f"Coherent state preservation failed for α={alpha}, d={d}"
            )


    #
    # def test_factorize_displacement_gate(self):
    #     """Test that the symbolic displacement gate matches the numerical one after substitution."""
    #     for d in range(2, 11):
    #         for alpha_val in [0.1, 0.5 + 0.1j, -0.8j, 1.2 + 0.5j]:
    #             D = displacement_gate(alpha_val, d)
    #             D_fact = displacement_gate_torch_factorized(torch.tensor(alpha_val, dtype=torch.complex64), d)
    #
    #             np.testing.assert_allclose(
    #                 D, D_fact, rtol = 1e-6, atol = 1e-6,
    #                 err_msg = f"Factorized and regular displacement gates differ for α={alpha_val}, d={d}"
    #             )


    # def test_symbolic_vs_numerical_displacement_gate(self):
    #     """Test that the symbolic displacement gate matches the numerical one after substitution."""
    #     for d in range(2, 11):
    #         alpha_sym = sp.Symbol('alpha', complex=True)
    #         D_sym = symbolic_displacement_gate(alpha_sym, d)
    #
    #         for alpha_val in [0.1, 0.5 + 0.1j, -0.8j, 1.2 + 0.5j]:
    #             # Substitute numeric value into symbolic matrix
    #             D_sym_evaluated = D_sym.subs(alpha_sym, alpha_val).evalf()
    #             D_sym_np = np.array(D_sym_evaluated.tolist()).astype(np.complex128)
    #             #D_sym_np = matrix2numpy(D_sym_evaluated, dtype=np.complex128)
    #
    #             # Compare with numerical implementation
    #             D_num = displacement_gate(alpha_val, d)
    #
    #             np.testing.assert_allclose(
    #                 D_sym_np, D_num, rtol=1e-6, atol=1e-6,
    #                 err_msg=f"Symbolic and numerical displacement gates differ for α={alpha_val}, d={d}"
    #             )
    #











class TestSNAPGate(unittest.TestCase):

    def test_snap_unitary(self):
        """Test if the SNAP gate is unitary."""
        for d in range(2, 11):
            theta_list = np.random.uniform(0, 2*np.pi, d)  # Random phases
            S = SNAP_gate(theta_list, d)

            identity = np.eye(d, dtype=complex)
            self.assertTrue(
                np.allclose(S.dot(S.conj().T), identity, atol=1e-6),
                msg=f"SNAP gate is not unitary for d={d}"
            )

    def test_snap_commutativity(self):
        """Test that SNAP gates commute when applied sequentially."""
        for d in range(2, 11):
            theta_list_1 = np.random.uniform(0, 2 * np.pi, d)  # Random phases
            theta_list_2 = np.random.uniform(0, 2 * np.pi, d)  # Another random phase set

            S1 = SNAP_gate(theta_list_1, d)
            S2 = SNAP_gate(theta_list_2, d)

            # Verify that applying in sequence is equivalent to summing phase shifts
            S_combined = SNAP_gate(np.array(theta_list_1) + np.array(theta_list_2), d)

            self.assertTrue(
                np.allclose(S1.dot(S2), S_combined, atol=1e-6),
                msg=f"SNAP gate commutativity failed for d={d}"
            )

    def test_snap_application(self):
        """Test the effect of the SNAP gate on a sample state."""
        for d in range(2, 11):
            theta_list = np.random.uniform(0, 2*np.pi, d)  # Random phases
            S = SNAP_gate(theta_list, d)

            # Define a random quantum state
            state = np.random.rand(d, 1) + 1j * np.random.rand(d, 1)
            state /= np.linalg.norm(state)  # Normalize the state

            transformed_state = S.dot(state)
            expected_state = np.exp(1j * np.array(theta_list).reshape(d, 1)) * state

            self.assertTrue(
                np.allclose(transformed_state, expected_state, atol=1e-6),
                msg=f"SNAP gate application failed for d={d}"
            )




class TestFourierGateTorch(unittest.TestCase):

    def test_unitarity(self):
        """Test that the Fourier gate is unitary for various dimensions."""
        from quantum_circuit.qudit_gates import fourier_gate_torch

        for d in range(2, 11):
            F = fourier_gate_torch(d)
            identity = torch.eye(d, dtype=F.dtype, device=F.device)
            product = F @ F.conj().T
            self.assertTrue(
                torch.allclose(product, identity, atol=1e-6),
                msg=f"Fourier gate is not unitary for d={d}"
            )

    def test_fourier_basis_vectors(self):
        """Check Fourier transform of each basis state against the known formula."""
        from quantum_circuit.qudit_gates import fourier_gate_torch

        for d in range(2, 11):
            F = fourier_gate_torch(d)
            for n in range(d):
                state = torch.zeros((d, 1), dtype=torch.complex64)
                state[n, 0] = 1.0
                transformed = F @ state

                # Expected result: normalized exponential vector
                expected = torch.exp(2j * torch.pi * torch.arange(d).reshape(-1, 1) * n / d) / torch.sqrt(torch.tensor(d, dtype=torch.float32))
                expected = expected.to(dtype=torch.complex64)

                self.assertTrue(
                    torch.allclose(transformed, expected, atol=1e-6),
                    msg=f"Fourier gate failed on |{n}⟩ for d={d}"
                )



if __name__ == '__main__':

    # Run the unit tests
    unittest.main()

