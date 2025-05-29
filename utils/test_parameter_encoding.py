#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/02/2025

@author: Maurizio Ferrari Dacrema
"""


import unittest, torch
import numpy as np
from utils.parameter_encoding import RealImg_Phase_Mapping, Polar_Phase_Mapping
from utils.parameter_encoding_unbounded import RealImg_Phase_Mapping as RealImg_Phase_Mapping_unbounded, Polar_Phase_Mapping as Polar_Phase_Mapping_unbounded
from quantum_circuit.ansatz import SNAP_Displacement
#from run_differentiable_simulation import create_learnable_parameters

class TestRealImg_Phase_Mapping_unbounded(unittest.TestCase):
    
    def setUp(self):
        """
        Initializes an instance of RealImg_Phase_Mapping for testing.
        """
        
        n_qudits = 3  # Two qudits
        n_bumpers = 0
        d = 3  # Three energy levels per qudit
        n_blocks = 2  # Two blocks
        
        ansatz = SNAP_Displacement(d, n_qudits, n_bumpers, n_blocks)
        self.gate_parameter_types = ansatz.get_parameter_types()
        
        self.mapping = RealImg_Phase_Mapping_unbounded(self.gate_parameter_types)
    
    # def test_encoding_decoding_gradient(self):
    #     """
    #     Tests whether the PyTorch gradient correctly flows through the encoding and decoding process.
    #     """
    #
    #     learnable_mapped_parameters_dict, torch_parameters_for_optimizer_dict = create_learnable_parameters(self.mapping.get_learnable_parameter_types(), device = "cpu", dtype = torch.float32)
    #
    #     def _check_gradient_flow(starting_dict, learnable_tensors):
    #
    #         loss = sum(torch.abs(value) for value in starting_dict.values())
    #         loss.backward()
    #
    #         # Check if gradients exist for all original learnable parameters
    #         for label, value in learnable_tensors.items():
    #             self.assertIsNotNone(value.grad, f"Gradient not computed for {label}")
    #             self.assertTrue(torch.any(value.grad != 0), f"Gradient is zero for {label}")
    #             value.grad.zero_()  # Reset gradient for the next loss
    #
    #
    #
    #     _check_gradient_flow(torch_parameters_for_optimizer_dict, torch_parameters_for_optimizer_dict)
    #
    #     _check_gradient_flow(torch_parameters_for_optimizer_dict, torch_parameters_for_optimizer_dict)
    #
    #     _check_gradient_flow(learnable_mapped_parameters_dict, torch_parameters_for_optimizer_dict)
    #
    #     _check_gradient_flow(learnable_mapped_parameters_dict, torch_parameters_for_optimizer_dict)
    #
    #     # gate_parameters = self.mapping.encoded_to_gate_params(learnable_mapped_parameters_dict)
    #     # _check_gradient_flow(gate_parameters, torch_parameters_for_optimizer_dict)
    #     #
    #     # decoded_learnable_parameters = self.mapping.gate_params_to_encoded(learnable_mapped_parameters_dict)
    #     # _check_gradient_flow(decoded_learnable_parameters, torch_parameters_for_optimizer_dict)
        
        
    def test_encoding_decoding_consistency(self):
        """
        Tests whether the gate parameters can be transformed in learnable ones and then back
        """

        original_gate_parameter_values = {}
        for label, type in self.gate_parameter_types.items():
            if type == "angle":
                # Value is in range [-π, +π]
                value = torch.rand(1, dtype = torch.float)
                original_gate_parameter_values[label] = value * 2 * torch.pi - torch.pi
            else:
                # Complex number unbounded
                value = torch.rand(1, dtype = torch.cfloat)
                original_gate_parameter_values[label] = value
        
        learnable_values = self.mapping.gate_params_to_encoded(original_gate_parameter_values)
        decoded_gate_parameter_values = self.mapping.encoded_to_gate_params(learnable_values)
        
        # Ensure original and decoded parameter values match within a numerical tolerance
        gate_parameter_labels = list(self.gate_parameter_types.keys())
        np.testing.assert_almost_equal(np.array([original_gate_parameter_values[key] for key in gate_parameter_labels]),
                                       np.array([decoded_gate_parameter_values[key] for key in gate_parameter_labels]),
                                       decimal = 6,
                                       err_msg = "Decoding did not correctly reconstruct original values.")


class TestPolar_Phase_Mapping_unbounded(TestRealImg_Phase_Mapping_unbounded):
    
    def setUp(self):
        """
        Initializes an instance of Polar_Phase_Mapping for testing.
        """
        
        n_qudits = 3  # Two qudits
        n_bumpers = 0
        d = 3  # Three energy levels per qudit
        n_blocks = 2  # Two blocks
        
        ansatz = SNAP_Displacement(d, n_qudits, n_bumpers, n_blocks)
        self.gate_parameter_types = ansatz.get_parameter_types()
        
        self.mapping = Polar_Phase_Mapping_unbounded(self.gate_parameter_types)


class TestRealImg_Phase_Mapping(TestRealImg_Phase_Mapping_unbounded):

    def setUp(self):
        """
        Initializes an instance of RealImg_Phase_Mapping for testing.
        """

        n_qudits = 3    # Two qudits
        n_bumpers = 0
        self.d = 3      # Three energy levels per qudit
        n_blocks = 2    # Two blocks
        
        ansatz = SNAP_Displacement(self.d, n_qudits, n_bumpers, n_blocks)
        self.gate_parameter_types = ansatz.get_parameter_types()
        
        self.mapping = RealImg_Phase_Mapping(self.gate_parameter_types, d = self.d)
        
        
    def test_decoded_parameters_bounded(self):
        """
        Tests whether the learnable parameters are encoded within a bounded [-1, +1]
        """
        
        gate_parameter_values = {}
        for label, type in self.gate_parameter_types.items():
            if type == "angle":
                # Value is in range [-π, +π]
                value = torch.rand(1, dtype = torch.float)
                gate_parameter_values[label] = value * 2 * torch.pi - torch.pi
            else:
                # Value is in range [-d/sqrt(2), +d/sqrt(2)]
                value = torch.rand(1, dtype = torch.cfloat)
                gate_parameter_values[label] = value * 2 * self.d/np.sqrt(2) - self.d/np.sqrt(2) - self.d/np.sqrt(2)*1j
        
        decoded_learnable_values = self.mapping.gate_params_to_encoded(gate_parameter_values)
        decoded_learnable_values = np.array(list(decoded_learnable_values.values()))

        # Ensure decoded values are within expected range [-1,1]
        self.assertTrue(np.all((decoded_learnable_values >= -1) & (decoded_learnable_values <= 1)),
                        "Decoded parameters contain values outside of expected range [-1,1].")
    
    
    def test_encoding_decoding_gradient(self):
        """
        Tests whether the PyTorch gradient correctly flows through the encoding and decoding process.
        """
        
        learnable_mapped_parameters_dict, torch_parameters_for_optimizer_dict = create_learnable_parameters(self.mapping.get_learnable_parameter_types(), device = "cpu", dtype = torch.float32)
        
        def _check_gradient_flow(starting_dict, learnable_tensors):
            loss = sum(torch.abs(value) for value in starting_dict.values())
            loss.backward()
            
            # Check if gradients exist for all original learnable parameters
            for label, value in learnable_tensors.items():
                self.assertIsNotNone(value.grad, f"Gradient not computed for {label}")
                self.assertTrue(torch.any(value.grad != 0), f"Gradient is zero for {label}")
                value.grad.zero_()  # Reset gradient for the next loss
        
        _check_gradient_flow(torch_parameters_for_optimizer_dict, torch_parameters_for_optimizer_dict)
        
        _check_gradient_flow(learnable_mapped_parameters_dict, torch_parameters_for_optimizer_dict)
        
        # They are supposed to be bounded for some of the encoders
        learnable_mapped_parameters_dict = {label: torch.tanh(value) for label, value in learnable_mapped_parameters_dict.items()}
        
        gate_parameters = self.mapping.encoded_to_gate_params(learnable_mapped_parameters_dict)
        _check_gradient_flow(gate_parameters, torch_parameters_for_optimizer_dict)
        
        decoded_learnable_parameters = self.mapping.gate_params_to_encoded(learnable_mapped_parameters_dict)
        _check_gradient_flow(decoded_learnable_parameters, torch_parameters_for_optimizer_dict)
        

class TestPolar_Phase_Mapping(TestRealImg_Phase_Mapping):
    
    def setUp(self):
        """
        Initializes an instance of Polar_Phase_Mapping for testing.
        """
        
        n_qudits = 3    # Two qudits
        n_bumpers = 0
        self.d = 5      # Three energy levels per qudit
        n_blocks = 2    # Two blocks
        
        ansatz = SNAP_Displacement(self.d, n_qudits, n_bumpers, n_blocks)
        self.gate_parameter_types = ansatz.get_parameter_types()
        
        self.mapping = Polar_Phase_Mapping(self.gate_parameter_types, d = self.d)
        

if __name__ == '__main__':

    # Run the unit tests
    unittest.main()





