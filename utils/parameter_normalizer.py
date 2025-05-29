#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/03/2025

@author: Maurizio Ferrari Dacrema
"""

import torch
import torch.nn as nn

class ParameterNormalizer(object):
    """
    A utility class for normalizing learnable parameters in quantum circuit optimization.

    This class provides a mechanism to create and normalize PyTorch learnable parameters
    based on predefined types such as angles, positive values, and real numbers.
    The normalization ensures that the parameters are transformed into the appropriate
    range for use in quantum gate optimization.

    Attributes:
        learnable_parameter_types (dict):
            A dictionary mapping parameter labels to their respective types
            (e.g., "angle", "positive", "real").
        learnable_torch_parameter_dict (dict):
            A dictionary of PyTorch nn.Parameter tensors representing the raw
            learnable parameters before normalization.

    Methods:
        get_torch_parameters():
            Returns the dictionary of raw learnable PyTorch parameters.

        get_normalized_parameters():
            Returns the dictionary of parameters transformed into their
            required range for quantum gate optimization.
    """
    
    def __init__(self, learnable_parameter_types: dict,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        """
        Initializes the ParameterNormalizer.

        Args:
            learnable_parameter_types (dict):
                A dictionary defining the types of learnable parameters
                (e.g., "angle", "positive", "real").
            device (str, optional):
                The device where tensors are stored (default: 'cpu').
            dtype (torch.dtype, optional):
                The data type of the learnable parameters (default: torch.float32).
        """
        super(ParameterNormalizer, self).__init__()
        
        self.learnable_parameter_types = learnable_parameter_types.copy()
        self.learnable_torch_parameter_dict = {label: nn.Parameter(torch.rand(1, dtype = dtype, device = device)) for label in self.learnable_parameter_types.keys()}
    
    def get_torch_parameters(self):
        """
        Retrieves the dictionary of raw learnable PyTorch parameters.

        Returns:
            dict: A dictionary where keys are parameter labels,
                  and values are PyTorch nn.Parameter tensors.
        """
        return self.learnable_torch_parameter_dict
    
    def get_normalized_parameters(self):
        """
        Transforms raw learnable parameters into the required range for quantum gate optimization.

        Normalization rules:
        - "angle": Transformed using `tanh` to ensure values stay within [-π, +π].
        - "positive": Transformed using `ReLU` to ensure non-negative values.
        - "real": Unaltered.

        Returns:
            dict: A dictionary containing the normalized gate parameters.
        """
        
        normalized_parameter_dict = {}
        
        for label, parameter in self.learnable_torch_parameter_dict.items():
            
            parameter_type = self.learnable_parameter_types[label]
            
            if parameter_type == "angle":
                normalized_parameter_dict[label] = nn.Tanh()(parameter) * torch.pi
            elif parameter_type == "positive":
                normalized_parameter_dict[label] = nn.ReLU()(parameter)
            elif parameter_type == "real":
                normalized_parameter_dict[label] = parameter
            else:
                raise ValueError(f"Parameter type not recognized. Parameter is {label}, type is {type}")
        
        return normalized_parameter_dict


