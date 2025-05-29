#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/02/2025

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import torch
from abc import ABC, abstractmethod

def complex_to_polar(complex):
    r = torch.abs(complex)
    phi = torch.angle(complex)
    return r, phi

def polar_to_complex(r, phi):
    # This is a simple r = e^(i*phi)
    # torch.exp(1j * phi)
    # It is written like this because it is compilable by Triton
    return r * torch.exp(torch.complex(torch.zeros(1, device=phi.device), phi))
    
class ParameterMapping(ABC):
    """Abstract base class for different parameter mapping methods."""

    def __init__(self):
        self.learnable_parameter_types = {}
    
    @abstractmethod
    def encoded_to_gate_params(self, learnable_parameter_values):
        """
        Transforms the parameter values from the format used to train them into the representation required to run the simulation (gate parameters).
        """
        pass

    @abstractmethod
    def gate_params_to_encoded(self, gate_parameter_values):
        """
        Transforms the gate parameters into their corresponding learnable values.
        """
        pass
    
    def get_learnable_parameter_types(self):
        return self.learnable_parameter_types.copy()

    def get_n_learnable_parameters(self):
        return len(self.learnable_parameter_types)



class Polar_Phase_Mapping(ParameterMapping):
    """
    Parameter Mapping for SNAP and Displacement Gates using Polar Encoding.
    Learnable parameters are assumed to be bounded between [-1, 1].
    
    Each *complex value* (Displacement Gate parameters) is encoded using radius r and angle phi in polar form bounded between [-1, 1].
    The gate parameters are computed first by rescaling the learnable parameters as follows:
        -  r is rescaled to [0, max_radius]
        -  phi is rescaled to [-π, +π]
    The complex number is then computed as: alpha = r * e^{i * phi}
    
    Each *angle* value corresponds to learnable parameters in [-1, +1].
    The gate parameter is computed by rescaling this learnable parameter to [-π, +π].
    """
    
    def __init__(self, gate_parameter_types, d):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param gate_parameter_types: Dictionary with the parameter types (angle, complex) required by the ansatz.
        :param d: The maximum radius for complex values, used for scaling the decoding correctly.
        """
        
        super(Polar_Phase_Mapping, self).__init__()
        self.gate_parameter_types = gate_parameter_types.copy()
        self.max_radius = d
        
        self.learnable_parameter_types = {}
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                self.learnable_parameter_types[label + "_radius"] = "positive"
                self.learnable_parameter_types[label + "_phi"] = "angle"
            elif type == "angle":
                self.learnable_parameter_types[label] = "angle"
            else:
                raise ValueError("Parameter type not recognized")
    
        
        
    def encoded_to_gate_params(self, encoded_parameter_values):
        """
        Transforms the parameter values from the format used to train them into the representation required to
        run the simulation (gate parameters).

        :param encoded_parameter_values: Dictionary of the encoded parameter values in [-1, +1].
        :return: Dictionary with the gate parameters required to run the simulation (complex values).
        """
        
        for label, value in encoded_parameter_values.items():
            assert -1.0 <= value <= 1.0, f"All parameters must be in range [-1, 1]. Parameter {label} has a value of {value}."
        
        gate_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_radius = label + "_radius"
                label_phi = label + "_phi"
                
                # Map radius from [-1, +1] to [0, max_radius] and phase to [-π, π]
                r = ((encoded_parameter_values[label_radius] + 1) / 2) * self.max_radius
                phi = encoded_parameter_values[label_phi] * torch.pi
                
                complex_value = polar_to_complex(r, phi)
                gate_parameter_values[label] = complex_value
                
            elif type == "angle":
                angle_value = encoded_parameter_values[label]
                gate_parameter_values[label] = angle_value * torch.pi
            
        return gate_parameter_values


    def gate_params_to_encoded(self, gate_parameter_values):
        """
        Transforms the gate parameters into their corresponding learnable values.
        
        :param gate_parameter_values: Dictionary of gate parameters (angles and complex values).
        :return: Dictionary of the parameters in their encoded format, within [-1, +1].
        """
        
        encoded_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_radius = label + "_radius"
                label_phi = label + "_phi"
                
                # Extract radius and phase and rescale
                r, phi = complex_to_polar(gate_parameter_values[label])
                
                # Rescale the learnable parameters in [-1, +1]
                r = (2 * r / self.max_radius) - 1
                phi = phi / torch.pi
                
                encoded_parameter_values[label_radius] = r
                encoded_parameter_values[label_phi] = phi
                
            elif type == "angle":
                angle_value = gate_parameter_values[label]
                # Rescale the learnable parameters in [-1, +1]
                encoded_parameter_values[label] = angle_value / torch.pi
        
        return encoded_parameter_values



class RealImg_Phase_Mapping(ParameterMapping):
    """
    Parameter Mapping for SNAP and Displacement Gates using Real-Imaginary Encoding.
    Learnable parameters are assumed to be bounded between [-1, 1].

    Each *complex value* (Displacement Gate parameters) is encoded using two separate real numbers, one
    for the real part and one for the imaginary one bounded between [-1, 1].
    The complex number is then computed as: alpha = (real + imag) * max_radius

    Each *angle* value corresponds to learnable parameters in [-1, +1].
    The gate parameter is computed by rescaling this learnable parameter to [-π, +π].
    """
    
    def __init__(self, gate_parameter_types, d):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param gate_parameter_types: Dictionary with the parameter types (angle, complex) required by the ansatz.
        :param d: The maximum radius for complex values, used for scaling the decoding correctly.
        """
        
        super(RealImg_Phase_Mapping, self).__init__()
        self.gate_parameter_types = gate_parameter_types.copy()
        
        # Approximately ensure that had this complex number been represented in polar coordinates
        # then its radius would have been in [0, d]. Since r=sqrt(re^2+im^2) then a max value for re=im=d/sqrt(2) achieves this
        self.max_equal_norm = d / np.sqrt(2)
        
        self.learnable_parameter_types = {}
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                self.learnable_parameter_types[label + "_real"] = "real"
                self.learnable_parameter_types[label + "_imag"] = "real"
            elif type == "angle":
                self.learnable_parameter_types[label] = "angle"
            else:
                raise ValueError("Parameter type not recognized")
    
    
    def encoded_to_gate_params(self, encoded_parameter_values):
        """
        Transforms the parameter values from the format used to train them into the representation required to
        run the simulation (gate parameters).

        :param encoded_parameter_values: Dictionary of the encoded parameter values in [-1, +1].
        :return: Dictionary with the gate parameters required to run the simulation (complex values).
        """
        
        for label, value in encoded_parameter_values.items():
            assert -1.0 <= value <= 1.0, f"All parameters must be in range [-1, 1]. Parameter {label} has a value of {value}."
        
        gate_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_real = label + "_real"
                label_imag = label + "_imag"

                real = encoded_parameter_values[label_real]
                imag = encoded_parameter_values[label_imag]
                
                # Map complex number from [-1, +1] to [-self.max_equal_norm, self.max_equal_norm]
                complex_value = torch.complex(real, imag) * self.max_equal_norm
                gate_parameter_values[label] = complex_value
            
            elif type == "angle":
                # Map real angle from [-1, +1] to [-π, +π]
                angle_value = encoded_parameter_values[label]
                gate_parameter_values[label] = angle_value * torch.pi
        
        return gate_parameter_values
    
    def gate_params_to_encoded(self, gate_parameter_values):
        """
        Transforms the gate parameters into their corresponding learnable values.

        :param gate_parameter_values: Dictionary of gate parameters (angles and complex values).
        :return: Dictionary of the parameters in their encoded format, within [-1, +1].
        """
        
        encoded_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_real = label + "_real"
                label_imag = label + "_imag"
                
                complex_value = gate_parameter_values[label]
                
                # Map complex number from [-self.max_radius, self.max_radius] to [-1, +1]
                assert (-self.max_equal_norm <= complex_value.real <= self.max_equal_norm and
                        -self.max_equal_norm <= complex_value.imag <= self.max_equal_norm), \
                    f"All complex valued parameters must be in range [-self.max_equal_norm, self.max_equal_norm]. Parameter {label} has a value of {complex_value}."

                encoded_parameter_values[label_real] = complex_value.real / self.max_equal_norm
                encoded_parameter_values[label_imag] = complex_value.imag / self.max_equal_norm
                
            elif type == "angle":
                angle_value = gate_parameter_values[label]
                
                # Map real angle from [-π, +π] to [-1, +1]
                assert -torch.pi <= angle_value <= torch.pi, f"Angles must be in range [-π, +π]. Parameter {label} has a value of {angle_value}."
                encoded_parameter_values[label] = angle_value / torch.pi
        
        return encoded_parameter_values
    
    
    