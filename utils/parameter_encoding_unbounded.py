#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/02/2025

@author: Maurizio Ferrari Dacrema
"""

import torch, warnings
import numpy as np
from utils.parameter_encoding import complex_to_polar, polar_to_complex, ParameterMapping

def clamp_angle(angle_value, label):

    if not (-torch.pi <= angle_value <= torch.pi):
        warnings.warn(f"Clipping angle '{label}' with value {angle_value:.5e} to [-π, π].", RuntimeWarning)

    if isinstance(angle_value, torch.Tensor):
        return angle_value.clamp(-torch.pi, torch.pi)

    elif isinstance(angle_value, (float, int)):
        return float(np.clip(angle_value, -torch.pi, torch.pi))

    else:
        raise TypeError("Input must be a float, int, or scalar torch.Tensor")



class Polar_Phase_Mapping(ParameterMapping):
    """
    Parameter Mapping for SNAP and Displacement Gates using Polar Encoding.
    Learnable parameters are NOT BOUNDED, except for angles.
    
    Each *complex value* (Displacement Gate parameters) is encoded using radius r and angle phi in polar form bounded between [-1, 1].
    The gate parameters are computed first by rescaling the learnable parameters as follows:
        -  r is in [0, inf)
        -  phi is in [-π, +π]
    The complex number is then computed as: alpha = r * e^{i * phi}
    
    Each *angle* value corresponds to learnable parameters in [-π, +π].
    """
    
    def __init__(self, gate_parameter_types):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param gate_parameter_types: Dictionary with the parameter types (angle, complex) required by the ansatz.
        """
        
        super(Polar_Phase_Mapping, self).__init__()
        self.gate_parameter_types = gate_parameter_types.copy()
        
        self.learnable_parameter_types = {}
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                self.learnable_parameter_types[label + "_radius"] = "positive"
                self.learnable_parameter_types[label + "_phi"] = "angle"
            elif type == "angle":
                self.learnable_parameter_types[label] = "angle"
            elif type == "positive":
                self.learnable_parameter_types[label] = "positive"
            elif type == "real":
                self.learnable_parameter_types[label] = "real"
            else:
                raise ValueError("Parameter type not recognized")
    
        
        
    def encoded_to_gate_params(self, encoded_parameter_values):
        """
        Transforms the parameter values from the format used to train them into the representation required to
        run the simulation (gate parameters).

        :param encoded_parameter_values: Dictionary of the encoded parameter values.
        :return: Dictionary with the gate parameters required to run the simulation (complex values).
        """
        
        gate_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_radius = label + "_radius"
                label_phi = label + "_phi"
                
                r = encoded_parameter_values[label_radius]
                phi = encoded_parameter_values[label_phi]
                
                assert r >= 0.0, f"Radii must be in range [0.0, inf). Parameter {label_radius} has a value of {r}."

                phi = clamp_angle(phi, label)
                
                complex_value = polar_to_complex(r, phi)
                gate_parameter_values[label] = complex_value
                
            elif type == "angle":
                angle_value = encoded_parameter_values[label]
                angle_value = clamp_angle(angle_value, label)
                gate_parameter_values[label] = angle_value

            elif type == "positive":
                positive_value = encoded_parameter_values[label]
                assert positive_value >= 0, f"Positive values must be in range [0, inf). Parameter {label} has a value of {positive_value}."
                gate_parameter_values[label] = positive_value

            elif type == "real":
                real_value = encoded_parameter_values[label]
                gate_parameter_values[label] = real_value

            else:
                raise ValueError(f"Type {type} for parameter {label} not recognized")
            
        return gate_parameter_values


    def gate_params_to_encoded(self, gate_parameter_values):
        """
        Transforms the gate parameters into their corresponding learnable values.
        
        :param gate_parameter_values: Dictionary of gate parameters (angles and complex values).
        :return: Dictionary of the parameters in their encoded format.
        """
        
        encoded_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_radius = label + "_radius"
                label_phi = label + "_phi"
                
                r, phi = complex_to_polar(gate_parameter_values[label])
                
                encoded_parameter_values[label_radius] = r
                encoded_parameter_values[label_phi] = phi
                
            elif type == "angle":
                angle_value = gate_parameter_values[label]
                angle_value = clamp_angle(angle_value, label)
                encoded_parameter_values[label] = angle_value

            elif type == "positive":
                positive_value = gate_parameter_values[label]
                assert positive_value >= 0, f"Positive values must be in range [0, inf). Parameter {label} has a value of {positive_value}."
                encoded_parameter_values[label] = positive_value

            elif type == "real":
                real_value = gate_parameter_values[label]
                encoded_parameter_values[label] = real_value

            else:
                raise ValueError(f"Type {type} for parameter {label} not recognized")
        
        return encoded_parameter_values



class RealImg_Phase_Mapping(ParameterMapping):
    """
    Parameter Mapping for SNAP and Displacement Gates using Real-Imaginary Encoding.
    Learnable parameters are NOT BOUNDED, except for angles.

    Each *complex value* (Displacement Gate parameters) is encoded using two separate real numbers, one
    for the real part and one for the imaginary one.
    The complex number is then computed as: alpha = real + imag * 1j

    Each *angle* value corresponds to learnable parameters in [-π, +π].
    """
    
    def __init__(self, gate_parameter_types):
        """
        Initializes the parameter mapping for displacement and SNAP gates.

        :param gate_parameter_types: Dictionary with the parameter types (angle, complex) required by the ansatz.
        """
        
        super(RealImg_Phase_Mapping, self).__init__()
        self.gate_parameter_types = gate_parameter_types.copy()
        
        self.learnable_parameter_types = {}
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                self.learnable_parameter_types[label + "_real"] = "real"
                self.learnable_parameter_types[label + "_imag"] = "real"
            elif type == "angle":
                self.learnable_parameter_types[label] = "angle"
            elif type == "positive":
                self.learnable_parameter_types[label] = "positive"
            elif type == "real":
                self.learnable_parameter_types[label] = "real"
            else:
                raise ValueError("Parameter type not recognized")
    
    
    def encoded_to_gate_params(self, encoded_parameter_values):
        """
        Transforms the parameter values from the format used to train them into the representation required to
        run the simulation (gate parameters).

        :param encoded_parameter_values: Dictionary of the encoded parameter values.
        :return: Dictionary with the gate parameters required to run the simulation (complex values).
        """
        
        gate_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_real = label + "_real"
                label_imag = label + "_imag"

                real = encoded_parameter_values[label_real]
                imag = encoded_parameter_values[label_imag]
                
                complex_value = real + 1j * imag
                gate_parameter_values[label] = complex_value
            
            elif type == "angle":
                angle_value = encoded_parameter_values[label]
                angle_value = clamp_angle(angle_value, label)
                gate_parameter_values[label] = angle_value

            elif type == "positive":
                positive_value = encoded_parameter_values[label]
                assert positive_value >= 0, f"Positive values must be in range [0, inf). Parameter {label} has a value of {positive_value}."
                gate_parameter_values[label] = positive_value

            elif type == "real":
                real_value = encoded_parameter_values[label]
                gate_parameter_values[label] = real_value

            else:
                raise ValueError(f"Type {type} for parameter {label} not recognized")

        return gate_parameter_values
    
    def gate_params_to_encoded(self, gate_parameter_values):
        """
        Transforms the gate parameters into their corresponding learnable values.

        :param gate_parameter_values: Dictionary of gate parameters (angles and complex values).
        :return: Dictionary of the parameters in their encoded format
        """
        
        encoded_parameter_values = {}
        
        for label, type in self.gate_parameter_types.items():
            if type == "complex":
                label_real = label + "_real"
                label_imag = label + "_imag"
                
                complex_value = gate_parameter_values[label]

                encoded_parameter_values[label_real] = complex_value.real
                encoded_parameter_values[label_imag] = complex_value.imag
                
            elif type == "angle":
                angle_value = gate_parameter_values[label]
                angle_value = clamp_angle(angle_value, label)
                encoded_parameter_values[label] = angle_value

            elif type == "positive":
                positive_value = gate_parameter_values[label]
                assert positive_value >= 0, f"Positive values must be in range [0, inf). Parameter {label} has a value of {positive_value}."
                encoded_parameter_values[label] = positive_value

            elif type == "real":
                real_value = gate_parameter_values[label]
                encoded_parameter_values[label] = real_value

            else:
                raise ValueError(f"Type {type} for parameter {label} not recognized")

        return encoded_parameter_values
    
    
    