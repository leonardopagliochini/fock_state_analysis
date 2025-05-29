#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/02/2025

@author: Maurizio Ferrari Dacrema
"""

import numpy as np

def normalize_state(state):
    state[state != 0.0] = state[state != 0.0] / np.sqrt(np.sum(np.abs(state) ** 2))
    return state




