import time, traceback

import pandas as pd
import numpy as np
from function_generator import FunctionGenerator
import optuna as op
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils_quantum import normalize_state
from utils.DataIO import DataIO
import torch


def index_to_qudit_state(index, n_qudits, d):
    """Convert an index to its corresponding qudit state representation in base d."""
    state = []
    for _ in range(n_qudits):
        state.append(str(index % d))
        index //= d
    return "".join(reversed(state))


def generate_target_state(state_name: str, d: int) -> torch.Tensor:
    if state_name.lower().startswith("fock_"):
        try:
            n = int(state_name.split("_")[1])
            state = torch.zeros((d,), dtype=torch.complex64)
            state[n] = 1.0 + 0j
            return state
        except:
            raise ValueError(f"Invalid Fock state name: {state_name}")
    raise ValueError(f"State generation for '{state_name}' is not implemented in generate_target_state().")


def read_or_create_target_state(function_name, n_qudit, d, freeze_data=True):
    if function_name.lower().startswith("fock_"):
        return generate_target_state(function_name, d)

    dataIO = DataIO(folder_path=f"data_target_states/{function_name}")
    file_name = f"n_qudit={n_qudit}-d={d}"

    try:
        target_state_data = dataIO.load_data(file_name)
        target_state = target_state_data["target_state"]
    except FileNotFoundError as e:
        if freeze_data:
            raise e

        try:
            function = FunctionGenerator(function_name, d, n_qudit)
            target_state = normalize_state(function.get_function())
        except Exception as exc:
            traceback.print_exc()
            return None

        target_state_data = {
            "target_state": target_state,
        }

        dataIO.save_data(file_name=file_name, data_dict_to_save=target_state_data)

    # Generate state labels
    state_labels = [index_to_qudit_state(i, n_qudit, d) for i in range(len(target_state))]

    # Plot and save the state as a histogram
    width = 0.25  # the width of the bars
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(target_state)) - width/2, np.real(target_state), alpha=0.7, color='blue', width=width, label="Real")
    plt.bar(np.arange(len(target_state)) + width/2, np.imag(target_state), alpha=0.7, color='red', width=width, label="Imag")
    plt.xticks(range(len(target_state)), state_labels, rotation=90)
    plt.xlabel(f"Qudit State ({function_name})")
    plt.ylabel("Amplitude")
    plt.title(f"Target State: {function_name}, n_qudit={n_qudit}, d={d}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"data_target_states/{function_name}/{file_name}.png")
    plt.close()

    return target_state


if __name__ == '__main__':

    n_qudit_range = [1]
    d_range = [2**p for p in range(1, 15)]

    functions_names = ["fock_3", "fock_5", "random_i"]  # test anche dei fock

    for n_qudit in n_qudit_range:
        for d in d_range:
            for function_name in functions_names:
                read_or_create_target_state(function_name, n_qudit, d, freeze_data=False)
