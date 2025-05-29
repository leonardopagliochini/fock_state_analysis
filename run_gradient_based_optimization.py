#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/03/2025

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import torch, multiprocessing
from utils.parameter_encoding_unbounded import Polar_Phase_Mapping, RealImg_Phase_Mapping
from generate_target_states import read_or_create_target_state
from quantum_circuit.ansatz_all_in_one import *
from train_gradient_based import ExperimentCase, reset_seed, train_parameters, evenly_spaced_integers
from utils.DataIO import DataIO
import os, itertools, traceback
import pandas as pd

import logging
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor*")


# # === Logging setup ===
# os.makedirs(experiment_folder, exist_ok=True)
# log_path = os.path.join(experiment_folder, "log.txt")
# logging.basicConfig(filename=log_path,
#                     filemode='a',
#                     format='[%(asctime)s] %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     level=logging.INFO)

# logging.info(f"Started new experiment batch at {datetime.now()}")


def run_experiment(config):
    try:
        if isinstance(config.state_name, str):
            target_state = read_or_create_target_state(config.state_name, config.n_qudits, config.d, freeze_data=True)
            target_state = torch.tensor(target_state, dtype=torch.complex64, device=config.device)
            folder_state_name = config.state_name
        else:
            fock_state_target = config.state_name
            equally_spaced_focks = evenly_spaced_integers(config.d, values=7)[1:-1]
            if fock_state_target >= len(equally_spaced_focks):
                return
            fock_state_target = equally_spaced_focks[fock_state_target]
            config.state_name = f"fock_{fock_state_target}"
            folder_state_name = "fock"
            target_state = torch.zeros(config.d, dtype=torch.complex64, device=config.device)
            target_state[fock_state_target] = 1.0

        config.experiment_folder = config.experiment_folder + f"/{folder_state_name}/initial_{config.initial_state}/{config.architecture}/"
        config.results_folder = config.experiment_folder + "results/"

        if not isinstance(config.blocks_policy, str) and config.blocks_policy is not None:
            config.n_blocks = int(config.blocks_policy * config.d)
        elif config.blocks_policy == "sqrt":
            config.n_blocks = np.ceil(np.sqrt(config.d)).astype(int)
        elif config.blocks_policy == "d":
            config.n_blocks = config.d
        elif config.blocks_policy is not None:
            raise ValueError("blocks_policy type not recognized")
        
        config.use_unitary = False
        config.use_density = False


        if isinstance(config.initial_state, str):
            if config.initial_state == "d-1":
                config.initial_state = config.d - 1
            else:
                raise ValueError("initial_state type not recognized")

        reset_seed()
        lr = 1e-3

        result_file_name = (f"state_name={config.state_name}-loss={config.loss_type}"
                            f"-n_qudits={config.n_qudits}-d={config.d}-n_bumpers={config.n_bumpers}"
                            f"-blocks_policy={config.blocks_policy}-n_blocks={config.n_blocks}"
                            f"-trial={config.trial}-device={config.device}")
        dataIO = DataIO(folder_path=config.results_folder)

        try:
            result_dict = None
            for check_device in ["cuda", "cpu"]:
                result_file_name_device = result_file_name.replace(f"-device={config.device}", f"-device={check_device}")
                if result_dict is None and os.path.exists(config.results_folder + result_file_name_device + "-best_parameters.zip"):
                    result_dict = dataIO.load_data(result_file_name_device + "-best_parameters")

            if result_dict is None or "best_parameters_realimag" not in result_dict:
                raise FileNotFoundError()

        except FileNotFoundError:
            initial_state = torch.zeros(config.d + config.n_bumpers, dtype=torch.complex64, device=config.device)
            initial_state[config.initial_state] = 1.0
            zeros = torch.zeros(config.n_bumpers, dtype=target_state.dtype, device=target_state.device)
            target_state_padded = torch.cat((target_state, zeros), dim=0)

            if config.architecture == "DdSD_angle=Tanh":
                ansatz = SNAP_Displacement(config.d, config.n_bumpers, config.n_blocks, config.device)
            elif config.architecture == "DdSnD_angle=Tanh":
                ansatz = SNAPn_Displacement(config.d, config.n_bumpers, config.n_blocks, config.n_learnable_phases, config.device)
            else:
                raise ValueError(f"Architecture not recognized: {config.architecture}")

            parameter_normalizer = ansatz
            parameter_mapping_realimag = RealImg_Phase_Mapping(ansatz.get_parameter_types())

            train_parameters(config,
                             ansatz,
                             parameter_normalizer,
                             parameter_mapping_realimag,
                             result_file_name,
                             lr,
                             initial_state,
                             target_state_padded)

        # === Save summary result ===
        summary_file = os.path.join(config.experiment_folder, "summary.csv")
        os.makedirs(config.experiment_folder, exist_ok=True)

        row = {
            "state_name": config.state_name,
            "n_blocks": config.n_blocks,
            "trial": config.trial,
            "loss_type": config.loss_type,
            "d": config.d,
            "n_bumpers": config.n_bumpers,
            "architecture": config.architecture,
        }

        result_file_path = os.path.join(config.results_folder, result_file_name + "-best_parameters.zip")
        if os.path.exists(result_file_path):
            result_dict = dataIO.load_data(result_file_name + "-best_parameters")
            row["best_fidelity"] = result_dict.get("best_fidelity", None)
        else:
            row["best_fidelity"] = None

        df = pd.DataFrame([row])
        if os.path.exists(summary_file):
            df.to_csv(summary_file, mode='a', index=False, header=False)
        else:
            df.to_csv(summary_file, mode='w', index=False, header=True)

    except Exception as e:
        traceback.print_exc()


if __name__ == '__main__':
    experiment_folder = "./results_fock_experiment"

    d = 32
    n_qudits = 1
    n_bumpers = 64

    # Fock states da fock_1 a fock_d-1 escluso fock_0
    all_fock_states = list(range(1, d))
    step = len(all_fock_states) // 7
    fock_state_names = [f"fock_{i}" for i in all_fock_states[::step]][:7]
    # fock_state_names = []

    # n_blocks dispari da 1 a d-1
    n_blocks_list = list(range(1, d, 2))

    experiments_iterator = itertools.product(
        fock_state_names,     # state_name
        n_blocks_list,        # n_blocks
        ["cpu"],              # device
        [0],                  # initial_state
        ["DdSD_angle=Tanh"],  # architecture
        range(5),             # trial (1 sola ripetizione)
        ["fidelity"]          # loss_type
    )

    torch.set_num_threads(6)
    all_cases_list = []

    for (state_name, n_blocks, device, initial_state, architecture, trial, loss_type) in experiments_iterator:
        experiment_case = ExperimentCase(
            experiment_folder=experiment_folder,
            results_folder=None,
            state_name=state_name,
            initial_state=initial_state,
            loss_type=loss_type,
            blocks_policy=None,  # non usiamo policy
            n_qudits=n_qudits,
            d=d,
            n_bumpers=n_bumpers,
            n_bumpers_per_d=None,
            trial=trial,
            backend=None,
            device=device,
        )
        experiment_case.architecture = architecture
        experiment_case.n_blocks = n_blocks  # forziamo direttamente il valore

        all_cases_list.append(experiment_case)

    print(f"Running {len(all_cases_list)} experiments...")

    processes = 1
    if processes == 1:
        for idx, case in enumerate(all_cases_list):
            print(f"[ {idx+1:3d} / {len(all_cases_list)} ] Running: state={case.state_name}, blocks={case.n_blocks}, trial={case.trial} ...")
            logging.info(f"Starting simulation {idx+1}/{len(all_cases_list)}: state={case.state_name}, blocks={case.n_blocks}, trial={case.trial}")
            run_experiment(case)
            logging.info(f"Completed simulation {idx+1}/{len(all_cases_list)}")
            run_experiment(case)
    else:
        pool = multiprocessing.Pool(processes=processes, maxtasksperchild=1)
        pool.map(run_experiment, all_cases_list, chunksize=1)
        pool.close()
        pool.join()
