#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/04/2025

@author: Maurizio Ferrari Dacrema
"""

import time
import pandas as pd
import numpy as np
import torch, multiprocessing, random, os, re, pickle
import torch.optim as optim
from utils.parameter_encoding_unbounded import Polar_Phase_Mapping, RealImg_Phase_Mapping
from generate_target_states import read_or_create_target_state
#from quantum_circuit.ansatz import SNAP_Displacement, SNAP1_Displacement
from quantum_circuit.ansatz_all_in_one import *
from utils.DataIO import DataIO
from quantum_circuit.simulation_circuits_MQT import compute_fidelity as compute_fidelity_numpy
from utils.parameter_normalizer import ParameterNormalizer
from torch.utils.tensorboard import SummaryWriter
from utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from utils.load_save_checkpoint import load_latest_checkpoint, save_checkpoint_grouped, split_dataframe_into_chunks, merge_chunks_to_dataframe
import logging
from datetime import datetime

# === Logging setup ===
os.makedirs("experiment_folder", exist_ok=True)
log_path = os.path.join("experiment_folder", "log.txt")
logging.basicConfig(filename=log_path,
                    filemode='a',
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info(f"Started new experiment batch at {datetime.now()}")




def reset_seed():
    random.seed(os.getpid())
    torch.manual_seed(os.getpid())
    np.random.seed(os.getpid())


def complex_mse_loss(c_pred, c_target):
    """
    Computes MSE between complex amplitude vectors.

    Args:
        c_pred (torch.Tensor): Predicted complex amplitudes, shape (d,), dtype=torch.cfloat
        c_target (torch.Tensor): Target complex amplitudes, shape (d,), dtype=torch.cfloat

    Returns:
        torch.Tensor: Scalar loss (real-valued)
        Note that the loss is real by definition but we need to convert the torch dtype from complex to real, so we get the .real part only
    """
    return torch.sum(torch.abs(c_pred - c_target) ** 2).real


def compute_state_fidelity(state1, state2):
    # Vdot computes \sum_{n} a_n^* b_n = <a|b>
    # Hence it already ensures that a is conjugate
    # return torch.abs(torch.vdot(state1, state2)) ** 2
    # Unfortunately Vdot and dot in general is unstable for complex numbers
    return torch.abs(torch.sum(torch.conj(state1) * state2)) ** 2

def compute_unitary_fidelity(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """
    Compute the average gate fidelity (Hilbert-Schmidt fidelity) between two unitary operators U1 and U2.

    Mathematically:
        F(U1, U2) = |Tr(U1† U2)|² / d²

    This measure is:
        - Equal to 1 if and only if U1 and U2 differ by a global phase
        - Invariant under basis change
        - Physically meaningful for comparing how similarly the two unitaries act on all quantum states

    Args:
        U1 (torch.Tensor): First unitary matrix of shape (d, d)
        U2 (torch.Tensor): Second unitary matrix of shape (d, d)

    Returns:
        torch.Tensor: Scalar fidelity in [0, 1], real-valued
    """

    d = U1.shape[0]
    overlap = torch.trace(torch.matmul(U1.conj().T, U2))
    return (torch.abs(overlap) ** 2 / d**2).real


def fidelity_with_density_matrix(rho: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Compute fidelity between a density matrix rho and a pure state |psi⟩.
    Assumes rho is (d, d) and psi is (d,)
    """
    return torch.dot(psi.conj(), rho @ psi).real
    #return (psi.conj().view(1, -1) @ rho @ psi.view(-1, 1)).real.squeeze()


def fidelity_leakage_decomposition(prepared, target, d):
    """
    Decompose fidelity into:
      - total fidelity (F)
      - core support norm (rho_core)
      - leakage norm (rho_leak)
      - normalized fidelity (F_normalized)

    Args:
        prepared (torch.Tensor): Prepared state (complex), shape (d + b,)
        target (torch.Tensor): Target state (complex), shape (d,)

    Returns:
        dict with:
            - 'fidelity': scalar (real)
            - 'core_norm': scalar (real)
            - 'leakage': scalar (real)
            - 'normalized_fidelity': scalar (real)
    """
    
    if target.ndim == 2 and target.shape[1] == 1:
        target = target.squeeze()
    
    if prepared.ndim == 2 and prepared.shape[1] == 1:
        prepared = prepared.squeeze()
    
    assert target.ndim == 1 and prepared.ndim == 1, "Vectors must be 1D after squeezing"
    assert prepared.shape[0] == target.shape[0], "Prepared and target states must have the same number of components"
    
    # Core overlap fidelity ⟨target|prepared⟩
    global_fidelity = compute_state_fidelity(target, prepared)
    overlap_fidelity = compute_state_fidelity(target[:d], prepared[:d])
    
    # Norms
    overlap_norm = torch.sum(torch.abs(prepared[:d]) ** 2).real
    leakage_norm = 1.0 - overlap_norm
    
    # Normalized fidelity
    normalized_overlap_fidelity = overlap_fidelity / overlap_norm if overlap_norm > 0 else torch.tensor(0.0)
    
    return global_fidelity.item(), overlap_fidelity.item(), overlap_norm.item(), leakage_norm.item(), normalized_overlap_fidelity.item()



def fidelity_leakage_decomposition_unitary(U_prepared: torch.Tensor, U_target: torch.Tensor, d: int):
    """
    Decomposes fidelity and leakage when comparing two unitary operators U_prepared and U_target
    acting on a Hilbert space of dimension D = d + n_bumpers.

    It computes the following quantities:

    - Global fidelity (over the full Hilbert space):
        F_global = |Tr(U_prepared† · U_target)|² / D²

    - Overlap fidelity (restricted to the core d×d subspace):
        F_overlap = |Tr(U_prepared_core† · U_target_core)|² / d²

    - Core norm (squared Frobenius norm of the prepared unitary on the core):
        ||U_prepared_core||²_F / d = Tr(U_prepared_core† · U_prepared_core) / d

    - Leakage: 1 - core_norm (how much norm spills into bumper rows/columns)

    - Normalized overlap fidelity:
        F_overlap_normalized = F_overlap / core_norm

    These metrics allow one to track both:
    - the fidelity of the synthesized unitary with respect to the target
    - and the amount of "leakage" into the extended bumper space


    Args:
        U_prepared (torch.Tensor): Synthesized unitary, shape (d + b, d + b)
        U_target (torch.Tensor): Target padded unitary, shape (d + b, d + b)
        d (int): Dimension of the core subspace (no bumpers)

    Returns:
        tuple of floats:
            - global_fidelity
            - overlap_fidelity
            - core_norm
            - leakage
            - normalized_overlap_fidelity
    """

    D = U_prepared.shape[0]

    # Compute full fidelity
    global_fidelity = compute_unitary_fidelity(U_prepared, U_target)

    # Extract top-left submatrices
    U_prep_core = U_prepared[:d, :d]
    U_target_core = U_target[:d, :d]
    overlap_fidelity = compute_unitary_fidelity(U_prep_core, U_target_core)

    # Norm of core block (should be ~1 for unitaries, but check leakage)
    core_norm = torch.real(torch.trace(torch.matmul(U_prep_core.conj().T, U_prep_core)) / d)
    leakage = 1.0 - core_norm.item()

    # Normalize overlap fidelity by norm of core
    normalized_overlap_fidelity = overlap_fidelity / core_norm if core_norm > 0 else torch.tensor(0.0)

    return global_fidelity.item(), overlap_fidelity.item(), core_norm.item(), leakage, normalized_overlap_fidelity.item()


def evenly_spaced_integers(d, values = 10):
    if d < values:
        return list(range(d + 1))
    else:
        # TODO dovrebbe essere 0/d-1 non d
        return [round(i * d / (values - 1)) for i in range(values)]


def train_parameters(config,
                     ansatz,
                     parameter_normalizer,
                     parameter_mapping_realimag,
                     result_file_name,
                     lr,
                     initial_state,
                     target_state_or_unitary,
                     regularization_weight = 0.0,
                     allow_load_from_disk = True):

    #config.dataframe_folder = config.experiment_folder + "dataframe/"
    os.makedirs(config.results_folder, exist_ok = True)
    #os.makedirs(config.dataframe_folder, exist_ok = True)
    
    dataIO = DataIO(folder_path = config.results_folder)
    
    # parameter_normalizer = ParameterNormalizer(parameter_mapping.get_learnable_parameter_types(), device = device, dtype = torch.float32)
    optimizer = optim.Adam(list(parameter_normalizer.get_torch_parameters().values()), lr = lr, weight_decay = 0.0, maximize = True)
    
    full_protocol_parameters = config.d * (config.d + 2) + 2
    logging.info(result_file_name + f" - Number of gates = {ansatz.get_n_gates()}, "
                             f"Number of learnable parameters = {parameter_normalizer.get_n_parameters()} ({parameter_normalizer.get_n_parameters() / full_protocol_parameters * 100:.0f} %), "
                             f"Full-protocol parameters = {full_protocol_parameters}")
    log_dir = os.path.join(config.experiment_folder, "tensorboard_logs", result_file_name)
    #writer = SummaryWriter(log_dir = log_dir)
    
    ########################################################################################################
    ########    Setting up the training loop and early-stopping variables
    ########
    
    max_train_step = int(1e5)
    patience_max = 1000
    min_delta_fidelity = 1e-6
    checkpoint_interval_sec = 30 * 60  # 30 * 60  # seconds
    last_checkpoint_time = time.time()
    best_gate_parameters = None
    
    try:
        if not allow_load_from_disk:
            raise FileNotFoundError()

        checkpoint_dict = load_latest_checkpoint(config.results_folder, result_file_name + "-checkpoint")
        
        best_fidelity = checkpoint_dict["best_fidelity"]
        best_torch_parameters = checkpoint_dict["best_torch_parameters"]
        best_parameters_realimag = checkpoint_dict["best_parameters_realimag"]
        initial_train_step = checkpoint_dict["step"] + 1
        best_prepared_state = checkpoint_dict["best_parameters_realimag"]
        patience_counter = checkpoint_dict["patience_counter"]
        training_time = checkpoint_dict["training_time"]
        
        if "result_df" in checkpoint_dict:
            result_df = checkpoint_dict["result_df"]
        else:
            result_df = merge_chunks_to_dataframe(checkpoint_dict["result_df_chunks"])
        
        # result_df = pd.read_csv(config.results_folder + result_file_name + "-dataframe.csv")
        
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        ansatz.set_torch_parameters(checkpoint_dict["torch_parameters"])
        
        new_time_value, new_time_unit = seconds_to_biggest_unit(training_time)
        logging.info(result_file_name + f" - Restoring checkpoint - Step {initial_train_step}, Fidelity Global = {best_fidelity:.6f}, "
                                 f"Elapsed Time = {new_time_value:.2f} {new_time_unit}")
    
    except FileNotFoundError:
        best_fidelity = -np.inf
        best_prepared_state = None
        best_torch_parameters = None
        best_parameters_realimag = {}
        initial_train_step = 0
        patience_counter = 0
        training_time = 0
        result_df = None
    
    data_dict_to_save = {}
    step_start_time = time.time()
    
    for step in range(initial_train_step, max_train_step + 1):
        
        optimizer.zero_grad()
        
        ### These are required for the versions that use separate parameter encoding
        # normalized_learnable_parameters_dict = parameter_normalizer.get_normalized_parameters()
        # gate_parameters = parameter_mapping.encoded_to_gate_params(normalized_learnable_parameters_dict)
        # prepared_state = ansatz.run_state(gate_parameters, initial_state = initial_state, device = device)

        if config.use_unitary:
            prepared_state_or_unitary = ansatz.run_unitary()
            fidelity = compute_unitary_fidelity(prepared_state_or_unitary, target_state_or_unitary)
        else:
            if config.use_density:
                density_matrix = ansatz.run_density(initial_state = initial_state)
                fidelity = fidelity_with_density_matrix(density_matrix, target_state_or_unitary)
                assert False
            else:
                prepared_state_or_unitary = ansatz.run_state(initial_state = initial_state)
                fidelity = compute_state_fidelity(prepared_state_or_unitary, target_state_or_unitary)
        
        # Return as a real-valued scalar
        if config.loss_type == "log_infidelity":
            loss = 1 - torch.log(1 - fidelity.squeeze() + 1e-9)
        elif config.loss_type == "fidelity":
            loss = fidelity.squeeze()
        else:
            raise ValueError("Loss type not recognized")
        
        # Regularization on gate parameters (not the learnable ones)
        D_gate, S_gate = ansatz._cached_gate_params
        gate_par_reg = torch.sum(torch.abs(D_gate) ** 2) + torch.sum(S_gate ** 2)
        loss = loss - gate_par_reg * regularization_weight
        
        #################################################################
        #### Log and early-stopping
        if config.use_unitary:
            global_fidelity_torch, overlap_fidelity, overlap_norm, leakage_norm, normalized_overlap_fidelity = fidelity_leakage_decomposition_unitary(prepared_state_or_unitary, target_state_or_unitary, config.d)
        else:
            global_fidelity_torch, overlap_fidelity, overlap_norm, leakage_norm, normalized_overlap_fidelity = fidelity_leakage_decomposition(prepared_state_or_unitary, target_state_or_unitary, config.d)

        prepared_state_detatched = prepared_state_or_unitary.detach().cpu().numpy().squeeze()
        
        if global_fidelity_torch > best_fidelity + min_delta_fidelity:
            best_fidelity = global_fidelity_torch
            
            # Extract the single float value contained in the tensor
            best_gate_parameters = ansatz.encoded_to_gate_params()
            best_gate_parameters = {key: value.cpu().item() for key, value in best_gate_parameters.items()}
            best_parameters_realimag = parameter_mapping_realimag.gate_params_to_encoded(best_gate_parameters)
            
            best_prepared_state = prepared_state_detatched.copy()
            best_torch_parameters = ansatz.clone_torch_parameters()
            patience_counter = 0
        else:
            patience_counter += 1
        
        training_time += time.time() - step_start_time
        step_start_time = time.time()
        
        new_result_df = pd.Series({
            "step": step,
            "state_name": config.state_name,
            "n_qudits": config.n_qudits,
            "d": config.d,
            "n_blocks": config.n_blocks,
            "best_fidelity_global": best_fidelity,
            "fidelity_global": global_fidelity_torch,
            "fidelity_overlap": overlap_fidelity,
            "overlap_norm": overlap_norm,
            "fidelity_overlap_normalized": normalized_overlap_fidelity,
            "training_time": training_time,
            "trial": config.trial,
            #**best_parameters_realimag,
        })
        
        if result_df is None:
            result_df = pd.DataFrame(index = range(initial_train_step, max_train_step), columns = new_result_df.index)
        
        result_df.loc[step] = new_result_df
        
        if step % 100 == 0 or step == max_train_step:
            fidelity_MQT = np.nan
            new_time_value, new_time_unit = seconds_to_biggest_unit(training_time)
            
            # writer.add_scalar("Fidelity/global", global_fidelity_torch, step)
            # writer.add_scalar("Fidelity/overlap", overlap_fidelity, step)
            # writer.add_scalar("Fidelity/normalized_overlap", normalized_overlap_fidelity, step)
            # writer.add_scalar("Norm/overlap", overlap_norm, step)
            # writer.add_scalar("Norm/leakage", leakage_norm, step)
            # writer.add_scalar("Loss", loss.item(), step)
            # result_df.to_csv(results_folder + result_file_name + ".csv", index = False)
            
            if time.time() - last_checkpoint_time >= checkpoint_interval_sec or step == max_train_step:
                checkpoint_dict = {
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "torch_parameters": ansatz.clone_torch_parameters(),
                    "best_fidelity": best_fidelity,
                    "best_torch_parameters": best_torch_parameters,
                    "best_parameters_realimag": best_parameters_realimag,
                    "best_prepared_state": best_prepared_state,
                    "patience_counter": patience_counter,
                    "training_time": training_time,
                    "result_df": result_df #split_dataframe_into_chunks(result_df, target_chunk_size_mb = 100)
                }
                save_checkpoint_grouped(step, config.results_folder, result_file_name + f"-checkpoint", checkpoint_dict)
                last_checkpoint_time = time.time()
            
            logging.info(result_file_name + f" - Step {step}, Loss = {loss.item():.2E}, Fidelity Global = {global_fidelity_torch:.6f}, (MQT = {fidelity_MQT:.6f}), "
                                     f"F. Overlap = {overlap_fidelity:.6f}, Overlap Norm = {overlap_norm:.6f}, F. Overlap Normalized = {normalized_overlap_fidelity:.6f}, "
                                     f"Elapsed Time = {new_time_value:.2f} {new_time_unit}")
        
        if patience_counter >= patience_max:
            logging.info(result_file_name + f" - Terminating at {step}, best fidelity was = {best_fidelity:.6f}, current is = {global_fidelity_torch:.6f}")
            break
        
        #################################################################
        #### Backpropagation
        loss.backward()
        
        # Update parameters
        optimizer.step()
    
    #writer.close()
    
    data_dict_to_save["result_df"] = result_df #split_dataframe_into_chunks(result_df, target_chunk_size_mb = 100)
    data_dict_to_save["best_fidelity"] = best_fidelity
    data_dict_to_save["best_prepared_state"] = best_prepared_state
    data_dict_to_save["best_torch_parameters"] = best_torch_parameters
    data_dict_to_save["best_parameters_realimag"] = best_parameters_realimag
    dataIO.save_data(result_file_name + "-best_parameters", data_dict_to_save = data_dict_to_save)
    
    # Save all but the parameters
    #result_df = result_df.drop(columns = list(best_parameters_realimag.keys()))
    #result_df.to_csv(config.dataframe_folder + result_file_name + "-dataframe.csv", index = False)

    return best_fidelity, best_parameters_realimag


class ExperimentCase:
    def __init__(self,
                 experiment_folder,
                 results_folder,
                 state_name,
                 initial_state,
                 loss_type,
                 n_qudits,
                 n_bumpers_per_d,
                 n_bumpers,
                 d,
                 blocks_policy,
                 trial,
                 backend,
                 device):
        self.experiment_folder = experiment_folder
        self.results_folder = results_folder
        self.state_name = state_name
        self.initial_state = initial_state
        self.loss_type = loss_type
        self.n_qudits = n_qudits
        self.n_bumpers_per_d = n_bumpers_per_d
        self.n_bumpers = n_bumpers
        self.d = d
        self.blocks_policy = blocks_policy
        self.trial = trial
        self.backend = backend
        self.device = device
    
    def __repr__(self):
        return (f"ExperimentCase(state_name={self.state_name}, "
                f"n_qudits={self.n_qudits}, d={self.d}, "
                f"n_bumpers={self.n_bumpers}, trial={self.trial})")

