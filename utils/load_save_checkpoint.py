#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/04/2025

@author: Maurizio Ferrari Dacrema
"""

import os, re, time, torch, pickle
import pandas as pd

def split_dataframe_into_chunks(df, target_chunk_size_mb = 100):
    """
    Splits a DataFrame into a dictionary of smaller DataFrames.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to split.
    chunk_size : int
        The maximum number of rows per chunk.

    Returns:
    --------
    dict
        A dictionary where keys are chunk indices (0, 1, 2, ...) and values are DataFrame chunks.
    """
    # Estimate memory usage per row
    total_memory_bytes = df.memory_usage(deep=True).sum()
    n_rows = len(df)

    if n_rows == 0:
        chunk_size = 1  # Avoid division by zero, arbitrary small chunk
    else:
        avg_row_size_bytes = total_memory_bytes / n_rows
    
        # Convert target size to bytes
        target_chunk_size_bytes = target_chunk_size_mb * 1024 * 1024
    
        # Calculate how many rows fit in target size
        estimated_chunk_size = int(target_chunk_size_bytes / avg_row_size_bytes)
    
        # Safety check: at least 1 row per chunk
        chunk_size = max(1, estimated_chunk_size)
    
    
    n_rows = df.shape[0]
    n_chunks = (n_rows + chunk_size - 1) // chunk_size  # Ceiling division

    chunks = {}
    for i in range(n_chunks):
        start_idx = i * target_chunk_size_mb
        end_idx = min(start_idx + target_chunk_size_mb, n_rows)
        chunks[f"chunk-{i}"] = df.iloc[start_idx:end_idx].copy()

    return chunks

def merge_chunks_to_dataframe(chunks_dict):
    """
    Merges a dictionary of DataFrame chunks back into a single DataFrame.

    Parameters:
    -----------
    chunks_dict : dict
        A dictionary where keys are integers (chunk indices) and values are pandas DataFrames.

    Returns:
    --------
    pd.DataFrame
        The unified DataFrame, ordered by chunk indices.
    """
    if not chunks_dict:
        return pd.DataFrame()  # Return empty DataFrame if input is empty

    # Ensure chunks are merged in order of their keys
    ordered_chunks = [chunks_dict[i] for i in sorted(chunks_dict.keys())]

    full_df = pd.concat(ordered_chunks, ignore_index=True)
    return full_df


def load_latest_checkpoint(results_folder, result_file_name):
    pattern = re.compile(re.escape(result_file_name) + r"-step=(\d+)\.zip$")
    candidates = []

    for fname in os.listdir(results_folder):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            candidates.append((step, fname))

    if not candidates:
        raise FileNotFoundError()

    # Sort descending by step
    candidates.sort(reverse=True)

    for step, fname in candidates:
        checkpoint_path = os.path.join(results_folder, fname)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            return checkpoint
        except (EOFError, pickle.UnpicklingError, RuntimeError) as e:
            print(f"Warning: Failed to load {fname} due to {type(e).__name__}: {e}")
            corrupted_path = checkpoint_path.replace(".zip", "-corrupted.zip")
            try:
                os.rename(checkpoint_path, corrupted_path)
                print(f"Renamed {fname} to {os.path.basename(corrupted_path)}")
            except Exception as rename_err:
                print(f"Could not rename {fname}: {rename_err}")

    raise FileNotFoundError()

def save_checkpoint_grouped(step, results_folder, result_file_name, checkpoint_dict):
    """
    Save a checkpoint at an arbitrary step.
    When entering a new 1000-block (e.g., from 999 → 1000), delete all prior checkpoints
    in the previous block (e.g., steps 0–999).
    """

    import re
    start_time = time.time()

    # Save current checkpoint
    checkpoint_name = f"{result_file_name}-step={step}.zip"
    checkpoint_path = os.path.join(results_folder, checkpoint_name)
    torch.save(checkpoint_dict, checkpoint_path)

    # Define previous 1000-block
    current_block = (step // 1000) * 1000
    previous_block = current_block - 1000

    if previous_block >= 0:
        pattern = re.compile(re.escape(result_file_name) + r"-step=(\d+)\.zip$")
        candidates = []

        for fname in os.listdir(results_folder):
            match = pattern.match(fname)
            if match:
                s = int(match.group(1))
                if previous_block <= s < current_block:
                    candidates.append((s, fname))

        # Keep only the most recent in that block
        if candidates:
            max_step, keep_fname = max(candidates, key=lambda x: x[0])
            for s, fname in candidates:
                if fname != keep_fname:
                    fpath = os.path.join(results_folder, fname)
                    try:
                        os.remove(fpath)
                        print(f"removed {fpath}")
                    except Exception as e:
                        print(f"Warning: could not delete {fpath} due to: {e}")

    print(f"Checkpoint saved at step {step} ({checkpoint_path}). Took {time.time() - start_time:.0f} sec")


