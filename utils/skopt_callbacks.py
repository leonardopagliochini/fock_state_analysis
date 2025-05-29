#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/02/2025

@author: Maurizio Ferrari Dacrema
"""
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import unittest
import shutil

from collections import namedtuple

class HistoryDataframeCallback:
    def __init__(self, total, label, parameter_labels, history_df = None, save_dir="history_chunks", chunk_size=100000, shared_dictionary = {}, state_size = None):
        """Callback to track optimization progress, store parameters, and report the best fidelity."""
        self.total = total
        self.pbar = tqdm(total=total, desc=label, unit="iter", leave=True, dynamic_ncols=True)
        self.best_fidelity = 0  # Best fidelity found so far
        self.best_params = None  # Best parameters
        self.update_interval = min(self.total, 10000)  # Control update frequency
        self.parameter_labels = parameter_labels
        self.save_dir = save_dir
        self.chunk_size = self.total if chunk_size is None else chunk_size
        self.current_chunk = 0
        self.shared_dictionary = shared_dictionary
        
        assert state_size is not None
        self.state_size = state_size

        self.iteration = 0
        self.current_row = 0
        
        # # Define the column types
        # column_types = {param: float for param in parameter_labels}  # Set all parameter columns as float
        # column_types["fidelity"] = float  # Set fidelity as float
        # column_types["prepared_state"] = object  # Store prepared_state as an object (array, complex, etc.)

        self.history_df = pd.DataFrame(index=range(self.chunk_size), columns=[*self.parameter_labels,
                                                                              "fidelity",
                                                                              *[f"prepared_state_real_{index}" for index in range(self.state_size)],
                                                                              *[f"prepared_state_imag_{index}" for index in range(self.state_size)]], dtype = float)
        
        # Create directory if not exists
        os.makedirs(self.save_dir, exist_ok=True)

        # If a history_df is provided, initialize with it
        if history_df is not None:
            self._initialize_with_existing_history(history_df)

    def n_cases_left(self):
        return self.total - self.iteration

    def _initialize_with_existing_history(self, history_df):
        """Handles cases where an existing history_df is provided at initialization."""
        num_rows = len(history_df)

        if num_rows > self.chunk_size:
            # Split into chunks and save immediately
            for i in range(0, num_rows, self.chunk_size):
                if num_rows >= i + self.chunk_size:
                    chunk = history_df.iloc[i:i + self.chunk_size]
                    chunk_path = os.path.join(self.save_dir, f"history_chunk_{self.current_chunk}.csv")
                    chunk.to_csv(chunk_path, index=False)
                    self.current_chunk += 1

            # Keep only the remaining portion in memory (if any)
            remaining_rows = num_rows % self.chunk_size
            if remaining_rows > 0:
                self.history_df.iloc[:remaining_rows] = history_df.iloc[-remaining_rows:].values
                self.current_row = remaining_rows

        else:
            # If history_df fits within one chunk, store it directly
            self.history_df.iloc[:num_rows] = history_df.values
            self.current_row = num_rows


    def _save_current_chunk(self):
        """Saves the current history chunk to disk."""
        chunk_path = os.path.join(self.save_dir, f"history_chunk_{self.current_chunk}.csv")
        self.history_df.iloc[:self.current_row].to_csv(chunk_path, index=False)
        self.current_chunk += 1  # Move to next chunk

    def __call__(self, res):
        """Update progress and check for new best fidelity."""
        current_params = res.x_iters[-1]  # Last evaluated parameters
        current_fidelity = 1 - res.func_vals[-1]  # Convert cost to fidelity
        
        assert np.isclose(current_fidelity, self.shared_dictionary["fidelity"])

        # Store parameters and fidelity
        prepared_state = self.shared_dictionary["prepared_state"].squeeze()
        self.history_df.iloc[self.current_row] = current_params + [current_fidelity] + list(np.real(prepared_state)) + list(np.imag(prepared_state))
        self.current_row += 1


        # Save chunk and clear memory if full
        if self.current_row >= self.chunk_size:
            self._save_current_chunk()
            self.current_row = 0  # Reset row index
            self.history_df = pd.DataFrame(index=range(self.chunk_size), columns=[*self.parameter_labels,
                                                                              "fidelity",
                                                                              *[f"prepared_state_real_{index}" for index in range(self.state_size)],
                                                                              *[f"prepared_state_imag_{index}" for index in range(self.state_size)]], dtype = float)
        
        # Check if we found a new best fidelity
        if current_fidelity > self.best_fidelity:
            self.best_fidelity = current_fidelity
            self.best_params = current_params
            # tqdm.write(f"New best fidelity found: {self.best_fidelity:.6f}")

        # Update progress bar
        self.iteration += 1
        if self.iteration % self.update_interval == 0 or self.iteration == self.total:
            # Update progress bar with the best fidelity and iteration details
            self.pbar.set_postfix({
                "Fidelity (best)": f"{self.best_fidelity:.6f}",
                "Fidelity (current)": f"{current_fidelity:.6f}",
            })

            self.pbar.update(self.update_interval)

    def close(self):
        """Close the progress bar properly."""
        self.pbar.close()
        shutil.rmtree(self.save_dir)

    # def get_history(self):
    #     """Converts the stored history into a pandas DataFrame."""
    #     return self.history_df.copy()

    def get_history(self):
        """Loads all chunks and returns the full history DataFrame."""
        all_chunks = []

        # Load all chunks from disk
        for chunk_index in range(self.current_chunk):
            chunk_path = os.path.join(self.save_dir, f"history_chunk_{chunk_index}.csv")
            if os.path.exists(chunk_path):
                all_chunks.append(pd.read_csv(chunk_path))

        # Add the remaining in-memory data
        if self.current_row > 0:
            all_chunks.append(self.history_df.iloc[:self.current_row])
            
        empty_history_df = pd.DataFrame(columns=[*self.parameter_labels,
                                                  "fidelity",
                                                  *[f"prepared_state_real_{index}" for index in range(self.state_size)],
                                                  *[f"prepared_state_imag_{index}" for index in range(self.state_size)]], dtype = float)
        
        return pd.concat(all_chunks, ignore_index=True) if all_chunks else empty_history_df


class ClearInternalHistoryCallback:
    def __init__(self, keep_last=10000):
        self.keep_last = keep_last  # Keep only the last 500 results

    def __call__(self, res):
        if len(res.x_iters) > self.keep_last:
            res.x_iters = res.x_iters[-self.keep_last:]
            res.func_vals = res.func_vals[-self.keep_last:]


FakeSkoptState = namedtuple("FakeSkoptState", ["x_iters", "func_vals"])

class TestChunkedHistoryCallback(unittest.TestCase):
    def setUp(self):
        """Set up a clean test environment before each test."""
        self.test_dir = "test_history_chunks"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)  # Remove any existing test directory
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up the test directory after each test."""
        shutil.rmtree(self.test_dir)

    def test_initialization_empty(self):
        """Test initializing with no history."""
        state_size = 2
        callback = HistoryDataframeCallback(save_dir=self.test_dir, chunk_size=100, total=1000, label="test case", parameter_labels=["param1", "param2"], state_size = state_size)
        self.assertEqual(len(callback.get_history()), 0)

    def test_initialization_with_small_history(self):
        """Test initializing with a small history_df."""
        
        state_size = 2
        history_df = pd.DataFrame({
            "param1": [1, 2, 3],
            "param2": [4, 5, 6],
            "fidelity": [0.9, 0.8, 0.7],
            "prepared_state_real_0": [1.0, 2.0, 3.0],  # First component (real)
            "prepared_state_real_1": [0.5, 1.5, 2.5],  # Second component (real)
            "prepared_state_imag_0": [-1.0, -2.0, -3.0],  # First component (imaginary)
            "prepared_state_imag_1": [-0.5, -1.5, -2.5],  # Second component (imaginary)
        })
        
        callback = HistoryDataframeCallback(save_dir=self.test_dir, chunk_size=100, history_df=history_df, total=1000, label="test case", parameter_labels=["param1", "param2"], state_size = state_size)

        history_retrieved = callback.get_history()
        pd.testing.assert_frame_equal(history_df.astype(float), history_retrieved.astype(float))
        
    def test_initialization_with_large_history(self):
        """Test initializing with a large history_df that exceeds chunk size."""
        state_size = 2
        history_df = pd.DataFrame({
            "param1": range(250),
            "param2": range(250, 500),
            "fidelity": range(500, 750),
            "prepared_state_real_0": range(250),  # First component (real)
            "prepared_state_real_1": range(500, 750),  # Second component (real)
            "prepared_state_imag_0": range(750, 1000),  # First component (imaginary)
            "prepared_state_imag_1": range(1000, 1250),  # Second component (imaginary)
        })
        
        callback = HistoryDataframeCallback(save_dir=self.test_dir, chunk_size=100, history_df=history_df, total=1000, label="test case", parameter_labels=["param1", "param2"], state_size=state_size)
        history_retrieved = callback.get_history()
        
        # Check that chunks were created
        self.assertEqual(len(os.listdir(self.test_dir)), 2)  # Expect 2 saved chunks
        self.assertEqual(len(callback.get_history()), 250)  # Should match the original history
        
        pd.testing.assert_frame_equal(history_df.astype(float), history_retrieved.astype(float))

    def test_storing_data(self):
        """Test storing data and automatic chunk saving."""
        state_size = 2
        shared_dictionary = {}
        callback = HistoryDataframeCallback(save_dir=self.test_dir, chunk_size=10, total=1000, label="test case", parameter_labels=["param1", "param2"], shared_dictionary = shared_dictionary, state_size=state_size)

        for i in range(25):
            shared_dictionary["fidelity"] = 1-1/100
            shared_dictionary["prepared_state"] = np.array([complex(1.0 * i, -1.0 * i) for i in range(state_size)])
            res = FakeSkoptState([[i, i * 2]], [1/100])   # Add dummy data
            callback(res)  # Add dummy data

        # Ensure chunking occurred (2 full chunks saved, 1 partial)
        self.assertEqual(len(os.listdir(self.test_dir)), 2)
        self.assertEqual(len(callback.get_history()), 25)

    def test_history_retrieval_after_saving(self):
        """Test retrieving history after multiple chunk saves."""
        state_size = 2
        shared_dictionary = {}
        callback = HistoryDataframeCallback(save_dir=self.test_dir, chunk_size=10, total=1000, label="test case", parameter_labels=["param1", "param2"], shared_dictionary = shared_dictionary,  state_size=state_size)

        # Store 50 entries (5 full chunks)
        for i in range(50):
            shared_dictionary["fidelity"] = 1-1/100
            shared_dictionary["prepared_state"] = np.array([complex(1.0 * i, -1.0 * i) for i in range(state_size)])
            res = FakeSkoptState([[i, i * 2]], [1/100])   # Add dummy data
            callback(res)  # Add dummy data

        retrieved_history = callback.get_history()
        self.assertEqual(len(retrieved_history), 50)
        self.assertEqual(len(os.listdir(self.test_dir)), 5)

        # Ensure values match what was stored
        self.assertEqual(retrieved_history.iloc[0]["param1"], 0)
        self.assertEqual(retrieved_history.iloc[-1]["param1"], 49)
        

    # def test_large_scale_efficiency(self):
    #     """Test behavior with a very large number of entries."""
    #     callback = HistoryDataframeCallback(save_dir=self.test_dir, chunk_size=1000, total=10000, label="test case", parameter_labels=["param1", "param2"])
    #
    #     for i in range(5000):  # 5 full chunks
    #         res = FakeSkoptState([[i, i * 2]], [1/100])   # Add dummy data
    #         callback(res)  # Add dummy data
    #
    #     self.assertEqual(len(os.listdir(self.test_dir)), 5)  # Ensure correct chunk count
    #     self.assertEqual(len(callback.get_history()), 5000)

if __name__ == "__main__":
    unittest.main()
