import numpy as np


class FunctionGenerator():

    def __init__(self, function_name, d, n_qudit):
        self.function_name = function_name
        self.n_points = np.power(d, n_qudit)
        self.function = np.zeros(self.n_points)
        self.d = d
        self.n_qudit = n_qudit

    def get_function(self):
        x = np.linspace(0, 1, self.n_points, endpoint=False)

        if self.function_name == 'sinc':
            self.function[1:] = np.sin(np.pi * x[1:]) / (np.pi * x[1:])
            self.function[0] = 1

        elif self.function_name == 'gaussian':
            self.function = np.exp(-(x - 0.5) ** 2 / 2)

        elif self.function_name == 'custom_gaussian':
            self.function = np.exp(-(x - 0.5) ** 2 / (2 * 0.01**2)) / 0.01

        elif self.function_name == 'bimodal_gaussian':
            self.function = (0.1 * np.exp(- (x - 0.25) ** 2 / (2 * 0.3 ** 2)) / 0.3 +
                             (1 - 0.1) * np.exp(- (x - 0.75) ** 2 / (2 * 0.04 ** 2)) / 0.04)

        elif self.function_name == 'xlogx':
            self.function[1:] = x[1:] * np.log(x[1:])
            self.function[0] = 0

        elif self.function_name == 'lorentzian':
            self.function = 1 / (1 + 4 * (x - 0.5) ** 2)

        elif self.function_name == 'linear':
            self.function = x
            self.function[0] = 0

        elif self.function_name == 'shifted_sinc':
            shifted_x = x - 0.5
            self.function= np.sin(19 * np.pi * shifted_x) / (19 * np.pi * shifted_x)
            self.function[len(self.function)//2] = 1

        elif self.function_name == 'ghz':
            assert self.n_qudit >= 2
            self.function = np.zeros(len(x))
            for k in range(self.d):
                # This is just index = k + k*d^2 + k*d^3 +...
                index = 0
                for n in range(self.n_qudit):
                    index += np.power(self.d, n) * k  # Map k to corresponding index
                self.function[index] = 1 / np.sqrt(self.d)  # Proper normalization

        elif self.function_name == 'non_diff':
            self.function = np.sqrt(np.abs(x-0.5))

        elif self.function_name == 'w_state':
            self.function = np.zeros(len(x))
            n_qubits = np.log2(self.n_points).astype(int)
            indices = [2 ** i for i in range(n_qubits)]
            self.function[indices] = (1 / np.sqrt(n_qubits))

        elif self.function_name == 'random_sparse':
            self.function = np.zeros(len(x))
            self.function = np.zeros(len(x))
            n_qubits = np.log2(self.n_points).astype(int)
            indices = np.random.choice(range(2 ** n_qubits), n_qubits, replace=False)
            self.function[indices] = (1 / np.sqrt(n_qubits))

        elif self.function_name == 'log-normal':
            self.function = np.exp(- (np.log(x[1:]) - 0.25) ** 2 / 2) / x[1:]
            self.function[0] = 0

        elif self.function_name == 'random_i':
            self.function = np.random.random(len(x)) + 1j*np.random.random(len(x))

        elif self.function_name == 'random':
            self.function = np.random.random(len(x))

        else:
            raise ValueError("Function name not recognized")

        return self.function
