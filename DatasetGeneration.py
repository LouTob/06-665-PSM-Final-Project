# ===============================================================================================================================================
import os
from os.path import join
import numpy as np
from precomputed import * # PRECOMPUTED PARAMETERS DEFINED HERE
from util import generate_inputs, plot_ode_solutions
from ODE import generate_ode_data


# ===============================================================================================================================================
# NOTE: PARAMS (Number of data & save configuration)

N = 100
solutions_name = str(N) + "_solutions.npy"
F_ag_array_name = str(N) + "_F_ag.npy"
save_dir = "./saved"
os.makedirs (save_dir, exist_ok=True)


# ===============================================================================================================================================
if __name__ == "__main__":

    # SOLVE
    random_inputs = generate_inputs (N)
    solutions, F_ag_array = generate_ode_data(random_inputs, num_timesteps=500)

    print ("="*50)
    print ("Saved: ")
    print (f"{solutions_name}: {solutions.shape}")
    print (f"{F_ag_array_name}: {F_ag_array.shape}")
    print ("="*50)

    # SAVE
    np.save (join(save_dir, solutions_name), solutions)
    np.save (join(save_dir, F_ag_array_name), F_ag_array)

    # PLOT
    plot_ode_solutions (solutions)


# ===============================================================================================================================================
