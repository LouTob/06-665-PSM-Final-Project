# ===============================================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import random


# ===============================================================================================================================================
# NOTE: Random control input and ODE initial value generation

def GenerateRandomF_agInput(num_timesteps=500, plot=False):
    """
    For random periodic F_ag input generation
    """
    F_ag_random_sequence = []
    i = 0
    while i < num_timesteps:
        if i+1 % random.randint(5, 50) == 0:
            F_ag_random_sequence.append(random.randint(80, 200))
            temp_stable_input = random.randint(0, 20)
            temp_stable_input_length = random.randint(3,40)
            for j in range(temp_stable_input_length):
                F_ag_random_sequence.append(temp_stable_input)
            i += temp_stable_input_length
        else:
            F_ag_random_sequence.append(random.randint(0, 20))
            i += 1

    # shape (>num_timesteps,); Cut down to (num_timesteps,) in generate_ode_data
    F_ag_random_sequence = np.array(F_ag_random_sequence).astype(float)
    # F_ag_random_sequence[0] = 0

    # Plot sequence
    if plot:
      plt.plot(F_ag_random_sequence)
      plt.xlabel('Time')
      plt.ylabel('F_ag')
      plt.title('Randomly Generated Input Sequence')
      plt.show()
    return F_ag_random_sequence # (num_timesteps,)


def generate_inputs (N, Cx_low = 5, Cx_high = 60, Cs_low = 5, Cs_high = 60, \
                    T_in_low = 293, T_in_high = 308, T_inag_low = 293, T_inag_high = 308, \
                    stable_initial_conditions = False):
    """
    Randomly generates N of [Cx0, Cp0, Cs0, Co20, Tr0, Tag0] initial conditions.
    Range of reasonable initial values are manually chosen in the function.
    
    [Return]
    random_inputs: Array of random initial conditions (N,6)

    """
    Cx_range = [Cx_low, Cx_high]     # 60 g/L given on page 99
    Cs_range = [Cs_low, Cs_high]     # 60 g/L given on page 99
    T_in_range = [T_in_low, T_in_high]  # random in [298,350] K
    T_inag_range = [T_inag_low, T_inag_high]  # random in [298,350] K

    Cx_in = np.random.randint (*Cx_range, size=(N,1))
    Cs_in = np.random.randint (*Cs_range, size=(N,1))



    T_in = np.random.randint (*T_in_range, size=(N,1))
    T_inag = np.random.randint (*T_inag_range, size=(N,1))


    #### If we want to fix the initial conditions tighter, use the code below
    if stable_initial_conditions == True:
        T_in = np.ones((N,1)) * 298 # given on page 99
        T_inag = np.ones((N,1)) * 288 # given on page 99
        Cs_in = np.ones((N,1)) * 60 # 60 g/L given on page 99
        cx_range = [0.5, 1.5]     # inferred from graphs on page 101
        Cx_in = np.random.randint (*cx_range, size=(N,1))

    # Product starts with 0
    Cp_in = np.zeros ((N,1))
    Co2_in = np.zeros ((N,1))
    
    random_inputs = np.hstack ([Cx_in, Cp_in, Cs_in, Co2_in, T_in, T_inag]).astype (float)
    return random_inputs # N x 6


# ===============================================================================================================================================
# NOTE: Plot solutions from generate_ode_data

def plot_ode_solutions (solutions):
    """
    solutions : ODE solutions of shape (num_solutions, 6, num_timesteps)
    """
    num_soln, _, num_timesteps = solutions.shape
    t = np.arange (0, num_timesteps, 1)

    fig,ax = plt.subplots (1,4, figsize=(18,3))
    
    for i in range (num_soln):
      # [Cx, Cp, Cs, Co2, T_r, T_ag]
      concentrations = solutions[i][0:4]
      temperatures = solutions[i][4:]

      ax[0].plot (t,concentrations[:-1].T) # Cx, Cp, Cs
      ax[1].plot (t,concentrations[-1])  # Co2

      ax[2].plot (t,temperatures[-2]) # Tr
      ax[3].plot (t,temperatures[-1]) # Tag

    ax[0].set_title (r"$C_x, C_p, C_s$ [g/mL]")
    ax[1].set_title (r"$C_{O2}$ [g/mL]")
    ax[2].set_title (r"$T_r$ [K]")
    ax[3].set_title (r"$T_{ag}$ [K]")
    plt.show ()
# ===============================================================================================================================================