# ===============================================================================================================================================
import numpy as np
from scipy.integrate import solve_ivp
from precomputed import * # PRECOMPUTED PARAMETERS DEFINED HERE
from util import GenerateRandomF_agInput


# ===============================================================================================================================================
# NOTE: ODE and Solver

def odes (t, variables, F_ag):
    """
    Fermentation Reaction System ODEs

    * t         : Independent time variable
    * variables : Set of dependent variables
    * F_ag      : Control input array (num_timesteps,)
    
    """
    Cx, Cp, Cs, Co2, T_r, T_ag = variables

    def get_kla (T_r):
        """
        * T_r     : Reaction Temperature in Kelvin
        """
        T_r = T_r - 273 # K -> deg C
        # (Eq 27) Equilibrium conc of O2 in H2O
        Cstar_0 = 14.6 - 0.3943*T_r + 7.714e-3*(T_r**2) - 6.46e-5*(T_r**3)
        # (Eq 28)
        Cstar = Cstar_0 * 10**(-global_eff) 
        # (Eq 29)
        kla = kla_0 * 1.024**(T_r-20)
        return kla, Cstar

    # (Eq 31)
    # mu_x: max specific growth rate as a function of T_r
    mu_x = lambda T_r: A_1*np.exp(-E_a1/(Rg*T_r)) - A_2*np.exp(-E_a2/(Rg*T_r))

    # dVdt = 0
    dCxdt = mu_x(T_r) * Cx * Cs * np.exp(-K_P*Cp) / (K_S+Cs) - F_e*Cx/V
    dCpdt = mu_P * Cx * Cs * np.exp(-K_P1*Cp) / (K_S1+Cs) - F_e*Cp/V
    dCsdt = (-mu_x(T_r) * Cx * Cs * np.exp(-K_P*Cp) / (K_S+Cs) / R_SX) - (mu_P * Cx * Cs * np.exp(-K_P1*Cp) / (K_S1+Cs) / R_SP) + (F_i*c_Sin - F_e*Cs)/V
    
    # (Eq 30)
    r_o2 = mu_O2 * Cx * Co2 / (Y_O2 * (K_O2+Co2))
    kla, Cstar = get_kla (T_r)

    # (Eq 36)
    dCo2dt = kla * (Cstar-Co2) - r_o2
    
    # (Eq 37,38)
    dTrdt = (F_i*T_in - F_e*T_r)/V + r_o2*deltaH_r/(32*rho_r*C_heatr) + (K_T*A_T*(T_r-T_ag))/(V*rho_r*C_heatr)
    dTagdt = F_ag[int(t)-1]*(T_inag-T_ag)/V_j + (K_T*A_T*(T_r-T_ag))/(V_j*rho_ag*C_heatag)

    return [dCxdt, dCpdt, dCsdt, dCo2dt, dTrdt, dTagdt]


def solve_sys (vars0, F_ag, num_timesteps=500):
    """ ODE solver """
    tspan = (0, num_timesteps) # hrs
    teval = np.linspace (*tspan, num_timesteps)
    sol = solve_ivp (odes, tspan, vars0, t_eval=teval, args = (F_ag,))
    return sol.y, sol.success


# ===============================================================================================================================================
# NOTE: ODE Dataset

def generate_ode_data(random_inputs, num_timesteps=500):
    """
    [Input]
    * random_inputs : Initial guess on ODEs. Output from generate_inputs (N,6)
    
    [Return]
    * F_ag          : Control inputs used to solve the ODEs (num_timesteps,)
    * solutions     : ODE solutions of six states (num_soln, 6, num_timesteps)
    
    """
    # F_ag_array generation for each of solution data
    N, _ = random_inputs.shape
    F_ag_array = np.zeros((N, num_timesteps))
    for i in range(N):
        F_ag_array[i,:] = GenerateRandomF_agInput(num_timesteps)[:num_timesteps]
    
    # Solve and collect solutions
    solutions = []
    print (f"num_timesteps = {num_timesteps}")

    for i, vars0 in enumerate(random_inputs):
        F_ag = F_ag_array[i] # control input at t
        
        y, success = solve_sys(vars0, F_ag, num_timesteps)
        solutions.append (y)

        print (f"{i+1}/{random_inputs.shape[0]}")
        print (f"ODE solution found: {success}")
                
    solutions = np.array(solutions)
    return solutions, F_ag_array


# ===============================================================================================================================================
