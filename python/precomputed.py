# ===============================================================================================================================================
#  -------------------
# | NOTE: ALL PARAMS  |
#  -------------------

# Inorganic salts in the reaction medium
m_NaCl = 500   # g
m_CaCO3 = 100  # g
m_MgCl2 = 100  # g
pH = 6.        # pH of liquid phase

# Flow conditions (flow, conc, temp)
F_i = 51.           # L/h
c_Sin = 60.         # g/L
T_in = 273 + 25.    # K
T_inag = 273 + 15.  # K
F_e = 51.           # L/h

# Arrhenius pre-exponential factors
A_1 = 9.5e8
A_2 = 2.55e33
Rg = 8.314    # J/mol K

# Heat transfer coefficients
A_T = 1          # area, m2
C_heatag = 4.18  # J/g/K
C_heatr = 4.18   # J/g/K
E_a1 = 55000.    # J/mol
E_a2 = 220000.   # J/mol

# Specific ionic constants
H_Na = -0.550
H_Ca = -0.303
H_Mg = -0.314
H_H = -0.774
H_Cl = 0.844
H_CO3 = 0.485
H_HO = 0.941

# Mass transfer & Reaction-related
kla_0 = 38      # 1/hr; product of mass-transfer coefficient at 20 C for O2 and gas-phase specific area
K_O2 = 8.86     # mg/L
K_P =  0.139    # g/L
K_P1 = 0.070    # g/L
K_S = 1.030     # g/L
K_S1 = 1.680    # g/L
K_T = 3.6e5     # J/(h m2 K)
R_SP = 0.435    # (ethanol produced)/(glucose consumed)
R_SX = 0.607    # (biomass produced)/(glucose consumed)
Y_O2 = 0.970    # (O2 consumed)/(biomass produced)
V = 1000        # L; Reaction volume
V_j = 50        # L; Reactor jacket volume
deltaH_r = 518  # kJ/(mol O2 consumed)
mu_O2 = 0.5     # 1/hr; maximum specific oxygen consumption rate
mu_P = 1.790    #1/hr; maximum specific fermentation rate
rho_ag = 1000   # g/L; cooling agent density
rho_r = 1080    # g/L; reaction mass density

# Molar masses (g/mol)
M_Na = 22.99 
M_NaCl = 58.44
M_Ca = 40.08
M_CaCO3 = 100.09
M_Mg = 24.31
M_MgCl2 = 95.21
M_Cl = 35.45
M_CO3 = 60.01

# ===============================================================================================================================================
# NOTE: Precompute Section
import numpy as np

# molar concentration calculations (ions)
c_Na = m_NaCl*M_Na / (M_NaCl*V)
c_Ca = m_CaCO3*M_Ca / (M_CaCO3*V)
c_Mg = m_MgCl2*M_Mg / (M_MgCl2*V)
c_Cl = (m_NaCl/M_NaCl + 2*m_MgCl2/M_MgCl2) * M_Cl/V
c_CO3 = m_CaCO3*M_CO3 / (M_CaCO3*V)

# concentrations from pH
c_H = 10**(-pH)
c_OH = 10**(pH-14)

# Ionic strength calculations (Eq 18)
I_Na = 0.5*c_Na
I_Ca = 0.5*c_Ca*(2**2)
I_Mg = 0.5*c_Mg*(2**2)
I_Cl = 0.5*c_Cl
I_CO3 = 0.5*c_CO3*(2**2)
I_H = 0.5*c_H
I_OH = 0.5*c_OH

# global effect of ionic strengths (Eq 26)
ionic_strengths = np.array ([I_Na, I_Ca, I_Mg, I_Cl, I_CO3, I_H, I_OH])
Hs = np.array ([H_Na, H_Ca, H_Mg, H_Cl, H_CO3, H_H, H_HO])
global_eff = np.dot (ionic_strengths, Hs)

# ===============================================================================================================================================
