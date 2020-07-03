from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import time
from edNEGmodel.edNEGmodel import *
from solve_edNEGmodel import solve_edNEGmodel

start_time = time.time()

I_stim = 36e-12 # [A]
alpha = 2
t_dur = 30       # [s]
stim_start = 10
stim_end = 20

sol, my_cell = solve_edNEGmodel(t_dur, alpha, I_stim, stim_start, stim_end)
t = sol.t

phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = my_cell.membrane_potentials()
E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = my_cell.reversal_potentials()

q_sn = my_cell.total_charge([my_cell.Na_sn[-1], my_cell.K_sn[-1], my_cell.Cl_sn[-1], my_cell.Ca_sn[-1], my_cell.X_sn])
q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1], my_cell.X_se])        
q_sg = my_cell.total_charge([my_cell.Na_sg[-1], my_cell.K_sg[-1], my_cell.Cl_sg[-1], 0, my_cell.X_sg])        
q_dn = my_cell.total_charge([my_cell.Na_dn[-1], my_cell.K_dn[-1], my_cell.Cl_dn[-1], my_cell.Ca_dn[-1], my_cell.X_dn])
q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1], my_cell.X_de])
q_dg = my_cell.total_charge([my_cell.Na_dg[-1], my_cell.K_dg[-1], my_cell.Cl_dg[-1], 0, my_cell.X_dg])
print("Final values")
print("----------------------------")
print("total charge at the end (C): ", q_sn + q_se + q_sg + q_dn + q_de + q_dg)
print("Q_sn + Q_sg (C):", q_sn+q_sg)
print("Q_se (C): ", q_se)
print("Q_dn + Q_sg (C):", q_dn+q_dg)
print("Q_de (C): ", q_de)
print('total volume (m^3):', my_cell.V_sn[-1] + my_cell.V_se[-1] + my_cell.V_sg[-1] + my_cell.V_dn[-1] + my_cell.V_de[-1] + my_cell.V_dg[-1])
print("----------------------------")
print('elapsed time: ', round(time.time() - start_time, 1), 'seconds')

plt.figure(1)
plt.plot(t, phi_msn*1000, '-', label='soma')
plt.plot(t, phi_mdn*1000, '-', label='dendrite')
plt.title('Neuronal membrane potentials')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

plt.figure(100)
plt.plot(t, phi_msg*1000, '-', label='somatic layar')
plt.plot(t, phi_mdg*1000, '-', label='dendritic layer')
plt.title('Glial membrane potentials')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

plt.show()

