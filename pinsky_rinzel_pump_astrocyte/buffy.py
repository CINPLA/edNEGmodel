import numpy as np
#from math import fsum
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import warnings
#warnings.filterwarnings("error")

class Buffy(): 

    def __init__(self, T, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, \
        Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, \
        k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, \
        Ca0_si, Ca0_di, n, h, s, c, q, z):
        
        # temperature [K]
        self.T = T

        # ion concentraions [mol * m**-3]
        self.Na_si = Na_si
        self.Na_se = Na_se
        self.Na_sg = Na_sg
        self.Na_di = Na_di
        self.Na_de = Na_de
        self.Na_dg = Na_dg
        self.K_si = K_si
        self.K_se = K_se
        self.K_sg = K_sg
        self.K_di = K_di
        self.K_de = K_de
        self.K_dg = K_dg
        self.Cl_si = Cl_si
        self.Cl_se = Cl_se 
        self.Cl_sg = Cl_sg 
        self.Cl_di = Cl_di 
        self.Cl_de = Cl_de
        self.Cl_dg = Cl_dg
        self.Ca_si = Ca_si
        self.Ca_se = Ca_se 
        self.Ca_di = Ca_di 
        self.Ca_de = Ca_de
        self.k_res_si = k_res_si
        self.k_res_se = k_res_se
        self.k_res_sg = k_res_sg
        self.k_res_di = k_res_di
        self.k_res_de = k_res_de
        self.k_res_dg = k_res_dg
        self.Ca0_si = Ca0_si
        self.Ca0_di = Ca0_di
        self.free_Ca_si = 0.01*Ca_si
        self.free_Ca_di = 0.01*Ca_di

        # gating variables
        self.n = n
        self.h = h
        self.s = s
        self.c = c
        self.q = q
        self.z = z

        # membrane capacitance [F * m**-2]
        self.C_sm = 3e-2 # Pinsky and Rinzel, 1994
        self.C_dm = 3e-2 # Pinsky and Rinzel, 1994
       
        # volumes and areas
        self.alpha = alpha
        self.A_sn = 616e-12               # [m**2]
        self.A_dn = 616e-12               # [m**2]
        self.A_in = self.alpha*self.A_sn  # [m**2]
        self.A_e = self.A_in/2.           # [m**2]
        self.A_sg = 616e-12               # [m**2]
        self.A_dg = 616e-12               # [m**2]
        self.A_ig = self.alpha*self.A_sg  # [m**2]
        self.V_si = 1437e-18              # [m**3]
        self.V_di = 1437e-18              # [m**3]
        self.V_se = 718.5e-18             # [m**3]
        self.V_de = 718.5e-18             # [m**3]
        self.V_sg = 1437e-18              # [m**3]
        self.V_dg = 1437e-18              # [m**3]
        self.dx = 667e-6                  # [m]

        # diffusion constants [m**2 s**-1]
        self.D_Na = 1.33e-9 # Halnes et al. 2013
        self.D_K = 1.96e-9  # Halnes et al. 2013 
        self.D_Cl = 2.03e-9 # Halnes et al. 2013
        self.D_Ca = 0.71e-9 # Halnes et al. 2016

        # tortuosities
        self.lamda_i = 3.2 # Halnes et al. 2013
        self.lamda_e = 1.6 # Halnes et al. 2013

        # valencies
        self.Z_Na = 1.
        self.Z_K = 1.
        self.Z_Cl = -1.
        self.Z_Ca = 2.

        # constants
        self.F = 9.648e4    # [C * mol**-1]
        self.R = 8.314      # [J * mol**-1 * K**-1] 

        # conductances [S * m**-2]
        self.g_Na_leak = 0.247 # Wei et al. 2014
        self.g_K_leak = 0.5    # Wei et al. 2014
        self.g_Cl_leak = 1.0   # Wei et al. 2014
        self.g_Na = 300.
        self.g_DR = 150.
        self.g_Ca = 118.
        self.g_AHP = 8.
        self.g_C = 150.
        
        # pump strengths
        self.rho = 1.87e-6
        self.U_kcc2 = 7.00e-7
        self.U_nkcc1 = 2.33e-7



    def j_pump(self, Na_i, K_e):
        j = (self.rho / (1.0 + np.exp((25. - Na_i)/3.))) * (1.0 / (1.0 + np.exp(3.5 - K_e)))
        return j

    def j_kcc2(self, K_i, K_e, Cl_i, Cl_e):
        j = self.U_kcc2 * np.log(K_i*Cl_i/(K_e*Cl_e))
        return j
    
    def j_nkcc1(self, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp(16 - K_e))) * (np.log(K_i*Cl_i/(K_e*Cl_e)) + np.log(Na_i*Cl_i/(Na_e*Cl_e)))
        return j

    def j_Na_sn(self, phi_sm, E_Na_s):
        j = self.g_Na_leak*(phi_sm - E_Na_s) / (self.F*self.Z_Na) \
            + 3*self.j_pump(self.Na_si, self.K_se) \
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)        
        return j 

    def j_K_sn(self, phi_sm, E_K_s):
        j = self.g_K_leak*(phi_sm - E_K_s) / (self.F*self.Z_K) \
            - 2*self.j_pump(self.Na_si, self.K_se) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)        
        return j

    def j_Cl_sn(self, phi_sm, E_Cl_s):
        j = self.g_Cl_leak*(phi_sm - E_Cl_s) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + 2*self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j

    def j_Na_dn(self, phi_dm, E_Na_d):
        j = self.g_Na_leak*(phi_dm - E_Na_d) / (self.F*self.Z_Na) \
            + 3*self.j_pump(self.Na_di, self.K_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_K_dn(self, phi_dm, E_K_d):
        j = self.g_K_leak*(phi_dm - E_K_d) / (self.F*self.Z_K) \
            - 2*self.j_pump(self.Na_di, self.K_de) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_Cl_dn(self, phi_dm, E_Cl_d):
        j = self.g_Cl_leak*(phi_dm - E_Cl_d) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + 2*self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_k_diff(self, D_k, tortuosity, k_s, k_d):
        j = - D_k * (k_d - k_s) / (tortuosity**2 * self.dx)
        return j

    def j_k_drift(self, D_k, Z_k, tortuosity, k_s, k_d, phi_s, phi_d):
        j = - D_k * self.F * Z_k * (k_d + k_s) * (phi_d - phi_s) / (2 * tortuosity**2 * self.R * self.T * self.dx)
        return j

    def conductivity_k(self, D_k, Z_k, tortuosity, k_s, k_d): 
        sigma = self.F**2 * D_k * Z_k**2 * (k_d + k_s) / (2 * self.R * self.T * tortuosity**2)
        return sigma

    def total_charge(self, k, k_res, V):
        Z_k = [self.Z_Na, self.Z_K, self.Z_Cl, self.Z_Ca]
        q = 0.0
        for i in range(0, 4):
            q += Z_k[i]*k[i]
        q = self.F*(q + k_res)*V
        return q

    def nernst_potential(self, Z, k_i, k_e):
        E = self.R*self.T / (Z*self.F) * np.log(k_e / k_i)
        return E

    def reversal_potentials(self):
        E_Na_sn = self.nernst_potential(self.Z_Na, self.Na_si, self.Na_se)
        E_Na_sg = 0 #self.nernst_potential(self.Z_Na, self.Na_sg, self.Na_se)
        E_Na_dn = self.nernst_potential(self.Z_Na, self.Na_di, self.Na_de)
        E_Na_dg = 0 #self.nernst_potential(self.Z_Na, self.Na_dg, self.Na_de)
        E_K_sn = self.nernst_potential(self.Z_K, self.K_si, self.K_se)
        E_K_sg = 0 #self.nernst_potential(self.Z_K, self.K_sg, self.K_se)
        E_K_dn = self.nernst_potential(self.Z_K, self.K_di, self.K_de)
        E_K_dg = 0 #self.nernst_potential(self.Z_K, self.K_dg, self.K_de)
        E_Cl_sn = self.nernst_potential(self.Z_Cl, self.Cl_si, self.Cl_se)
        E_Cl_sg = 0 #self.nernst_potential(self.Z_Cl, self.Cl_sg, self.Cl_se)
        E_Cl_dn = self.nernst_potential(self.Z_Cl, self.Cl_di, self.Cl_de)
        E_Cl_dg = 0 #self.nernst_potential(self.Z_Cl, self.Cl_dg, self.Cl_de)
        E_Ca_sn = self.nernst_potential(self.Z_Ca, self.free_Ca_si, self.Ca_se)
        E_Ca_dn = self.nernst_potential(self.Z_Ca, self.free_Ca_di, self.Ca_de)
        return E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn

    def membrane_potentials(self):
        I_n_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di))
        I_g_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.Na_sg, self.Na_dg) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.K_sg, self.K_dg) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_sg, self.Cl_dg))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de))

        sigma_i = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di)
        sigma_g = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.Na_sg, self.Na_dg) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.K_sg, self.K_dg) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_sg, self.Cl_dg)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de)

        q_di = self.total_charge([self.Na_di, self.K_di, self.Cl_di, self.Ca_di], self.k_res_di, self.V_di)
        q_dg = self.total_charge([self.Na_dg, self.K_dg, self.Cl_dg, 0], self.k_res_dg, self.V_dg)
        q_si = self.total_charge([self.Na_si, self.K_si, self.Cl_si, self.Ca_si], self.k_res_si, self.V_si)
        q_sg = self.total_charge([self.Na_sg, self.K_sg, self.Cl_sg, 0], self.k_res_sg, self.V_sg)

        phi_di = q_di / (self.C_dm * self.A_dn)
        phi_de = 0.
        phi_dg = q_dg / (self.C_dm * self.A_dg)
        phi_se = ( - self.dx * self.A_in * I_n_diff + self.A_in * sigma_i * phi_di - self.A_in * sigma_i * q_si / (self.C_sm * self.A_sn) \
            - self.dx * self.A_ig * I_g_diff + self.A_ig * sigma_g * phi_dg - self.A_ig * sigma_g * q_sg / (self.C_sm * self.A_sg) - self.dx * self.A_e * I_e_diff ) \
            / ( self.A_e * sigma_e + self.A_in * sigma_i + self.A_ig * sigma_g )
        phi_si = q_si / (self.C_sm * self.A_sn) + phi_se
        phi_sg = q_sg / (self.C_sm * self.A_sg) + phi_se
        phi_msn = phi_si - phi_se
        phi_msg = phi_sg - phi_se
        phi_mdn = phi_di - phi_de
        phi_mdg = phi_dg - phi_de

        return phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg

    def dkdt(self):
       
        phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
        E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = self.reversal_potentials()

        j_Na_msn = self.j_Na_sn(phi_msn, E_Na_sn)
        j_K_msn = self.j_K_sn(phi_msn, E_K_sn)
        j_Cl_msn = self.j_Cl_sn(phi_msn, E_Cl_sn)

        j_Na_msg = 0 #self.j_Na_s(phi_msg, E_Na_sg)
        j_K_msg = 0 #self.j_K_s(phi_msg, E_K_sg)
        j_Cl_msg = 0 #self.j_Cl_s(phi_msg, E_Cl_sg)

        j_Na_mdn = self.j_Na_dn(phi_mdn, E_Na_dn)
        j_K_mdn = self.j_K_dn(phi_mdn, E_K_dn)    
        j_Cl_mdn = self.j_Cl_dn(phi_mdn, E_Cl_dn)

        j_Na_mdg = 0 #self.j_Na_d(phi_mdg, E_Na_dg)
        j_K_mdg = 0 #self.j_K_d(phi_mdg, E_K_dg)
        j_Cl_mdg = 0 #self.j_Cl_d(phi_mdg, E_Cl_dg)

        j_Na_in = self.j_k_diff(self.D_Na, self.lamda_i, self.Na_si, self.Na_di) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.Na_si, self.Na_di, phi_si, phi_di) 
        j_K_in = self.j_k_diff(self.D_K, self.lamda_i, self.K_si, self.K_di) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.K_si, self.K_di, phi_si, phi_di)
        j_Cl_in = self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_si, self.Cl_di) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_si, self.Cl_di, phi_si, phi_di)
        j_Ca_in = self.j_k_diff(self.D_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_Ca_si, self.free_Ca_di, phi_si, phi_di)

        j_Na_ig = self.j_k_diff(self.D_Na, self.lamda_i, self.Na_sg, self.Na_dg) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.Na_sg, self.Na_dg, phi_sg, phi_dg) 
        j_K_ig = self.j_k_diff(self.D_K, self.lamda_i, self.K_sg, self.K_dg) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.K_sg, self.K_dg, phi_sg, phi_dg)
        j_Cl_ig = self.j_k_diff(self.D_Cl, self.lamda_i, self.Cl_sg, self.Cl_dg) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.Cl_sg, self.Cl_dg, phi_sg, phi_dg)

        j_Na_e = self.j_k_diff(self.D_Na, self.lamda_e, self.Na_se, self.Na_de) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.Na_se, self.Na_de, phi_se, phi_de)
        j_K_e = self.j_k_diff(self.D_K, self.lamda_e, self.K_se, self.K_de) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.K_se, self.K_de, phi_se, phi_de)
        j_Cl_e = self.j_k_diff(self.D_Cl, self.lamda_e, self.Cl_se, self.Cl_de) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.Cl_se, self.Cl_de, phi_se, phi_de)
        j_Ca_e = self.j_k_diff(self.D_Ca, self.lamda_e, self.Ca_se, self.Ca_de) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_e, self.Ca_se, self.Ca_de, phi_se, phi_de)

        dNadt_si = -j_Na_msn*(self.A_sn / self.V_si) - j_Na_in*(self.A_in / self.V_si)
        dNadt_se = j_Na_msn*(self.A_sn / self.V_se) + j_Na_msg*(self.A_sg / self.V_se) - j_Na_e*(self.A_e / self.V_se)
        dNadt_sg = -j_Na_msg*(self.A_sg / self.V_sg) - j_Na_ig*(self.A_ig / self.V_sg)
        dNadt_di = -j_Na_mdn*(self.A_dn / self.V_di) + j_Na_in*(self.A_in / self.V_di)
        dNadt_de = j_Na_mdn*(self.A_dn / self.V_de) + j_Na_mdg*(self.A_dg / self.V_de) + j_Na_e*(self.A_e / self.V_de)
        dNadt_dg = -j_Na_mdg*(self.A_dg / self.V_dg) + j_Na_ig*(self.A_ig / self.V_dg)

        dKdt_si = -j_K_msn*(self.A_sn / self.V_si) - j_K_in*(self.A_in / self.V_si)
        dKdt_se = j_K_msn*(self.A_sn / self.V_se) + j_K_msg*(self.A_sg / self.V_se) - j_K_e*(self.A_e / self.V_se)
        dKdt_sg = -j_K_msg*(self.A_sg / self.V_sg) - j_K_ig*(self.A_ig / self.V_sg)
        dKdt_di = -j_K_mdn*(self.A_dn / self.V_di) + j_K_in*(self.A_in / self.V_di)
        dKdt_de = j_K_mdn*(self.A_dn / self.V_de) + j_K_mdg*(self.A_dg / self.V_de) + j_K_e*(self.A_e / self.V_de)
        dKdt_dg = -j_K_mdg*(self.A_dg / self.V_dg) + j_K_ig*(self.A_ig / self.V_dg)

        dCldt_si = -j_Cl_msn*(self.A_sn / self.V_si) - j_Cl_in*(self.A_in / self.V_si)
        dCldt_se = j_Cl_msn*(self.A_sn / self.V_se) + j_Cl_msg*(self.A_sg / self.V_se) - j_Cl_e*(self.A_e / self.V_se)
        dCldt_sg = -j_Cl_msg*(self.A_sg / self.V_sg) - j_Cl_ig*(self.A_ig / self.V_sg)
        dCldt_di = -j_Cl_mdn*(self.A_dn / self.V_di) + j_Cl_in*(self.A_in / self.V_di)
        dCldt_de = j_Cl_mdn*(self.A_dn / self.V_de) + j_Cl_mdg*(self.A_dg / self.V_de) + j_Cl_e*(self.A_e / self.V_de)
        dCldt_dg = -j_Cl_mdg*(self.A_dg / self.V_dg) + j_Cl_ig*(self.A_ig / self.V_dg)

        dCadt_si = - j_Ca_in*(self.A_in / self.V_si)
        dCadt_se = - j_Ca_e*(self.A_e / self.V_se)
        dCadt_di = j_Ca_in*(self.A_in / self.V_di)
        dCadt_de = j_Ca_e*(self.A_e / self.V_de)

#        dresdt_si = 0
#        dresdt_di = 0
#        dresdt_se = 0
#        dresdt_de = 0

        return dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, \
            dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de

    def dmdt(self):
        phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
        
        dndt = 0
        dhdt = 0
        dsdt = 0
        dcdt = 0
        dqdt = 0
        dzdt = 0

        return dndt, dhdt, dsdt, dcdt, dqdt, dzdt

if __name__ == "__main__":

    T = 309.14
    alpha = 1.

    Na_si0 = 18.
    Na_se0 = 139.
    Na_sg0 = 0.
    K_si0 = 99.
    K_se0 = 5.
    K_sg0 = 0.
    Cl_si0 = 7.
    Cl_se0 = 131.
    Cl_sg0 = 0.
    Ca_si0 = 0.01
    Ca_se0 = 1.1

    Na_di0 = 20.
    Na_de0 = 141.
    Na_dg0 = 0.
    K_di0 = 96.
    K_de0 = 4.
    K_dg0 = 0.
    Cl_di0 = 7.
    Cl_de0 = 131.
    Cl_dg0 = 0.
    Ca_di0 = 0.01
    Ca_de0 = 1.1

#    Cl_si0 = Na_si0 + K_si0 + 2*Ca_si0
#    Cl_se0 = Na_se0 + K_se0 + 2*Ca_se0
#    Cl_sg0 = Na_sg0 + K_sg0
#    Cl_di0 = Na_di0 + K_di0 + 2*Ca_di0
#    Cl_de0 = Na_de0 + K_de0 + 2*Ca_de0
#    Cl_dg0 = Na_dg0 + K_dg0
#
#    k_res_si = 0
#    k_res_se = 0
#    k_res_sg = 0
#    k_res_di = 0
#    k_res_de = 0
#    k_res_dg = 0

    res_i = -66e-3*3e-2*616e-12/(1437e-18*9.648e4)
    res_e = -66e-3*3e-2*616e-12/(718.5e-18*9.648e4)

    k_res_si = Cl_si0 - Na_si0 - K_si0 - 2*Ca_si0 + res_i
    k_res_se = Cl_se0 - Na_se0 - K_se0 - 2*Ca_se0 - res_e
    k_res_sg = Cl_sg0 - Na_sg0 - K_sg0
    k_res_di = Cl_di0 - Na_di0 - K_di0 - 2*Ca_di0 + res_i
    k_res_de = Cl_de0 - Na_de0 - K_de0 - 2*Ca_de0 - res_e
    k_res_dg = Cl_dg0 - Na_dg0 - K_dg0

    n0 = 0.0004
    h0 = 0.999
    s0 = 0.008
    c0 = 0.006
    q0 = 0.011
    z0 = 1.0    

    def dkdt(t,k):

        Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z = k

        my_cell = Buffy(T, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

        dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()

        return dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, \
            dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt 

    start_time = time.time()
    t_span = (0, 30)

    k0 = [Na_si0, Na_se0, Na_sg0, Na_di0, Na_de0, Na_dg0, K_si0, K_se0, K_sg0, K_di0, K_de0, K_dg0, Cl_si0, Cl_se0, Cl_sg0, Cl_di0, Cl_de0, Cl_dg0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, n0, h0, s0, c0, q0, z0]

    init_cell = Buffy(T, Na_si0, Na_se0, Na_sg0, Na_di0, Na_de0, Na_dg0, K_si0, K_se0, K_sg0, K_di0, K_de0, K_dg0, Cl_si0, Cl_se0, Cl_sg0, Cl_di0, Cl_de0, Cl_dg0, \
        Ca_si0, Ca_se0, Ca_di0, Ca_de0, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, Ca_si0, Ca_di0, n0, h0, s0, c0, q0, z0)

    phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = init_cell.membrane_potentials()
    
    E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = init_cell.reversal_potentials()

    q_si = init_cell.total_charge([init_cell.Na_si, init_cell.K_si, init_cell.Cl_si, init_cell.Ca_si], init_cell.k_res_si, init_cell.V_si)
    q_se = init_cell.total_charge([init_cell.Na_se, init_cell.K_se, init_cell.Cl_se, init_cell.Ca_se], init_cell.k_res_se, init_cell.V_se)        
    q_sg = init_cell.total_charge([init_cell.Na_sg, init_cell.K_sg, init_cell.Cl_sg, 0], init_cell.k_res_sg, init_cell.V_sg)        
    q_di = init_cell.total_charge([init_cell.Na_di, init_cell.K_di, init_cell.Cl_di, init_cell.Ca_di], init_cell.k_res_di, init_cell.V_di)
    q_de = init_cell.total_charge([init_cell.Na_de, init_cell.K_de, init_cell.Cl_de, init_cell.Ca_de], init_cell.k_res_de, init_cell.V_de)
    q_dg = init_cell.total_charge([init_cell.Na_dg, init_cell.K_dg, init_cell.Cl_dg, 0], init_cell.k_res_dg, init_cell.V_dg)
    print("----------------------------")
    print("Initial values")
    print("----------------------------")
    print("initial total charge(C): ", q_si + q_se + q_sg + q_di + q_de + q_dg)
    print("Q_si + Q_sg (C):", q_si+q_sg)
    print("Q_se (C): ", q_se)
    print("Q_di + Q_sg (C):", q_di+q_dg)
    print("Q_de (C): ", q_de)
    print("----------------------------")
    print('phi_si: ', round(phi_si*1000))
    print('phi_se: ', round(phi_se*1000))
    print('phi_di: ', round(phi_di*1000))
    print('phi_de: ', round(phi_de*1000))
    print('phi_msn: ', round(phi_msn*1000))
    print('phi_mdn: ', round(phi_mdn*1000))
    print('E_Na_sn: ', round(E_Na_sn*1000))
    print('E_Na_dn: ', round(E_Na_dn*1000))
    print('E_Na_dg: ', round(E_Na_dg*1000))
    print('E_K_sn: ', round(E_K_sn*1000))
    print('E_K_dn: ', round(E_K_dn*1000))
    print('E_K_dg: ', round(E_K_dg*1000))
    print('E_Cl_sn: ', round(E_Cl_sn*1000))
    print('E_Cl_dn: ', round(E_Cl_dn*1000))
    print('E_Cl_dg: ', round(E_Cl_dg*1000))
    print('E_Ca_sn: ', round(E_Ca_sn*1000))
    print('E_Ca_dn: ', round(E_Ca_dn*1000))
    print("----------------------------")

    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)

    Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z  = sol.y
    t = sol.t

    my_cell = Buffy(T, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, \
        Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)
    
    phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = my_cell.membrane_potentials()
    
    E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = my_cell.reversal_potentials()

    q_si = my_cell.total_charge([my_cell.Na_si[-1], my_cell.K_si[-1], my_cell.Cl_si[-1], my_cell.Ca_si[-1]], my_cell.k_res_si, my_cell.V_si)
    q_se = my_cell.total_charge([my_cell.Na_se[-1], my_cell.K_se[-1], my_cell.Cl_se[-1], my_cell.Ca_se[-1]], my_cell.k_res_se, my_cell.V_se)        
    q_sg = my_cell.total_charge([my_cell.Na_sg[-1], my_cell.K_sg[-1], my_cell.Cl_sg[-1], 0], my_cell.k_res_sg, my_cell.V_sg)        
    q_di = my_cell.total_charge([my_cell.Na_di[-1], my_cell.K_di[-1], my_cell.Cl_di[-1], my_cell.Ca_di[-1]], my_cell.k_res_di, my_cell.V_di)
    q_de = my_cell.total_charge([my_cell.Na_de[-1], my_cell.K_de[-1], my_cell.Cl_de[-1], my_cell.Ca_de[-1]], my_cell.k_res_de, my_cell.V_de)
    q_dg = my_cell.total_charge([my_cell.Na_dg[-1], my_cell.K_dg[-1], my_cell.Cl_dg[-1], 0], my_cell.k_res_dg, my_cell.V_dg)
    print("Final values")
    print("----------------------------")
    print("total charge at the end (C): ", q_si + q_se + q_sg + q_di + q_de + q_dg)
    print("Q_si + Q_sg (C):", q_si+q_sg)
    print("Q_se (C): ", q_se)
    print("Q_di + Q_sg (C):", q_di+q_dg)
    print("Q_de (C): ", q_de)

    print("----------------------------")
    print('elapsed time: ', round(time.time() - start_time, 1), 'seconds')

    plt.figure(1)
    plt.plot(t, phi_msn*1000, '-', label='sn')
    plt.plot(t, phi_mdn*1000, '-', label='dn')
#    plt.plot(t, phi_msg*1000, '-', label='sg')
#    plt.plot(t, phi_mdg*1000, '-', label='dg')
    plt.title('Membrane potentials')
    plt.xlabel('time [s]')
    plt.legend(loc='upper right')
#
#    plt.plot(t, E_Na_s, label='E_Na')
#    plt.plot(t, E_K_s, label='E_K')
#    plt.plot(t, E_Cl_s, label='E_Cl')
#    plt.title('Reversal potentials soma')
#    plt.xlabel('time [s]')
#    plt.legend()
#    plt.show()
#
#    plt.plot(t, E_Na_d, label='E_Na')
#    plt.plot(t, E_K_d, label='E_K')
#    plt.plot(t, E_Cl_d, label='E_Cl')
#    plt.title('Reversal potentials dendrite')
#    plt.xlabel('time [s]')
#    plt.legend()
#    plt.show()

    plt.figure(2)
    plt.plot(t, Na_si, label='Na_si')
    plt.plot(t, Na_se, label='Na_se')
    plt.plot(t, Na_sg, label='Na_sg')
    plt.plot(t, Na_di, label='Na_di')
    plt.plot(t, Na_de, label='Na_de')
    plt.plot(t, Na_dg, label='Na_dg')
    plt.title('Sodium concentrations')
    plt.xlabel('time [s]')
    plt.legend(loc='upper right')

    plt.figure(3)
    plt.plot(t, K_si, label='K_si')
    plt.plot(t, K_se, label='K_se')
    plt.plot(t, K_sg, label='K_sg')
    plt.plot(t, K_di, label='K_di')
    plt.plot(t, K_de, label='K_de')
    plt.plot(t, K_dg, label='K_dg')
    plt.title('Potassium concentrations')
    plt.xlabel('time [s]')
    plt.legend(loc='upper right')

    plt.figure(4)
    plt.plot(t, Cl_si, label='Cl_si')
    plt.plot(t, Cl_se, label='Cl_se')
    plt.plot(t, Cl_sg, label='Cl_sg')
    plt.plot(t, Cl_di, label='Cl_di')
    plt.plot(t, Cl_de, label='Cl_de')
    plt.plot(t, Cl_dg, label='Cl_dg')
    plt.title('Chloride concentrations')
    plt.xlabel('time [s]')
    plt.legend(loc='upper right')

    plt.figure(5)
    plt.plot(t, Ca_si, label='Ca_si')
    plt.plot(t, Ca_se, label='Ca_se')
    plt.plot(t, Ca_di, label='Ca_di')
    plt.plot(t, Ca_de, label='Ca_de')
    plt.title('Calsium concentrations')
    plt.xlabel('time [s]')
    plt.legend(loc='upper right')
    plt.show()

