from .buffy import Buffy
import numpy as np

class Swella(Buffy):

    def __init__(self, T, nNa_si, nNa_se, nNa_sg, nNa_di, nNa_de, nNa_dg, nK_si, nK_se, nK_sg, nK_di, nK_de, nK_dg, \
        nCl_si, nCl_se, nCl_sg, nCl_di, nCl_de, nCl_dg, nCa_si, nCa_se, nCa_di, nCa_de, \
        nk_res_si, nk_res_se, nk_res_sg, nk_res_di, nk_res_de, nk_res_dg, alpha, \
        K0_se, K0_sg, K0_de, K0_dg, \
        Ca0_si, Ca0_di, n, h, s, c, q, z, \
        V_si, V_se, V_sg, V_di, V_de, V_dg):

        # temperature [K]
        self.T = T

        # ion concentraions [mol * m**-3]
        self.nNa_si = nNa_si
        self.nNa_se = nNa_se
        self.nNa_sg = nNa_sg
        self.nNa_di = nNa_di
        self.nNa_de = nNa_de
        self.nNa_dg = nNa_dg
        self.nK_si = nK_si
        self.nK_se = nK_se
        self.nK_sg = nK_sg
        self.nK_di = nK_di
        self.nK_de = nK_de
        self.nK_dg = nK_dg
        self.nCl_si = nCl_si
        self.nCl_se = nCl_se 
        self.nCl_sg = nCl_sg 
        self.nCl_di = nCl_di 
        self.nCl_de = nCl_de
        self.nCl_dg = nCl_dg
        self.nCa_si = nCa_si
        self.nCa_se = nCa_se 
        self.nCa_di = nCa_di 
        self.nCa_de = nCa_de
        self.nk_res_si = nk_res_si
        self.nk_res_se = nk_res_se
        self.nk_res_sg = nk_res_sg
        self.nk_res_di = nk_res_di
        self.nk_res_de = nk_res_de
        self.nk_res_dg = nk_res_dg
        self.Na_si = nNa_si/V_si
        self.Na_se = nNa_se/V_se
        self.Na_sg = nNa_sg/V_sg
        self.Na_di = nNa_di/V_di
        self.Na_de = nNa_de/V_de
        self.Na_dg = nNa_dg/V_dg
        self.K_si = nK_si/V_si
        self.K_se = nK_se/V_se
        self.K_sg = nK_sg/V_sg
        self.K_di = nK_di/V_di
        self.K_de = nK_de/V_de
        self.K_dg = nK_dg/V_dg
        self.Cl_si = nCl_si/V_si
        self.Cl_se = nCl_se/V_se 
        self.Cl_sg = nCl_sg/V_sg 
        self.Cl_di = nCl_di/V_di 
        self.Cl_de = nCl_de/V_de
        self.Cl_dg = nCl_dg/V_dg
        self.Ca_si = nCa_si/V_si
        self.Ca_se = nCa_se/V_se 
        self.Ca_di = nCa_di/V_di 
        self.Ca_de = nCa_de/V_de
        self.free_Ca_si = 0.01*self.Ca_si
        self.free_Ca_di = 0.01*self.Ca_di
        self.K0_se = K0_se           
        self.K0_sg = K0_sg          
        self.K0_de = K0_de     
        self.K0_dg = K0_dg
        self.Ca0_si = Ca0_si
        self.Ca0_di = Ca0_di
        self.k_res_si = nk_res_si/V_si
        self.k_res_se = nk_res_se/V_se
        self.k_res_sg = nk_res_sg/V_sg
        self.k_res_di = nk_res_di/V_di
        self.k_res_de = nk_res_de/V_de
        self.k_res_dg = nk_res_dg/V_dg

        self.KNaI = 10
        self.KKE = 1.5

        # gating variables
        self.n = n
        self.h = h
        self.s = s
        self.c = c
        self.q = q
        self.z = z

        # membrane capacitance [F * m**-2]
        self.C_msn = 3e-2 # Pinsky and Rinzel, 1994
        self.C_mdn = 3e-2 # Pinsky and Rinzel, 1994
        self.C_msg = 3e-2
        self.C_mdg = 3e-2
       
        # volumes and areas
        self.alpha = alpha
        self.A_sn = 616e-12               # [m**2]
        self.A_dn = 616e-12               # [m**2]
        self.A_in = self.alpha*self.A_sn  # [m**2]
        self.A_e = self.A_in/2.           # [m**2]
        self.A_sg = 616e-12               # [m**2]
        self.A_dg = 616e-12               # [m**2]
        self.A_ig = self.alpha*self.A_sg  # [m**2]
        self.dx = 667e-6                  # [m]
        self.V_si = V_si
        self.V_se = V_se
        self.V_sg = V_sg
        self.V_di = V_di
        self.V_de = V_de
        self.V_dg = V_dg
        
        self.G_n = 7e-13 #7e-13 # [m**3/Pa/s]
        self.G_g = 7e-13 #7e-13 # [m**3/Pa/s]

        self.tau = 75.
        
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
        self.g_Na_astro = 1.     # Halnes et al
        self.g_K_astro = 16.96   # Halnes et al
        self.g_Cl_astro = 0.5    # Halnes et al
        
        # pump strengths
        self.rho = 1.87e-6
        self.U_kcc2 = 7.00e-7
        self.U_nkcc1 = 2.33e-7
        self.rho_astro = 1.12e-6

        self.E0_K_sg = self.nernst_potential(self.Z_K, self.K0_sg, self.K0_se)
        self.E0_K_dg = self.nernst_potential(self.Z_K, self.K0_dg, self.K0_de)

#        res = 0#-68e-3*3e-2*616e-12/9.648e4
#        k_res_i0 = -18-99+7-2*0.01 + res
#        k_res_e0 = -140-4.3+134-2*1.1 - 2*res
#        k_res_g0 = -18-99+7 + res
#        ni = 18+99+7+0.01+k_res_i0
#        ne = 140+4.3+134+1.1+k_res_e0
#        ng = 18+99+7+k_res_g0

        # solute potentials OBS lagt til noen konstanter her
        self.psi_si = self.R * self.T * (self.Na_si + self.K_si + self.Cl_si + self.Ca_si -  122.01)
        self.psi_se = self.R * self.T * (self.Na_se + self.K_se + self.Cl_se + self.Ca_se - 287.55)
        self.psi_sg = self.R * self.T * (self.Na_sg + self.K_sg + self.Cl_sg - 121.6)
        self.psi_di = self.R * self.T * (self.Na_di + self.K_di + self.Cl_di + self.Ca_di - 122.01)
        self.psi_de = self.R * self.T * (self.Na_de + self.K_de + self.Cl_de + self.Ca_de - 287.55)
        self.psi_dg = self.R * self.T * (self.Na_dg + self.K_dg + self.Cl_dg - 121.6)

    def alpha_m(self, phi_sm):
        phi_1 = phi_sm*1e3 + 46.9
        alpha = - 0.32 * phi_1 / (np.exp(-phi_1 / 4) - 1.)
        alpha = alpha*1e3
        return alpha

    def beta_m(self, phi_sm):
        phi_2 = phi_sm*1e3 + 19.9
        beta = 0.28 * phi_2 / (np.exp(phi_2 / 5.) - 1.)
        beta = beta*1e3
        return beta

    def alpha_h(self, phi_sm):
        alpha = 0.128 * np.exp((-43. - phi_sm*1e3) / 18.)
        alpha = alpha*1e3
        return alpha

    def beta_h(self, phi_sm):
        phi_5 = phi_sm*1e3 + 20.
        beta = 4. / (1 + np.exp(-phi_5 / 5.))
        beta = beta*1e3
        return beta

    def alpha_n(self, phi_sm):
        phi_3 = phi_sm*1e3 + 24.9
        alpha = - 0.016 * phi_3 / (np.exp(-phi_3 / 5.) - 1)
        alpha = alpha*1e3
        return alpha

    def beta_n(self, phi_sm):
        phi_4 = phi_sm*1e3 + 40.
        beta = 0.25 * np.exp(-phi_4 / 40.)
        beta = beta*1e3
        return beta

    def alpha_s(self, phi_dm):
        alpha = 1.6 / (1 + np.exp(-0.072 * (phi_dm*1000 - 5.)))
        alpha = alpha*1000
        return alpha

    def beta_s(self, phi_dm):
        phi_6 = phi_dm*1000 + 8.9
        beta = 0.02 * phi_6 / (np.exp(phi_6 / 5.) - 1.)
        beta = beta*1000
        return beta

    def alpha_c(self, phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        phi_8 = phi_dm*1e3 + 50.0
        if phi_dm*1e3 <= -10:
            alpha = 0.0527 * np.exp(phi_8/11.- phi_7/27.)
        else:
            alpha = 2 * np.exp(-phi_7 / 27.)
        alpha = alpha*1e3
        return alpha

    def beta_c(self, phi_dm):
        phi_7 = phi_dm*1e3 + 53.5
        if phi_dm*1e3 <= -10:
            beta = 2. * np.exp(-phi_7 / 27.) - self.alpha_c(phi_dm)/1e3
        else:
            beta = 0.
        beta = beta*1e3
        return beta

    def chi(self):
        return min((self.free_Ca_di-99.8e-6)/2.5e-4, 1.0)

    def alpha_q(self):
        return min(2e4*(self.free_Ca_di-99.8e-6), 10.0) 

    def beta_q(self):
        return 1.0

    def m_inf(self, phi_sm):
        return self.alpha_m(phi_sm) / (self.alpha_m(phi_sm) + self.beta_m(phi_sm))

    def z_inf(self, phi_dm):
        phi_half = -30
        k = 1
        return 1/(1 + np.exp((phi_dm*1000 - phi_half)/k))

    def j_pump(self, Na_i, K_e):
        j = (self.rho / (1.0 + np.exp((25. - Na_i)/3.))) * (1.0 / (1.0 + np.exp(3.5 - K_e)))
        return j

    def j_pump_astro(self, Na_i, K_e):
        j = self.rho_astro * (Na_i**1.5 / (Na_i**1.5 + self.KNaI**1.5)) * (K_e / (K_e + self.KKE))
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
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + self.g_Na * self.m_inf(phi_sm)**2 * self.h * (phi_sm - E_Na_s) / (self.F*self.Z_Na) \
            - 2*self.tau*(self.Ca_si - self.Ca0_si)*self.V_si/self.A_sn
        return j 

    def j_K_sn(self, phi_sm, E_K_s):
        j = self.g_K_leak*(phi_sm - E_K_s) / (self.F*self.Z_K) \
            - 2*self.j_pump(self.Na_si, self.K_se) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + self.g_DR * self.n * (phi_sm - E_K_s) / (self.F*self.Z_K)
        return j

    def j_Cl_sn(self, phi_sm, E_Cl_s):
        j = self.g_Cl_leak*(phi_sm - E_Cl_s) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.K_si, self.K_se, self.Cl_si, self.Cl_se) \
            + 2*self.j_nkcc1(self.Na_si, self.Na_se, self.K_si, self.K_se, self.Cl_si, self.Cl_se)
        return j

    def j_Ca_sn(self):
        j =  self.tau * (self.Ca_si - self.Ca0_si)*self.V_si/self.A_sn
        return j

    def j_Na_dn(self, phi_dm, E_Na_d):
        j = self.g_Na_leak*(phi_dm - E_Na_d) / (self.F*self.Z_Na) \
            + 3*self.j_pump(self.Na_di, self.K_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            - 2*self.tau*(self.Ca_di - self.Ca0_di)*self.V_di/self.A_dn
        return j

    def j_K_dn(self, phi_dm, E_K_d):
        j = self.g_K_leak*(phi_dm - E_K_d) / (self.F*self.Z_K) \
            - 2*self.j_pump(self.Na_di, self.K_de) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + self.g_AHP * self.q * (phi_dm - E_K_d) / (self.F*self.Z_K) \
            + self.g_C * self.c * self.chi() * (phi_dm - E_K_d) / (self.F*self.Z_K)
        return j

    def j_Cl_dn(self, phi_dm, E_Cl_d):
        j = self.g_Cl_leak*(phi_dm - E_Cl_d) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.K_di, self.K_de, self.Cl_di, self.Cl_de) \
            + 2*self.j_nkcc1(self.Na_di, self.Na_de, self.K_di, self.K_de, self.Cl_di, self.Cl_de)
        return j

    def j_Ca_dn(self, phi_dm, E_Ca_d):
        j = self.g_Ca * self.s**2 * self.z * (phi_dm - E_Ca_d) / (self.F*self.Z_Ca) \
            + self.tau*(self.Ca_di - self.Ca0_di)*self.V_di/self.A_dn
        return j

    def j_Na_sg(self, phi_sm, E_Na_g):
        j = self.g_Na_astro * (phi_sm - E_Na_g) / self.F \
        + 3*self.j_pump_astro(self.Na_sg, self.K_se)
        return j

    def j_K_sg(self, phi_sm, E_K_g):
        dphi = (phi_sm - E_K_g)*1000
#        if self.K_se/self.K0_se <= 0:
#            print(self.K_se, self.K0_se)
        f = np.sqrt(self.K_se/self.K0_se) * ((1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))) * ((1 + np.exp(-(118.6+self.E0_K_sg)/44.1))/(1+np.exp(-(118.6+phi_sm)/44.1)))
        j = self.g_K_astro * f * (phi_sm - E_K_g) / self.F \
        - 2 * self.j_pump_astro(self.Na_sg, self.K_se)
        return j

    def j_Cl_sg(self, phi_sm, E_Cl_g):
        j = - self.g_Cl_astro * (phi_sm - E_Cl_g) / self.F
        return j

    def j_Na_dg(self, phi_dm, E_Na_g):
        j = self.g_Na_astro * (phi_dm - E_Na_g) / self.F \
        + 3*self.j_pump_astro(self.Na_dg, self.K_de)
        return j

    def j_K_dg(self, phi_dm, E_K_g):
        dphi = (phi_dm - E_K_g)*1000
#        if self.K_de/self.K0_de <= 0:
#            print(self.K_de, self.K0_de)
        f = np.sqrt(self.K_de/self.K0_de) * ((1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))) * ((1 + np.exp(-(118.6+self.E0_K_dg)/44.1))/(1+np.exp(-(118.6+phi_dm)/44.1)))
        j = self.g_K_astro * f * (phi_dm - E_K_g) / self.F \
        - 2 * self.j_pump_astro(self.Na_dg, self.K_de)
        return j

    def j_Cl_dg(self, phi_dm, E_Cl_g):
        j = - self.g_Cl_astro * (phi_dm - E_Cl_g) / self.F
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
#        if k_e/k_i <= 0:
#            print(k_e, k_i)
        E = self.R*self.T / (Z*self.F) * np.log(k_e / k_i)
        return E

    def reversal_potentials(self):
        E_Na_sn = self.nernst_potential(self.Z_Na, self.Na_si, self.Na_se)
        E_Na_sg = self.nernst_potential(self.Z_Na, self.Na_sg, self.Na_se)
        E_Na_dn = self.nernst_potential(self.Z_Na, self.Na_di, self.Na_de)
        E_Na_dg = self.nernst_potential(self.Z_Na, self.Na_dg, self.Na_de)
        E_K_sn = self.nernst_potential(self.Z_K, self.K_si, self.K_se)
        E_K_sg = self.nernst_potential(self.Z_K, self.K_sg, self.K_se)
        E_K_dn = self.nernst_potential(self.Z_K, self.K_di, self.K_de)
        E_K_dg = self.nernst_potential(self.Z_K, self.K_dg, self.K_de)
        E_Cl_sn = self.nernst_potential(self.Z_Cl, self.Cl_si, self.Cl_se)
        E_Cl_sg = self.nernst_potential(self.Z_Cl, self.Cl_sg, self.Cl_se)
        E_Cl_dn = self.nernst_potential(self.Z_Cl, self.Cl_di, self.Cl_de)
        E_Cl_dg = self.nernst_potential(self.Z_Cl, self.Cl_dg, self.Cl_de)
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

        phi_di = q_di / (self.C_mdn * self.A_dn)
        phi_de = 0.
        phi_dg = q_dg / (self.C_mdg * self.A_dg)
        phi_se = ( - self.dx * self.A_in * I_n_diff + self.A_in * sigma_i * phi_di - self.A_in * sigma_i * q_si / (self.C_msn * self.A_sn) \
            - self.dx * self.A_ig * I_g_diff + self.A_ig * sigma_g * phi_dg - self.A_ig * sigma_g * q_sg / (self.C_msg * self.A_sg) - self.dx * self.A_e * I_e_diff ) \
            / ( self.A_e * sigma_e + self.A_in * sigma_i + self.A_ig * sigma_g )
        phi_si = q_si / (self.C_msn * self.A_sn) + phi_se
        phi_sg = q_sg / (self.C_msg * self.A_sg) + phi_se
        phi_msn = phi_si - phi_se
        phi_msg = phi_sg - phi_se
        phi_mdn = phi_di - phi_de
        phi_mdg = phi_dg - phi_de

        return phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg

    def dkdt(self):
       
        phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
        E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = self.reversal_potentials()

        V_fr_s = self.V_si/self.V_se
        V_fr_d = self.V_di/self.V_de 

        j_Na_msn = self.j_Na_sn(phi_msn, E_Na_sn)
        j_K_msn = self.j_K_sn(phi_msn, E_K_sn)
        j_Cl_msn = self.j_Cl_sn(phi_msn, E_Cl_sn)

        j_Na_msg = self.j_Na_sg(phi_msg, E_Na_sg)
        j_K_msg = self.j_K_sg(phi_msg, E_K_sg)
        j_Cl_msg = self.j_Cl_sg(phi_msg, E_Cl_sg)

        j_Na_mdn = self.j_Na_dn(phi_mdn, E_Na_dn)
        j_K_mdn = self.j_K_dn(phi_mdn, E_K_dn)    
        j_Cl_mdn = self.j_Cl_dn(phi_mdn, E_Cl_dn)

        j_Na_mdg = self.j_Na_dg(phi_mdg, E_Na_dg)
        j_K_mdg = self.j_K_dg(phi_mdg, E_K_dg)
        j_Cl_mdg = self.j_Cl_dg(phi_mdg, E_Cl_dg)

        j_Ca_mdn = self.j_Ca_dn(phi_mdn, E_Ca_dn)

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

        dnNadt_si = -j_Na_msn*(self.A_sn) - j_Na_in*(self.A_in) 
        dnNadt_se = j_Na_msn*(self.A_sn) + j_Na_msg*(self.A_sg) - j_Na_e*self.A_e 
        dnNadt_sg = -j_Na_msg*(self.A_sg) - j_Na_ig*(self.A_ig)
        dnNadt_di = -j_Na_mdn*(self.A_dn) + j_Na_in*(self.A_in) 
        dnNadt_de = j_Na_mdn*(self.A_dn) + j_Na_mdg*(self.A_dg) + j_Na_e*self.A_e 
        dnNadt_dg = -j_Na_mdg*(self.A_dg) + j_Na_ig*(self.A_ig)

        dnKdt_si = -j_K_msn*(self.A_sn) - j_K_in*(self.A_in)
        dnKdt_se = j_K_msn*(self.A_sn) + j_K_msg*(self.A_sg) - j_K_e*(self.A_e)
        dnKdt_sg = -j_K_msg*(self.A_sg) - j_K_ig*(self.A_ig)
        dnKdt_di = -j_K_mdn*(self.A_dn) + j_K_in*(self.A_in)
        dnKdt_de = j_K_mdn*(self.A_dn) + j_K_mdg*(self.A_dg) + j_K_e*(self.A_e)
        dnKdt_dg = -j_K_mdg*(self.A_dg) + j_K_ig*(self.A_ig)

        dnCldt_si = -j_Cl_msn*(self.A_sn) - j_Cl_in*(self.A_in)
        dnCldt_se = j_Cl_msn*(self.A_sn) + j_Cl_msg*(self.A_sg) - j_Cl_e*(self.A_e)
        dnCldt_sg = -j_Cl_msg*(self.A_sg) - j_Cl_ig*(self.A_ig)
        dnCldt_di = -j_Cl_mdn*(self.A_dn) + j_Cl_in*(self.A_in)
        dnCldt_de = j_Cl_mdn*(self.A_dn) + j_Cl_mdg*(self.A_dg) + j_Cl_e*(self.A_e)
        dnCldt_dg = -j_Cl_mdg*(self.A_dg) + j_Cl_ig*(self.A_ig)

        dnCadt_si = - j_Ca_in*(self.A_in) - self.j_Ca_sn()*self.A_sn
        dnCadt_se = - j_Ca_e*(self.A_e) + self.j_Ca_sn()*self.A_sn
        dnCadt_di = j_Ca_in*(self.A_in) - j_Ca_mdn*(self.A_dn) 
        dnCadt_de = j_Ca_e*(self.A_e) + j_Ca_mdn*(self.A_dn) 

#        dNadt_si = -j_Na_msn*(self.A_sn / self.V_si) - j_Na_in*(self.A_in / self.V_si) + 2*75.*(self.Ca_si - self.Ca0_si)
#        dNadt_se = j_Na_msn*(self.A_sn / self.V_se) + j_Na_msg*(self.A_sg / self.V_se) - j_Na_e*(self.A_e / self.V_se) - 2*75.*V_fr_s*(self.Ca_si - self.Ca0_si)
#        dNadt_sg = -j_Na_msg*(self.A_sg / self.V_sg) - j_Na_ig*(self.A_ig / self.V_sg)
#        dNadt_di = -j_Na_mdn*(self.A_dn / self.V_di) + j_Na_in*(self.A_in / self.V_di) + 2*75.*(self.Ca_di - self.Ca0_di)
#        dNadt_de = j_Na_mdn*(self.A_dn / self.V_de) + j_Na_mdg*(self.A_dg / self.V_de) + j_Na_e*(self.A_e / self.V_de) - 2*75.*V_fr_d*(self.Ca_di - self.Ca0_di)
#        dNadt_dg = -j_Na_mdg*(self.A_dg / self.V_dg) + j_Na_ig*(self.A_ig / self.V_dg)
#
#        dKdt_si = -j_K_msn*(self.A_sn / self.V_si) - j_K_in*(self.A_in / self.V_si)
#        dKdt_se = j_K_msn*(self.A_sn / self.V_se) + j_K_msg*(self.A_sg / self.V_se) - j_K_e*(self.A_e / self.V_se)
#        dKdt_sg = -j_K_msg*(self.A_sg / self.V_sg) - j_K_ig*(self.A_ig / self.V_sg)
#        dKdt_di = -j_K_mdn*(self.A_dn / self.V_di) + j_K_in*(self.A_in / self.V_di)
#        dKdt_de = j_K_mdn*(self.A_dn / self.V_de) + j_K_mdg*(self.A_dg / self.V_de) + j_K_e*(self.A_e / self.V_de)
#        dKdt_dg = -j_K_mdg*(self.A_dg / self.V_dg) + j_K_ig*(self.A_ig / self.V_dg)
#
#        dCldt_si = -j_Cl_msn*(self.A_sn / self.V_si) - j_Cl_in*(self.A_in / self.V_si)
#        dCldt_se = j_Cl_msn*(self.A_sn / self.V_se) + j_Cl_msg*(self.A_sg / self.V_se) - j_Cl_e*(self.A_e / self.V_se)
#        dCldt_sg = -j_Cl_msg*(self.A_sg / self.V_sg) - j_Cl_ig*(self.A_ig / self.V_sg)
#        dCldt_di = -j_Cl_mdn*(self.A_dn / self.V_di) + j_Cl_in*(self.A_in / self.V_di)
#        dCldt_de = j_Cl_mdn*(self.A_dn / self.V_de) + j_Cl_mdg*(self.A_dg / self.V_de) + j_Cl_e*(self.A_e / self.V_de)
#        dCldt_dg = -j_Cl_mdg*(self.A_dg / self.V_dg) + j_Cl_ig*(self.A_ig / self.V_dg)
#
#        dCadt_si = - j_Ca_in*(self.A_in / self.V_si) - 75.*(self.Ca_si - self.Ca0_si)
#        dCadt_se = - j_Ca_e*(self.A_e / self.V_se) + V_fr_s*75.*(self.Ca_si - self.Ca0_si)
#        dCadt_di = j_Ca_in*(self.A_in / self.V_di) - j_Ca_mdn*(self.A_dn / self.V_di) - 75.*(self.Ca_di - self.Ca0_di)
#        dCadt_de = j_Ca_e*(self.A_e / self.V_de) + j_Ca_mdn*(self.A_dn / self.V_de) + V_fr_d*75.*(self.Ca_di - self.Ca0_di)

        return dnNadt_si, dnNadt_se, dnNadt_sg, dnNadt_di, dnNadt_de, dnNadt_dg, dnKdt_si, dnKdt_se, dnKdt_sg, dnKdt_di, dnKdt_de, dnKdt_dg, \
            dnCldt_si, dnCldt_se, dnCldt_sg, dnCldt_di, dnCldt_de, dnCldt_dg, dnCadt_si, dnCadt_se, dnCadt_di, dnCadt_de
        #return dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, \
        #    dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de

    def dmdt(self):
        phi_si, phi_se, phi_sg, phi_di, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
        
        dndt = self.alpha_n(phi_msn)*(1.0-self.n) - self.beta_n(phi_msn)*self.n
        dhdt = self.alpha_h(phi_msn)*(1.0-self.h) - self.beta_h(phi_msn)*self.h 
        dsdt = self.alpha_s(phi_mdn)*(1.0-self.s) - self.beta_s(phi_mdn)*self.s
        dcdt = self.alpha_c(phi_mdn)*(1.0-self.c) - self.beta_c(phi_mdn)*self.c
        dqdt = self.alpha_q()*(1.0-self.q) - self.beta_q()*self.q
        dzdt = (self.z_inf(phi_mdn) - self.z)
        
        return dndt, dhdt, dsdt, dcdt, dqdt, dzdt

    def dVdt(self):

        dVsidt = -self.G_n * (self.psi_se - self.psi_si)
        dVsgdt = -self.G_g * (self.psi_se - self.psi_sg)
        dVdidt = -self.G_n * (self.psi_de - self.psi_di)
        dVdgdt = -self.G_g * (self.psi_de - self.psi_dg)
        dVsedt = - (dVsidt + dVsgdt)
        dVdedt = - (dVdidt + dVdgdt)

        return dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt
