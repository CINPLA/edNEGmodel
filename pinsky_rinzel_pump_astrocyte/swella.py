import numpy as np

class Swella():

    def __init__(self, T, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, \
        Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, \
        X_si, X_se, X_sg, X_di, X_de, X_dg, alpha, \
        cbK_se, cbK_sg, cbK_de, cbK_dg, \
        cbCa_si, cbCa_di, n, h, s, c, q, z, \
        V_si, V_se, V_sg, V_di, V_de, V_dg, \
        c_res_si, c_res_se, c_res_sg, c_res_di, c_res_de, c_res_dg):

        # temperature [K]
        self.T = T

        # ions [mol]
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
        self.X_si = X_si
        self.X_se = X_se
        self.X_sg = X_sg
        self.X_di = X_di
        self.X_de = X_de
        self.X_dg = X_dg
        
        # ion concentraions [mol * m**-3]
        self.cNa_si = Na_si/V_si
        self.cNa_se = Na_se/V_se
        self.cNa_sg = Na_sg/V_sg
        self.cNa_di = Na_di/V_di
        self.cNa_de = Na_de/V_de
        self.cNa_dg = Na_dg/V_dg
        self.cK_si = K_si/V_si
        self.cK_se = K_se/V_se
        self.cK_sg = K_sg/V_sg
        self.cK_di = K_di/V_di
        self.cK_de = K_de/V_de
        self.cK_dg = K_dg/V_dg
        self.cCl_si = Cl_si/V_si
        self.cCl_se = Cl_se/V_se 
        self.cCl_sg = Cl_sg/V_sg 
        self.cCl_di = Cl_di/V_di 
        self.cCl_de = Cl_de/V_de
        self.cCl_dg = Cl_dg/V_dg
        self.cCa_si = Ca_si/V_si
        self.cCa_se = Ca_se/V_se 
        self.cCa_di = Ca_di/V_di 
        self.cCa_de = Ca_de/V_de
        self.free_cCa_si = 0.01*self.cCa_si
        self.free_cCa_di = 0.01*self.cCa_di
        self.cX_si = X_si/V_si
        self.cX_se = X_se/V_se
        self.cX_sg = X_sg/V_sg
        self.cX_di = X_di/V_di
        self.cX_de = X_de/V_de
        self.cX_dg = X_dg/V_dg

        # concentrations of static molecules without charge [mol * m**-3] 
        self.c_res_si = c_res_si
        self.c_res_se = c_res_se
        self.c_res_sg = c_res_sg
        self.c_res_di = c_res_di
        self.c_res_de = c_res_de
        self.c_res_dg = c_res_dg

        # 
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
        self.V_si = V_si                  # [m**3]
        self.V_se = V_se                  # [m**3]
        self.V_sg = V_sg                  # [m**3]
        self.V_di = V_di                  # [m**3]
        self.V_de = V_de                  # [m**3]
        self.V_dg = V_dg                  # [m**3]
 
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
        self.Z_X = -1.

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
        self.tau = 75.
        self.rho_astro = 1.12e-6
        self.U_nkcc1_astro = 2.33e-7
        
        # water permeabilities [m**3/Pa/s] 
        self.G_n = 2e-23    # Dijkstra et al. 2016
        self.G_g = 5e-23    # Østby et al. 2009

        # initial values 
        self.cbK_se = cbK_se           
        self.cbK_sg = cbK_sg          
        self.cbK_de = cbK_de     
        self.cbK_dg = cbK_dg
        self.cbCa_si = cbCa_si
        self.cbCa_di = cbCa_di
        self.E0_K_sg = self.nernst_potential(self.Z_K, self.cbK_sg, self.cbK_se)
        self.E0_K_dg = self.nernst_potential(self.Z_K, self.cbK_dg, self.cbK_de)

        # solute potentials OBS lagt til noen konstanter her maa gjores til input
        self.psi_si = self.R * self.T * (self.cNa_si + self.cK_si + self.cCl_si + self.cCa_si -  c_res_si)
        self.psi_se = self.R * self.T * (self.cNa_se + self.cK_se + self.cCl_se + self.cCa_se - c_res_se)
        self.psi_sg = self.R * self.T * (self.cNa_sg + self.cK_sg + self.cCl_sg - c_res_sg)
        self.psi_di = self.R * self.T * (self.cNa_di + self.cK_di + self.cCl_di + self.cCa_di - c_res_di)
        self.psi_de = self.R * self.T * (self.cNa_de + self.cK_de + self.cCl_de + self.cCa_de - c_res_de)
        self.psi_dg = self.R * self.T * (self.cNa_dg + self.cK_dg + self.cCl_dg - c_res_dg)

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
        return min((self.free_cCa_di-99.8e-6)/2.5e-4, 1.0)

    def alpha_q(self):
        return min(2e4*(self.free_cCa_di-99.8e-6), 10.0) 

    def beta_q(self):
        return 1.0

    def m_inf(self, phi_sm):
        return self.alpha_m(phi_sm) / (self.alpha_m(phi_sm) + self.beta_m(phi_sm))

    def z_inf(self, phi_dm):
        phi_half = -30
        k = 1
        return 1/(1 + np.exp((phi_dm*1000 - phi_half)/k))

    def j_pump(self, cNa_i, cK_e):
        j = (self.rho / (1.0 + np.exp((25. - cNa_i)/3.))) * (1.0 / (1.0 + np.exp(3.5 - cK_e)))
        return j

    def j_pump_astro(self, cNa_i, cK_e):
        j = self.rho_astro * (cNa_i**1.5 / (cNa_i**1.5 + self.KNaI**1.5)) * (cK_e / (cK_e + self.KKE))
        return j

    def j_kcc2(self, cK_i, cK_e, cCl_i, cCl_e):
        j = self.U_kcc2 * np.log(cK_i*cCl_i/(cK_e*cCl_e))
        return j
    
    def j_nkcc1(self, cNa_i, cNa_e, cK_i, cK_e, cCl_i, cCl_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp(16 - cK_e))) * (np.log(cK_i*cCl_i/(cK_e*cCl_e)) + np.log(cNa_i*cCl_i/(cNa_e*cCl_e)))
        return j

    def j_nkcc1_astro(self, cNa_i, cNa_e, cK_i, cK_e, cCl_i, cCl_e):
        j = self.U_nkcc1_astro * (1 / (1 + np.exp(16 - cK_e))) * (np.log(cK_i*cCl_i/(cK_e*cCl_e)) + np.log(cNa_i*cCl_i/(cNa_e*cCl_e)))
        return j

    def j_Na_sn(self, phi_sm, E_Na_s):
        j = self.g_Na_leak*(phi_sm - E_Na_s) / (self.F*self.Z_Na) \
            + 3*self.j_pump(self.cNa_si, self.cK_se) \
            + self.j_nkcc1(self.cNa_si, self.cNa_se, self.cK_si, self.cK_se, self.cCl_si, self.cCl_se) \
            + self.g_Na * self.m_inf(phi_sm)**2 * self.h * (phi_sm - E_Na_s) / (self.F*self.Z_Na) \
            - 2*self.tau*(self.cCa_si - self.cbCa_si)*self.V_si/self.A_sn
        return j 

    def j_K_sn(self, phi_sm, E_K_s):
        j = self.g_K_leak*(phi_sm - E_K_s) / (self.F*self.Z_K) \
            - 2*self.j_pump(self.cNa_si, self.cK_se) \
            + self.j_kcc2(self.cK_si, self.cK_se, self.cCl_si, self.cCl_se) \
            + self.j_nkcc1(self.cNa_si, self.cNa_se, self.cK_si, self.cK_se, self.cCl_si, self.cCl_se) \
            + self.g_DR * self.n * (phi_sm - E_K_s) / (self.F*self.Z_K)
        return j

    def j_Cl_sn(self, phi_sm, E_Cl_s):
        j = self.g_Cl_leak*(phi_sm - E_Cl_s) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.cK_si, self.cK_se, self.cCl_si, self.cCl_se) \
            + 2*self.j_nkcc1(self.cNa_si, self.cNa_se, self.cK_si, self.cK_se, self.cCl_si, self.cCl_se)
        return j

    def j_Ca_sn(self):
        j =  self.tau * (self.cCa_si - self.cbCa_si)*self.V_si/self.A_sn
        return j

    def j_Na_dn(self, phi_dm, E_Na_d):
        j = self.g_Na_leak*(phi_dm - E_Na_d) / (self.F*self.Z_Na) \
            + 3*self.j_pump(self.cNa_di, self.cK_de) \
            + self.j_nkcc1(self.cNa_di, self.cNa_de, self.cK_di, self.cK_de, self.cCl_di, self.cCl_de) \
            - 2*self.tau*(self.cCa_di - self.cbCa_di)*self.V_di/self.A_dn
        return j

    def j_K_dn(self, phi_dm, E_K_d):
        j = self.g_K_leak*(phi_dm - E_K_d) / (self.F*self.Z_K) \
            - 2*self.j_pump(self.cNa_di, self.cK_de) \
            + self.j_kcc2(self.cK_di, self.cK_de, self.cCl_di, self.cCl_de) \
            + self.j_nkcc1(self.cNa_di, self.cNa_de, self.cK_di, self.cK_de, self.cCl_di, self.cCl_de) \
            + self.g_AHP * self.q * (phi_dm - E_K_d) / (self.F*self.Z_K) \
            + self.g_C * self.c * self.chi() * (phi_dm - E_K_d) / (self.F*self.Z_K)
        return j

    def j_Cl_dn(self, phi_dm, E_Cl_d):
        j = self.g_Cl_leak*(phi_dm - E_Cl_d) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.cK_di, self.cK_de, self.cCl_di, self.cCl_de) \
            + 2*self.j_nkcc1(self.cNa_di, self.cNa_de, self.cK_di, self.cK_de, self.cCl_di, self.cCl_de)
        return j

    def j_Ca_dn(self, phi_dm, E_Ca_d):
        j = self.g_Ca * self.s**2 * self.z * (phi_dm - E_Ca_d) / (self.F*self.Z_Ca) \
            + self.tau*(self.cCa_di - self.cbCa_di)*self.V_di/self.A_dn
        return j

    def j_Na_sg(self, phi_sm, E_Na_g):
        j = self.g_Na_astro * (phi_sm - E_Na_g) / self.F \
            + 3*self.j_pump_astro(self.cNa_sg, self.cK_se) \
            + self.j_nkcc1_astro(self.cNa_sg, self.cNa_se, self.cK_sg, self.cK_se, self.cCl_sg, self.cCl_se)
        return j

    def j_K_sg(self, phi_sm, E_K_g):
        dphi = (phi_sm - E_K_g)*1000
        f = np.sqrt(self.cK_se/self.cbK_se) * ((1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))) * ((1 + np.exp(-(118.6+self.E0_K_sg*1000)/44.1))/(1+np.exp(-(118.6+phi_sm*1000)/44.1)))
        j = self.g_K_astro * f * (phi_sm - E_K_g) / self.F \
            - 2 * self.j_pump_astro(self.cNa_sg, self.cK_se) \
            + self.j_nkcc1_astro(self.cNa_sg, self.cNa_se, self.cK_sg, self.cK_se, self.cCl_sg, self.cCl_se)
        return j

    def j_Cl_sg(self, phi_sm, E_Cl_g):
        j = - self.g_Cl_astro * (phi_sm - E_Cl_g) / self.F \
            + 2*self.j_nkcc1_astro(self.cNa_sg, self.cNa_se, self.cK_sg, self.cK_se, self.cCl_sg, self.cCl_se)
        return j

    def j_Na_dg(self, phi_dm, E_Na_g):
        j = self.g_Na_astro * (phi_dm - E_Na_g) / self.F \
            + 3*self.j_pump_astro(self.cNa_dg, self.cK_de) \
            + self.j_nkcc1_astro(self.cNa_dg, self.cNa_de, self.cK_dg, self.cK_de, self.cCl_dg, self.cCl_de)
        return j

    def j_K_dg(self, phi_dm, E_K_g):
        dphi = (phi_dm - E_K_g)*1000
        f = np.sqrt(self.cK_de/self.cbK_de) * ((1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))) * ((1 + np.exp(-(118.6+self.E0_K_dg*1000)/44.1))/(1+np.exp(-(118.6+phi_dm*1000)/44.1)))
        j = self.g_K_astro * f * (phi_dm - E_K_g) / self.F \
            - 2 * self.j_pump_astro(self.cNa_dg, self.cK_de) \
            + self.j_nkcc1_astro(self.cNa_dg, self.cNa_de, self.cK_dg, self.cK_de, self.cCl_dg, self.cCl_de)
        return j

    def j_Cl_dg(self, phi_dm, E_Cl_g):
        j = - self.g_Cl_astro * (phi_dm - E_Cl_g) / self.F \
            + 2*self.j_nkcc1_astro(self.cNa_dg, self.cNa_de, self.cK_dg, self.cK_de, self.cCl_dg, self.cCl_de)
        return j

    def j_k_diff(self, D_k, tortuosity, ck_s, ck_d):
        j = - D_k * (ck_d - ck_s) / (tortuosity**2 * self.dx)
        return j

    def j_k_drift(self, D_k, Z_k, tortuosity, ck_s, ck_d, phi_s, phi_d):
        j = - D_k * self.F * Z_k * (ck_d + ck_s) * (phi_d - phi_s) / (2 * tortuosity**2 * self.R * self.T * self.dx)
        return j

    def conductivity_k(self, D_k, Z_k, tortuosity, ck_s, ck_d): 
        sigma = self.F**2 * D_k * Z_k**2 * (ck_d + ck_s) / (2 * self.R * self.T * tortuosity**2)
        return sigma

    def total_charge(self, k):
#        Z_k = np.array([self.Z_Na, self.Z_K, self.Z_Cl, self.Z_Ca])
#        Z_k = Z_k.reshape(4,1)
#        k = k.transpose()
#        q = (k.dot(Z_k) + X)*self.F
#        q = q.transpose()[0]
        Z_k = [self.Z_Na, self.Z_K, self.Z_Cl, self.Z_Ca, self.Z_X]
        q = 0.0
        for i in range(0, 5):
            q += Z_k[i]*k[i]
        q = self.F*q
        return q

    def nernst_potential(self, Z, ck_i, ck_e):
        E = self.R*self.T / (Z*self.F) * np.log(ck_e / ck_i)
        return E

    def reversal_potentials(self):
        E_Na_sn = self.nernst_potential(self.Z_Na, self.cNa_si, self.cNa_se)
        E_Na_sg = self.nernst_potential(self.Z_Na, self.cNa_sg, self.cNa_se)
        E_Na_dn = self.nernst_potential(self.Z_Na, self.cNa_di, self.cNa_de)
        E_Na_dg = self.nernst_potential(self.Z_Na, self.cNa_dg, self.cNa_de)
        E_K_sn = self.nernst_potential(self.Z_K, self.cK_si, self.cK_se)
        E_K_sg = self.nernst_potential(self.Z_K, self.cK_sg, self.cK_se)
        E_K_dn = self.nernst_potential(self.Z_K, self.cK_di, self.cK_de)
        E_K_dg = self.nernst_potential(self.Z_K, self.cK_dg, self.cK_de)
        E_Cl_sn = self.nernst_potential(self.Z_Cl, self.cCl_si, self.cCl_se)
        E_Cl_sg = self.nernst_potential(self.Z_Cl, self.cCl_sg, self.cCl_se)
        E_Cl_dn = self.nernst_potential(self.Z_Cl, self.cCl_di, self.cCl_de)
        E_Cl_dg = self.nernst_potential(self.Z_Cl, self.cCl_dg, self.cCl_de)
        E_Ca_sn = self.nernst_potential(self.Z_Ca, self.free_cCa_si, self.cCa_se)
        E_Ca_dn = self.nernst_potential(self.Z_Ca, self.free_cCa_di, self.cCa_de)
        return E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn

    def membrane_potentials(self):
        I_n_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_si, self.cNa_di) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.cK_si, self.cK_di) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_si, self.cCl_di) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_i, self.free_cCa_si, self.free_cCa_di))
        I_g_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_e, self.cCa_se, self.cCa_de))

        sigma_i = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_si, self.cNa_di) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.cK_si, self.cK_di) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_si, self.cCl_di) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_si, self.free_cCa_di)
        sigma_g = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de)

        q_di = self.total_charge(np.array([self.Na_di, self.K_di, self.Cl_di, self.Ca_di, self.X_di]))
        q_dg = self.total_charge(np.array([self.Na_dg, self.K_dg, self.Cl_dg, 0, self.X_dg]))
        q_si = self.total_charge(np.array([self.Na_si, self.K_si, self.Cl_si, self.Ca_si, self.X_si]))
        q_sg = self.total_charge(np.array([self.Na_sg, self.K_sg, self.Cl_sg, 0, self.X_sg]))

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

        j_Na_in = self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_si, self.cNa_di) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_si, self.cNa_di, phi_si, phi_di) 
        j_K_in = self.j_k_diff(self.D_K, self.lamda_i, self.cK_si, self.cK_di) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.cK_si, self.cK_di, phi_si, phi_di)
        j_Cl_in = self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_si, self.cCl_di) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_si, self.cCl_di, phi_si, phi_di)
        j_Ca_in = self.j_k_diff(self.D_Ca, self.lamda_i, self.free_cCa_si, self.free_cCa_di) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_si, self.free_cCa_di, phi_si, phi_di)

        j_Na_ig = self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg, phi_sg, phi_dg) 
        j_K_ig = self.j_k_diff(self.D_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg, phi_sg, phi_dg)
        j_Cl_ig = self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg, phi_sg, phi_dg)

        j_Na_e = self.j_k_diff(self.D_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de, phi_se, phi_de)
        j_K_e = self.j_k_diff(self.D_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de, phi_se, phi_de)
        j_Cl_e = self.j_k_diff(self.D_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de, phi_se, phi_de)
        j_Ca_e = self.j_k_diff(self.D_Ca, self.lamda_e, self.cCa_se, self.cCa_de) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de, phi_se, phi_de)

        dNadt_si = -j_Na_msn*self.A_sn - j_Na_in*self.A_in 
        dNadt_se = j_Na_msn*self.A_sn + j_Na_msg*self.A_sg - j_Na_e*self.A_e 
        dNadt_sg = -j_Na_msg*self.A_sg - j_Na_ig*self.A_ig
        dNadt_di = -j_Na_mdn*self.A_dn + j_Na_in*self.A_in 
        dNadt_de = j_Na_mdn*self.A_dn + j_Na_mdg*self.A_dg + j_Na_e*self.A_e 
        dNadt_dg = -j_Na_mdg*self.A_dg + j_Na_ig*self.A_ig

        dKdt_si = -j_K_msn*self.A_sn - j_K_in*self.A_in
        dKdt_se = j_K_msn*self.A_sn + j_K_msg*self.A_sg - j_K_e*self.A_e
        dKdt_sg = -j_K_msg*self.A_sg - j_K_ig*self.A_ig
        dKdt_di = -j_K_mdn*self.A_dn + j_K_in*self.A_in
        dKdt_de = j_K_mdn*self.A_dn + j_K_mdg*self.A_dg + j_K_e*self.A_e
        dKdt_dg = -j_K_mdg*self.A_dg + j_K_ig*self.A_ig

        dCldt_si = -j_Cl_msn*self.A_sn - j_Cl_in*self.A_in
        dCldt_se = j_Cl_msn*self.A_sn + j_Cl_msg*self.A_sg - j_Cl_e*self.A_e
        dCldt_sg = -j_Cl_msg*self.A_sg - j_Cl_ig*self.A_ig
        dCldt_di = -j_Cl_mdn*self.A_dn + j_Cl_in*self.A_in
        dCldt_de = j_Cl_mdn*self.A_dn + j_Cl_mdg*self.A_dg + j_Cl_e*self.A_e
        dCldt_dg = -j_Cl_mdg*self.A_dg + j_Cl_ig*self.A_ig

        dCadt_si = - j_Ca_in*self.A_in - self.j_Ca_sn()*self.A_sn
        dCadt_se = - j_Ca_e*self.A_e + self.j_Ca_sn()*self.A_sn
        dCadt_di = j_Ca_in*self.A_in - j_Ca_mdn*(self.A_dn) 
        dCadt_de = j_Ca_e*self.A_e + j_Ca_mdn*(self.A_dn) 


        return dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, \
            dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de

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
