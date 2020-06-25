import numpy as np

class BrainTissueModule():
    """ 
    An electrodiffusive Pinsky-Rinzel model with neuron-glia interactions and cellular swelling
    
    Methods
    -------
    constructor(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, \
        Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, \
        X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, \
        cbK_se, cbK_sg, cbK_de, cbK_dg, \
        cbCa_sn, cbCa_dn, n, h, s, c, q, z, \
        V_sn, V_se, V_sg, V_dn, V_de, V_dg, \
        cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)
    j_Na_sn(phi_m, E_Na): compute the Na+ membrane flux (soma layer, neuron)
    j_K_sn(phi_m, E_K): compute the K+ membrane flux (soma layer, neuron)
    j_Cl_sn(phi_m, E_Cl): compute the Cl- membrane flux (soma layer, neuron) 
    j_Ca_sn(): compute the Ca2+ membrane flux (soma layer, neuron) 
    j_Na_dn(phi_m, E_Na): compute the Na+ membrane flux (dendrite layer, neuron)
    j_K_dn(phi_m, E_K): compute the K+ membrane flux (dendrite layer, neuron)
    j_Cl_dn(phi_m, E_Cl): compute the Cl- membrane flux (dendrite layer, neuron) 
    j_Ca_dn(phi_m, E_Cl): compute the Ca2+ membrane flux (dendrite layer, neuron) 
    j_Na_sg(phi_m, E_Na): compute the Na+ membrane flux (soma layer, glia)
    j_K_sg(phi_m, E_K): compute the K+ membrane flux (soma layer, glia)
    j_Cl_sg(phi_m, E_Cl): compute the Cl- membrane flux (soma layer, glia) 
    j_Na_dg(phi_m, E_Na): compute the Na+ membrane flux (dendrite layer, glia)
    j_K_dg(phi_m, E_K): compute the K+ membrane flux (dendrite layer, glia)
    j_Cl_dg(phi_m, E_Cl): compute the Cl- membrane flux (dendrite layer, glia) 
    j_pump_n(cNa_n, cK_e): compute the Na+/K+ pump flux across neuronal membrane
    j_pump_g(cNa_g, cK_e): compute the Na+/K+ pump flux across glial membrane
    j_kcc2(cK_n, cK_e, cCl_n, cCl_e): compute the K+/Cl- cotransporter flux across neuronal membrane
    j_nkcc1(cNa_n, cNa_e, cK_n, cK_e, cCl_n, cCl_e): compute the Na+/K+/Cl- cotransporter flux across neuronal membrane
    j_k_diff(D_k, tortuosity, ck_s, ck_d): compute the axial diffusion flux of ion k
    j_k_drift(D_k, Z_k, tortuosity, ck_s, ck_d, phi_s, phi_d): compute the axial drift flux of ion k
    conductivity_k(D_k, Z_k, tortuosity, ck_s, ck_d): compute axial conductivity of ion k
    total_charge(k): calculate the total charge within a compartment
    nernst_potential(Z, ck_i, ck_e): calculate the reversal potential of ion k
    reversal_potentials(): calculate the reversal potentials of all ion species
    membrane_potentials(): calculate the membrane potentials
    dkdt(): calculate dk/dt for all ion species k
    dmdt(): calculate dm/dt for all gating particles m
    dVdt(): calculate dV/dt for all volumes V
    """


    def __init__(self, T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, \
        Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, \
        X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, \
        cbK_se, cbK_sg, cbK_de, cbK_dg, \
        cbCa_sn, cbCa_dn, n, h, s, c, q, z, \
        V_sn, V_se, V_sg, V_dn, V_de, V_dg, \
        cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg):

        # temperature [K]
        self.T = T

        # ions [mol]
        self.Na_sn = Na_sn
        self.Na_se = Na_se
        self.Na_sg = Na_sg
        self.Na_dn = Na_dn
        self.Na_de = Na_de
        self.Na_dg = Na_dg
        self.K_sn = K_sn
        self.K_se = K_se
        self.K_sg = K_sg
        self.K_dn = K_dn
        self.K_de = K_de
        self.K_dg = K_dg
        self.Cl_sn = Cl_sn
        self.Cl_se = Cl_se 
        self.Cl_sg = Cl_sg 
        self.Cl_dn = Cl_dn 
        self.Cl_de = Cl_de
        self.Cl_dg = Cl_dg
        self.Ca_sn = Ca_sn
        self.Ca_se = Ca_se 
        self.Ca_dn = Ca_dn 
        self.Ca_de = Ca_de
        self.X_sn = X_sn
        self.X_se = X_se
        self.X_sg = X_sg
        self.X_dn = X_dn
        self.X_de = X_de
        self.X_dg = X_dg
        
        # ion concentraions [mol/m**3]
        self.cNa_sn = Na_sn/V_sn
        self.cNa_se = Na_se/V_se
        self.cNa_sg = Na_sg/V_sg
        self.cNa_dn = Na_dn/V_dn
        self.cNa_de = Na_de/V_de
        self.cNa_dg = Na_dg/V_dg
        self.cK_sn = K_sn/V_sn
        self.cK_se = K_se/V_se
        self.cK_sg = K_sg/V_sg
        self.cK_dn = K_dn/V_dn
        self.cK_de = K_de/V_de
        self.cK_dg = K_dg/V_dg
        self.cCl_sn = Cl_sn/V_sn
        self.cCl_se = Cl_se/V_se 
        self.cCl_sg = Cl_sg/V_sg 
        self.cCl_dn = Cl_dn/V_dn 
        self.cCl_de = Cl_de/V_de
        self.cCl_dg = Cl_dg/V_dg
        self.cCa_sn = Ca_sn/V_sn
        self.cCa_se = Ca_se/V_se 
        self.cCa_dn = Ca_dn/V_dn 
        self.cCa_de = Ca_de/V_de
        self.free_cCa_sn = 0.01*self.cCa_sn
        self.free_cCa_dn = 0.01*self.cCa_dn
        self.cX_sn = X_sn/V_sn
        self.cX_se = X_se/V_se
        self.cX_sg = X_sg/V_sg
        self.cX_dn = X_dn/V_dn
        self.cX_de = X_de/V_de
        self.cX_dg = X_dg/V_dg

        # concentrations of static molecules without charge [mol*m**-3] 
        self.cM_sn = cM_sn
        self.cM_se = cM_se
        self.cM_sg = cM_sg
        self.cM_dn = cM_dn
        self.cM_de = cM_de
        self.cM_dg = cM_dg

        # gating variables
        self.n = n
        self.h = h
        self.s = s
        self.c = c
        self.q = q
        self.z = z
        
        # baseline concentrations [mol/m**3] 
        self.cbK_se = cbK_se           
        self.cbK_sg = cbK_sg          
        self.cbK_de = cbK_de     
        self.cbK_dg = cbK_dg
        self.cbCa_sn = cbCa_sn
        self.cbCa_dn = cbCa_dn

        # threshold concentrations for the glial Na/K pump
        self.cNa_treshold = 10 # Halnes et al. 2013
        self.cK_treshold = 1.5 # Halnes et al. 2013

        # membrane capacitance [F/m**2]
        self.C_msn = 3e-2 # Pinsky and Rinzel 1994
        self.C_mdn = 3e-2 # Pinsky and Rinzel 1994
        self.C_msg = 3e-2 # Pinsky and Rinzel 1994
        self.C_mdg = 3e-2 # Pinsky and Rinzel 1994
       
        # volumes and areas
        self.alpha = alpha
        self.A_m = 616e-12                # [m**2] Saetra et al. 2020
        self.A_i = self.alpha*self.A_m    # [m**2] Saetra et al. 2020
        self.A_e = self.A_i/2.            # [m**2] Saetra et al. 2020
        self.dx = 667e-6                  # [m] Saetra et al. 2020
        self.V_sn = V_sn                  # [m**3]
        self.V_se = V_se                  # [m**3]
        self.V_sg = V_sg                  # [m**3]
        self.V_dn = V_dn                  # [m**3]
        self.V_de = V_de                  # [m**3]
        self.V_dg = V_dg                  # [m**3]
 
        # diffusion constants [m**2/s]
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
        self.F = 9.648e4    # [C/mol]
        self.R = 8.314      # [J/mol/K] 

        # conductances [S/m**2]
        self.g_Na_leak_n = 0.246
        self.g_K_leak_n = 0.245  
        self.g_Cl_leak_n = 0.668
        self.g_Na = 300.           # Pinsky and Rinzel 1994
        self.g_DR = 150.           # Pinsky and Rinzel 1994
        self.g_Ca = 141.
        self.g_AHP = 8.            # Pinsky and Rinzel 1994
        self.g_C = 150.            # Pinsky and Rinzel 1994
        self.g_Na_leak_g = 1.      # Halnes et al. 2013
        self.g_K_IR = 16.96        # Halnes et al. 2013
        self.g_Cl_leak_g = 0.5     # Halnes et al. 2013
        
        # exchanger strengths
        self.rho_n = 1.87e-6       # [mol/m**2/s] Wei et al. 2014
        self.U_kcc2 = 7.00e-7      # [mol/m**2/s] Wei et al. 2014
        self.U_nkcc1 = 2.33e-7     # [mol/m**2/s] Wei et al. 2014
        self.U_Cadec = 75.         # [1/s] Saetra et al. 2020
        self.rho_g = 1.12e-6       # [mol/m**2/s] Halnes et al. 2013 
        
        # water permeabilities [m**3/Pa/s] 
        self.G_n = 2e-23    # Dijkstra et al. 2016
        self.G_g = 5e-23    # Oestby et al. 2009
        
        # baseline reversal potentials [V]
        self.bE_K_sg = self.nernst_potential(self.Z_K, self.cbK_sg, self.cbK_se)
        self.bE_K_dg = self.nernst_potential(self.Z_K, self.cbK_dg, self.cbK_de)


        # solute potentials [Pa]
        self.psi_sn = -self.R * self.T * (self.cNa_sn + self.cK_sn + self.cCl_sn + self.cCa_sn -  cM_sn)
        self.psi_se = -self.R * self.T * (self.cNa_se + self.cK_se + self.cCl_se + self.cCa_se - cM_se)
        self.psi_sg = -self.R * self.T * (self.cNa_sg + self.cK_sg + self.cCl_sg - cM_sg)
        self.psi_dn = -self.R * self.T * (self.cNa_dn + self.cK_dn + self.cCl_dn + self.cCa_dn - cM_dn)
        self.psi_de = -self.R * self.T * (self.cNa_de + self.cK_de + self.cCl_de + self.cCa_de - cM_de)
        self.psi_dg = -self.R * self.T * (self.cNa_dg + self.cK_dg + self.cCl_dg - cM_dg)

    def alpha_m(self, phi_m):
        phi_1 = phi_m + 0.0469
        alpha = -3.2e5 * phi_1 / (np.exp(-phi_1 / 0.004) - 1.)
        return alpha

    def beta_m(self, phi_m):
        phi_2 = phi_m + 0.0199
        beta = 2.8e5 * phi_2 / (np.exp(phi_2 / 0.005) - 1)
        return beta

    def alpha_h(self, phi_m):
        alpha = 128 * np.exp((-0.043 - phi_m) / 0.018)
        return alpha

    def beta_h(self, phi_m):
        phi_3 = phi_m + 0.02
        beta = 4000 / (1 + np.exp(-phi_3 / 0.005))
        return beta

    def alpha_n(self, phi_m):
        phi_4 = phi_m + 0.0249
        alpha = - 1.6e4 * phi_4 / (np.exp(-phi_4 / 0.005) - 1)
        return alpha

    def beta_n(self, phi_m):
        phi_5 = phi_m + 0.04
        beta = 250 * np.exp(-phi_5 / 0.04)
        return beta

    def alpha_s(self, phi_m):
        alpha = 1600 / (1 + np.exp(-72 * (phi_m - 0.005)))
        return alpha

    def beta_s(self, phi_m):
        phi_6 = phi_m + 0.0089
        beta = 2e4 * phi_6 / (np.exp(phi_6 / 0.005) - 1.)
        return beta

    def alpha_c(self, phi_m):
        phi_8 = phi_m + 0.05
        phi_9 = phi_m + 0.0535
        if phi_m <= -0.01:
            alpha = 52.7 * np.exp(phi_8/0.011- phi_9/0.027)
        else:
            alpha = 2000 * np.exp(-phi_9 / 0.027)
        return alpha

    def beta_c(self, phi_m):
        phi_9 = phi_m + 0.0535
        if phi_m <= -0.01:
            beta = 2000 * np.exp(-phi_9 / 0.027) - self.alpha_c(phi_m)
        else:
            beta = 0.
        return beta

    def chi(self):
        return min((self.free_cCa_dn-99.8e-6)/2.5e-4, 1.0)

    def alpha_q(self):
        return min(2e4*(self.free_cCa_dn-99.8e-6), 10.0) 

    def beta_q(self):
        return 1.0

    def m_inf(self, phi_m):
        return self.alpha_m(phi_m) / (self.alpha_m(phi_m) + self.beta_m(phi_m))

    def z_inf(self, phi_m):
        phi_7 = phi_m + 0.03
        return 1/(1 + np.exp(phi_7/0.001))

    def j_pump_n(self, cNa_n, cK_e):
        j = (self.rho_n / (1.0 + np.exp((25. - cNa_n)/3.))) * (1.0 / (1.0 + np.exp(3.5 - cK_e)))
        return j

    def j_pump_g(self, cNa_g, cK_e):
        j = self.rho_g * (cNa_g**1.5 / (cNa_g**1.5 + self.cNa_treshold**1.5)) * (cK_e / (cK_e + self.cK_treshold))
        return j

    def j_kcc2(self, cK_n, cK_e, cCl_n, cCl_e):
        j = self.U_kcc2 * np.log(cK_n*cCl_n/(cK_e*cCl_e))
        return j
    
    def j_nkcc1(self, cNa_n, cNa_e, cK_n, cK_e, cCl_n, cCl_e):
        j = self.U_nkcc1 * (1 / (1 + np.exp(16 - cK_e))) * (np.log(cK_n*cCl_n/(cK_e*cCl_e)) + np.log(cNa_n*cCl_n/(cNa_e*cCl_e)))
        return j

    def j_Na_sn(self, phi_m, E_Na):
        j = self.g_Na_leak_n*(phi_m - E_Na) / (self.F*self.Z_Na) \
            + 3*self.j_pump_n(self.cNa_sn, self.cK_se) \
            + self.j_nkcc1(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + self.g_Na * self.m_inf(phi_m)**2 * self.h * (phi_m - E_Na) / (self.F*self.Z_Na) \
            - 2*self.U_Cadec*(self.cCa_sn - self.cbCa_sn)*self.V_sn/self.A_m
        return j 

    def j_K_sn(self, phi_m, E_K):
        j = self.g_K_leak_n*(phi_m - E_K) / (self.F*self.Z_K) \
            - 2*self.j_pump_n(self.cNa_sn, self.cK_se) \
            + self.j_kcc2(self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + self.j_nkcc1(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + self.g_DR * self.n * (phi_m - E_K) / (self.F*self.Z_K)
        return j

    def j_Cl_sn(self, phi_m, E_Cl):
        j = self.g_Cl_leak_n*(phi_m - E_Cl) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se) \
            + 2*self.j_nkcc1(self.cNa_sn, self.cNa_se, self.cK_sn, self.cK_se, self.cCl_sn, self.cCl_se)
        return j

    def j_Ca_sn(self):
        j =  self.U_Cadec * (self.cCa_sn - self.cbCa_sn)*self.V_sn/self.A_m
        return j

    def j_Na_dn(self, phi_m, E_Na):
        j = self.g_Na_leak_n*(phi_m - E_Na) / (self.F*self.Z_Na) \
            + 3*self.j_pump_n(self.cNa_dn, self.cK_de) \
            + self.j_nkcc1(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            - 2*self.U_Cadec*(self.cCa_dn - self.cbCa_dn)*self.V_dn/self.A_m
        return j

    def j_K_dn(self, phi_m, E_K):
        j = self.g_K_leak_n*(phi_m - E_K) / (self.F*self.Z_K) \
            - 2*self.j_pump_n(self.cNa_dn, self.cK_de) \
            + self.j_kcc2(self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            + self.j_nkcc1(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            + self.g_AHP * self.q * (phi_m - E_K) / (self.F*self.Z_K) \
            + self.g_C * self.c * self.chi() * (phi_m - E_K) / (self.F*self.Z_K)
        return j

    def j_Cl_dn(self, phi_m, E_Cl):
        j = self.g_Cl_leak_n*(phi_m - E_Cl) / (self.F*self.Z_Cl) \
            + self.j_kcc2(self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de) \
            + 2*self.j_nkcc1(self.cNa_dn, self.cNa_de, self.cK_dn, self.cK_de, self.cCl_dn, self.cCl_de)
        return j

    def j_Ca_dn(self, phi_m, E_Ca):
        j = self.g_Ca * self.s**2 * self.z * (phi_m - E_Ca) / (self.F*self.Z_Ca) \
            + self.U_Cadec*(self.cCa_dn - self.cbCa_dn)*self.V_dn/self.A_m
        return j

    def j_Na_sg(self, phi_m, E_Na):
        j = self.g_Na_leak_g * (phi_m - E_Na) / self.F \
            + 3*self.j_pump_g(self.cNa_sg, self.cK_se)
        return j

    def j_K_sg(self, phi_m, E_K):
        dphi = (phi_m - E_K)*1000
        phi_m_mil = phi_m*1000
        bE_K_mil = self.bE_K_sg*1000
        fact1 = (1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1 + np.exp(-(118.6+bE_K_mil)/44.1))/(1+np.exp(-(118.6+phi_m_mil)/44.1))
        f = np.sqrt(self.cK_se/self.cbK_se) * fact1 * fact2 
        j = self.g_K_IR * f * (phi_m - E_K) / self.F \
            - 2 * self.j_pump_g(self.cNa_sg, self.cK_se)
        return j

    def j_Cl_sg(self, phi_m, E_Cl):
        j = - self.g_Cl_leak_g * (phi_m - E_Cl) / self.F
        return j

    def j_Na_dg(self, phi_m, E_Na):
        j = self.g_Na_leak_g * (phi_m - E_Na) / self.F \
            + 3*self.j_pump_g(self.cNa_dg, self.cK_de)
        return j

    def j_K_dg(self, phi_m, E_K):
        dphi = (phi_m - E_K)*1000
        phi_m_mil = phi_m*1000
        bE_K_mil = self.bE_K_dg*1000
        fact1 = (1 + np.exp(18.4/42.4))/(1 + np.exp((dphi + 18.5)/42.5))
        fact2 = (1 + np.exp(-(118.6+bE_K_mil)/44.1))/(1+np.exp(-(118.6+phi_m_mil)/44.1))
        f = np.sqrt(self.cK_de/self.cbK_de) * fact1 * fact2 
        j = self.g_K_IR * f * (phi_m - E_K) / self.F \
            - 2 * self.j_pump_g(self.cNa_dg, self.cK_de)
        return j

    def j_Cl_dg(self, phi_m, E_Cl):
        j = - self.g_Cl_leak_g * (phi_m - E_Cl) / self.F
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
        E_Na_sn = self.nernst_potential(self.Z_Na, self.cNa_sn, self.cNa_se)
        E_Na_sg = self.nernst_potential(self.Z_Na, self.cNa_sg, self.cNa_se)
        E_Na_dn = self.nernst_potential(self.Z_Na, self.cNa_dn, self.cNa_de)
        E_Na_dg = self.nernst_potential(self.Z_Na, self.cNa_dg, self.cNa_de)
        E_K_sn = self.nernst_potential(self.Z_K, self.cK_sn, self.cK_se)
        E_K_sg = self.nernst_potential(self.Z_K, self.cK_sg, self.cK_se)
        E_K_dn = self.nernst_potential(self.Z_K, self.cK_dn, self.cK_de)
        E_K_dg = self.nernst_potential(self.Z_K, self.cK_dg, self.cK_de)
        E_Cl_sn = self.nernst_potential(self.Z_Cl, self.cCl_sn, self.cCl_se)
        E_Cl_sg = self.nernst_potential(self.Z_Cl, self.cCl_sg, self.cCl_se)
        E_Cl_dn = self.nernst_potential(self.Z_Cl, self.cCl_dn, self.cCl_de)
        E_Cl_dg = self.nernst_potential(self.Z_Cl, self.cCl_dg, self.cCl_de)
        E_Ca_sn = self.nernst_potential(self.Z_Ca, self.free_cCa_sn, self.cCa_se)
        E_Ca_dn = self.nernst_potential(self.Z_Ca, self.free_cCa_dn, self.cCa_de)
        return E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn

    def membrane_potentials(self):
        I_n_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sn, self.cNa_dn) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.cK_sn, self.cK_dn) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn))
        I_g_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg))
        I_e_diff = self.F * (self.Z_Na*self.j_k_diff(self.D_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.Z_K*self.j_k_diff(self.D_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.Z_Cl*self.j_k_diff(self.D_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.Z_Ca*self.j_k_diff(self.D_Ca, self.lamda_e, self.cCa_se, self.cCa_de))

        sigma_n = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn)
        sigma_g = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sg, self.cNa_dg) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_i, self.cK_sg, self.cK_dg) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sg, self.cCl_dg)
        sigma_e = self.conductivity_k(self.D_Na, self.Z_Na, self.lamda_e, self.cNa_se, self.cNa_de) \
            + self.conductivity_k(self.D_K, self.Z_K, self.lamda_e, self.cK_se, self.cK_de) \
            + self.conductivity_k(self.D_Cl, self.Z_Cl, self.lamda_e, self.cCl_se, self.cCl_de) \
            + self.conductivity_k(self.D_Ca, self.Z_Ca, self.lamda_e, self.cCa_se, self.cCa_de)

        q_dn = self.total_charge(np.array([self.Na_dn, self.K_dn, self.Cl_dn, self.Ca_dn, self.X_dn]))
        q_dg = self.total_charge(np.array([self.Na_dg, self.K_dg, self.Cl_dg, 0, self.X_dg]))
        q_sn = self.total_charge(np.array([self.Na_sn, self.K_sn, self.Cl_sn, self.Ca_sn, self.X_sn]))
        q_sg = self.total_charge(np.array([self.Na_sg, self.K_sg, self.Cl_sg, 0, self.X_sg]))

        phi_dn = q_dn / (self.C_mdn * self.A_m)
        phi_de = 0.
        phi_dg = q_dg / (self.C_mdg * self.A_m)
        phi_se = ( - self.dx * self.A_i * I_n_diff + self.A_i * sigma_n * phi_dn - self.A_i * sigma_n * q_sn / (self.C_msn * self.A_m) \
            - self.dx * self.A_i * I_g_diff + self.A_i * sigma_g * phi_dg - self.A_i * sigma_g * q_sg / (self.C_msg * self.A_m) - self.dx * self.A_e * I_e_diff ) \
            / ( self.A_e * sigma_e + self.A_i * sigma_n + self.A_i * sigma_g )
        phi_sn = q_sn / (self.C_msn * self.A_m) + phi_se
        phi_sg = q_sg / (self.C_msg * self.A_m) + phi_se
        phi_msn = phi_sn - phi_se
        phi_msg = phi_sg - phi_se
        phi_mdn = phi_dn - phi_de
        phi_mdg = phi_dg - phi_de

        return phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg

    def dkdt(self):
       
        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
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

        j_Na_in = self.j_k_diff(self.D_Na, self.lamda_i, self.cNa_sn, self.cNa_dn) \
            + self.j_k_drift(self.D_Na, self.Z_Na, self.lamda_i, self.cNa_sn, self.cNa_dn, phi_sn, phi_dn) 
        j_K_in = self.j_k_diff(self.D_K, self.lamda_i, self.cK_sn, self.cK_dn) \
            + self.j_k_drift(self.D_K, self.Z_K, self.lamda_i, self.cK_sn, self.cK_dn, phi_sn, phi_dn)
        j_Cl_in = self.j_k_diff(self.D_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn) \
            + self.j_k_drift(self.D_Cl, self.Z_Cl, self.lamda_i, self.cCl_sn, self.cCl_dn, phi_sn, phi_dn)
        j_Ca_in = self.j_k_diff(self.D_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn) \
            + self.j_k_drift(self.D_Ca, self.Z_Ca, self.lamda_i, self.free_cCa_sn, self.free_cCa_dn, phi_sn, phi_dn)

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

        dNadt_sn = -j_Na_msn*self.A_m - j_Na_in*self.A_i 
        dNadt_se = j_Na_msn*self.A_m + j_Na_msg*self.A_m - j_Na_e*self.A_e 
        dNadt_sg = -j_Na_msg*self.A_m - j_Na_ig*self.A_i
        dNadt_dn = -j_Na_mdn*self.A_m + j_Na_in*self.A_i 
        dNadt_de = j_Na_mdn*self.A_m + j_Na_mdg*self.A_m + j_Na_e*self.A_e 
        dNadt_dg = -j_Na_mdg*self.A_m + j_Na_ig*self.A_i

        dKdt_sn = -j_K_msn*self.A_m - j_K_in*self.A_i
        dKdt_se = j_K_msn*self.A_m + j_K_msg*self.A_m - j_K_e*self.A_e
        dKdt_sg = -j_K_msg*self.A_m - j_K_ig*self.A_i
        dKdt_dn = -j_K_mdn*self.A_m + j_K_in*self.A_i
        dKdt_de = j_K_mdn*self.A_m + j_K_mdg*self.A_m + j_K_e*self.A_e
        dKdt_dg = -j_K_mdg*self.A_m + j_K_ig*self.A_i

        dCldt_sn = -j_Cl_msn*self.A_m - j_Cl_in*self.A_i
        dCldt_se = j_Cl_msn*self.A_m + j_Cl_msg*self.A_m - j_Cl_e*self.A_e
        dCldt_sg = -j_Cl_msg*self.A_m - j_Cl_ig*self.A_i
        dCldt_dn = -j_Cl_mdn*self.A_m + j_Cl_in*self.A_i
        dCldt_de = j_Cl_mdn*self.A_m + j_Cl_mdg*self.A_m + j_Cl_e*self.A_e
        dCldt_dg = -j_Cl_mdg*self.A_m + j_Cl_ig*self.A_i

        dCadt_sn = - j_Ca_in*self.A_i - self.j_Ca_sn()*self.A_m
        dCadt_se = - j_Ca_e*self.A_e + self.j_Ca_sn()*self.A_m
        dCadt_dn = j_Ca_in*self.A_i - j_Ca_mdn*self.A_m 
        dCadt_de = j_Ca_e*self.A_e + j_Ca_mdn*self.A_m 

        return dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de

    def dmdt(self):
        phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg  = self.membrane_potentials()
        
        dndt = self.alpha_n(phi_msn)*(1.0-self.n) - self.beta_n(phi_msn)*self.n
        dhdt = self.alpha_h(phi_msn)*(1.0-self.h) - self.beta_h(phi_msn)*self.h 
        dsdt = self.alpha_s(phi_mdn)*(1.0-self.s) - self.beta_s(phi_mdn)*self.s
        dcdt = self.alpha_c(phi_mdn)*(1.0-self.c) - self.beta_c(phi_mdn)*self.c
        dqdt = self.alpha_q()*(1.0-self.q) - self.beta_q()*self.q
        dzdt = (self.z_inf(phi_mdn) - self.z)
        
        return dndt, dhdt, dsdt, dcdt, dqdt, dzdt

    def dVdt(self):

        dVsidt = self.G_n * (self.psi_se - self.psi_sn)
        dVsgdt = self.G_g * (self.psi_se - self.psi_sg)
        dVdidt = self.G_n * (self.psi_de - self.psi_dn)
        dVdgdt = self.G_g * (self.psi_de - self.psi_dg)
        dVsedt = - (dVsidt + dVsgdt)
        dVdedt = - (dVdidt + dVdgdt)

        return dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt
