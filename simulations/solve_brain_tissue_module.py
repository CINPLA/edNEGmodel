from brain_tissue_module.brain_tissue_module import *
from scipy.integrate import solve_ivp
import numpy as np

def solve_brain_tissue_module(t_dur, alpha, I_stim, stim_start, stim_end):
    """
    Solves the BrainTissueModule (Sætra et al. 2020) using the solve_ivp function from SciPy.

    Arguments:
        t_dur (float): duration of simulation [s]
        alpha (float): coupling strength
        I_stim (float): stimulus current [A]
        stim_start (float): time of stimulus onset [s]
        stim_end (float): time of stimulus offset [s]
    
    Returns:
        sol: solution from solve_ivp
        my_cell: BrainTissueModule object

    """

    T = 309.14

    # initial volumes [m**3]
    V_sn0 = 1437e-18
    V_se0 = 718.5e-18
    V_sg0 = 1437e-18
    V_dn0 = 1437e-18
    V_de0 = 718.5e-18
    V_dg0 = 1437e-18

    # intitial amounts of ions [mol]
    Na_sn0 = 18.4*V_sn0
    Na_se0 = 144*V_se0
    Na_sg0 = 14.0*V_sg0
    K_sn0 = 137.1*V_sn0
    K_se0 = 3.8*V_se0
    K_sg0 = 102.0*V_sg0
    Cl_sn0 = 4.5*V_sn0
    Cl_se0 = 133.7*V_se0
    Cl_sg0 = 6.0*V_sg0
    Ca_sn0 = 0.01*V_sn0
    Ca_se0 = 1.1*V_se0

    Na_dn0 = 18.4*V_dn0
    Na_de0 = 144*V_de0
    Na_dg0 = 14.0*V_dg0
    K_dn0 = 137.1*V_dn0
    K_de0 = 3.8*V_de0
    K_dg0 = 102.0*V_dg0
    Cl_dn0 = 4.5*V_dn0
    Cl_de0 = 133.7*V_de0
    Cl_dg0 = 6.0*V_dg0
    Ca_dn0 = 0.01*V_dn0
    Ca_de0 = 1.1*V_de0

    # baseline ion concentrations [mol/m**3]
    cbK_se = 3.8
    cbK_sg = 102.0
    cbK_de = 3.8
    cbK_dg = 102.0
    cbCa_sn = 0.01
    cbCa_dn = 0.01

    # residual charges [mol]
    resi = -70.3e-3*3e-2*616e-12/9.648e4 
    resg = -82.6e-3*3e-2*616e-12/9.648e4 
    rese =  resi+resg

    X_sn = Na_sn0 + K_sn0 - Cl_sn0 + 2*Ca_sn0 - resi
    X_se = Na_se0 + K_se0 - Cl_se0 + 2*Ca_se0 + rese
    X_sg = Na_sg0 + K_sg0 - Cl_sg0 - resg
    X_dn = Na_dn0 + K_dn0 - Cl_dn0 + 2*Ca_dn0 - resi
    X_de = Na_de0 + K_de0 - Cl_de0 + 2*Ca_de0 + rese
    X_dg = Na_dg0 + K_dg0 - Cl_dg0 - resg

    # residual mass [mol/m**3]
    cM_sn = (Na_sn0 + K_sn0 + Cl_sn0 + Ca_sn0)/V_sn0
    cM_se = (Na_se0 + K_se0 + Cl_se0 + Ca_se0)/V_se0 
    cM_sg = (Na_sg0 + K_sg0 + Cl_sg0)/V_sg0
    cM_dn = (Na_dn0 + K_dn0 + Cl_dn0 + Ca_dn0)/V_dn0
    cM_de = (Na_de0 + K_de0 + Cl_de0 + Ca_de0)/V_de0 
    cM_dg = (Na_dg0 + K_dg0 + Cl_dg0)/V_dg0 

    # gating variables
    n0 = 0.0002
    h0 = 0.9997
    s0 = 0.0057
    c0 = 0.0042
    q0 = 0.0092
    z0 = 1.0

    # print initial values
    init_cell = BrainTissueModule(T, Na_sn0, Na_se0, Na_sg0, Na_dn0, Na_de0, Na_dg0, K_sn0, K_se0, K_sg0, K_dn0, K_de0, K_dg0, Cl_sn0, Cl_se0, Cl_sg0, Cl_dn0, Cl_de0, Cl_dg0, Ca_sn0, Ca_se0, Ca_dn0, Ca_de0, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n0, h0, s0, c0, q0, z0, V_sn0, V_se0, V_sg0, V_dn0, V_de0, V_dg0, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)

    phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = init_cell.membrane_potentials()
    E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = init_cell.reversal_potentials()

    q_sn = init_cell.total_charge([init_cell.Na_sn, init_cell.K_sn, init_cell.Cl_sn, init_cell.Ca_sn, init_cell.X_sn])
    q_se = init_cell.total_charge([init_cell.Na_se, init_cell.K_se, init_cell.Cl_se, init_cell.Ca_se, init_cell.X_se])        
    q_sg = init_cell.total_charge([init_cell.Na_sg, init_cell.K_sg, init_cell.Cl_sg, 0, init_cell.X_sg])        
    q_dn = init_cell.total_charge([init_cell.Na_dn, init_cell.K_dn, init_cell.Cl_dn, init_cell.Ca_dn, init_cell.X_dn])
    q_de = init_cell.total_charge([init_cell.Na_de, init_cell.K_de, init_cell.Cl_de, init_cell.Ca_de, init_cell.X_de])
    q_dg = init_cell.total_charge([init_cell.Na_dg, init_cell.K_dg, init_cell.Cl_dg, 0, init_cell.X_dg])
    print("----------------------------")
    print("Initial values")
    print("----------------------------")
    print("initial total charge(C):", q_sn + q_se + q_sg + q_dn + q_de + q_dg)
    print("Q_sn + Q_sg (C):", q_sn+q_sg)
    print("Q_se (C):", q_se)
    print("Q_dn + Q_sg (C):", q_dn+q_dg)
    print("Q_de (C):", q_de)
    print("----------------------------")
    print('phi_sn: ', round(phi_sn*1000))
    print('phi_se: ', round(phi_se*1000))
    print('phi_sg: ', round(phi_sg*1000))
    print('phi_dn: ', round(phi_dn*1000))
    print('phi_de: ', round(phi_de*1000))
    print('phi_dg: ', round(phi_dg*1000))
    print('phi_msn: ', round(phi_msn*1000))
    print('phi_mdn: ', round(phi_mdn*1000))
    print('phi_msg: ', round(phi_msg*1000))
    print('phi_mdg: ', round(phi_mdg*1000))
    print('E_Na_sn: ', round(E_Na_sn*1000))
    print('E_Na_sg: ', round(E_Na_sg*1000))
    print('E_K_sn: ', round(E_K_sn*1000))
    print('E_K_sg: ', round(E_K_sg*1000))
    print('E_Cl_sn: ', round(E_Cl_sn*1000))
    print('E_Cl_sg: ', round(E_Cl_sg*1000))
    print('E_Ca_sn: ', round(E_Ca_sn*1000))
    print("----------------------------")
    print('psi_se-psi_sn', init_cell.psi_se-init_cell.psi_sn) 
    print('psi_se-psi_sg', init_cell.psi_se-init_cell.psi_sg) 
    print('psi_de-psi_dn', init_cell.psi_de-init_cell.psi_dn) 
    print('psi_de-psi_dg', init_cell.psi_de-init_cell.psi_dg) 
    print("----------------------------")
    print('initial total volume (m^3):', V_sn0 + V_se0 + V_sg0 + V_dn0 + V_de0 + V_dg0)
    print("----------------------------")

    # define differential equations
    def dkdt(t,k):

        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k

        my_cell = BrainTissueModule(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)

        dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()

        if t > stim_start and t < stim_end:
            dKdt_sn += I_stim / my_cell.F
            dKdt_se -= I_stim / my_cell.F

        return dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt 

    # solve 
    t_span = (0, t_dur)

    k0 = [Na_sn0, Na_se0, Na_sg0, Na_dn0, Na_de0, Na_dg0, K_sn0, K_se0, K_sg0, K_dn0, K_de0, K_dg0, Cl_sn0, Cl_se0, Cl_sg0, Cl_dn0, Cl_de0, Cl_dg0, Ca_sn0, Ca_se0, Ca_dn0, Ca_de0, n0, h0, s0, c0, q0, z0, V_sn0, V_se0, V_sg0, V_dn0, V_de0, V_dg0]

    sol = solve_ivp(dkdt, t_span, k0, method='RK23', max_step=1e-4)

    Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg  = sol.y

    my_cell = BrainTissueModule(T, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, cbK_se, cbK_sg, cbK_de, cbK_dg, cbCa_sn, cbCa_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)

    return sol, my_cell
