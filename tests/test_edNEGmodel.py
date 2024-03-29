from scipy.integrate import solve_ivp
from edNEGmodel.edNEGmodel import *
import numpy as np

def test_modules():

    alpha = 2.0
    
    V_sn = 1.437e-9             # [cm**3]
    V_se = 7.185e-10            # [cm**3]
    V_sg = 1.437e-9             # [cm**3]
    V_dn = 1.437e-9             # [cm**3]
    V_de = 7.185e-10            # [cm**3]
    V_dg = 1.437e-9             # [cm**3]
    Na_sn = 14*V_sn * 1e3
    Na_se = 145*V_se * 1e3
    Na_sg = 14.6*V_sg * 1e3
    K_sn = 100*V_sn * 1e3
    K_se = 3.45*V_se * 1e3
    K_sg = 101*V_sg * 1e3
    Cl_sn = 115*V_sn * 1e3
    Cl_se = 138*V_se * 1e3
    Cl_sg = 6*V_sg * 1e3
    Ca_sn = 0.01*V_sn * 1e3
    Ca_se = 1.1*V_se * 1e3

    Na_dn = 16*V_dn * 1e3
    Na_de = 145*V_de * 1e3
    Na_dg = 14.6*V_dg * 1e3
    K_dn = 97*V_dn * 1e3
    K_de = 3.45*V_de * 1e3
    K_dg = 101*V_dg * 1e3
    Cl_dn = 6*V_dn * 1e3
    Cl_de = 138*V_de * 1e3
    Cl_dg = 6*V_dg * 1e3
    Ca_dn = 0.01*V_dn * 1e3
    Ca_de = 1.1*V_de * 1e3

    resi = -72*3e-2*616e-12/9.648e4 *1e6 
    resg = -83*3e-2*616e-12/9.648e4 *1e6 
    rese =  resi+resg

    X_sn = Na_sn + K_sn - Cl_sn + 2*Ca_sn - resi
    X_se = Na_se + K_se - Cl_se + 2*Ca_se + rese
    X_sg = Na_sg + K_sg - Cl_sg - resg
    X_dn = Na_dn + K_dn - Cl_dn + 2*Ca_dn - resi
    X_de = Na_de + K_de - Cl_de + 2*Ca_de + rese
    X_dg = Na_dg + K_dg - Cl_dg - resg

    cM_sn = 122.01 * 1e3
    cM_se = 287.55 * 1e3
    cM_sg = 121.6 * 1e3
    cM_dn = 122.01 * 1e3
    cM_de = 287.55 * 1e3
    cM_dg = 121.6 * 1e3

    n = 0.001
    h = 0.999
    s = 0.009
    c = 0.007
    q = 0.01
    z = 1.0

    test_cell = edNEGmodel(279, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, K_se/V_se, K_sg/V_sg, K_de/V_de, K_dg/V_dg, Ca_sn/V_sn, Ca_dn/V_dn, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)

    assert round(test_cell.nernst_potential(1., 400.* 1e3, 20.* 1e3), 4) == -72.0244

    assert round(test_cell.conductivity_k(test_cell.D_Na, test_cell.Z_Na, 3.2, test_cell.cNa_sn, test_cell.cNa_dn), 4) == 0.0782

    assert test_cell.total_charge(np.array([10, 10, 20, 0, 0])) == 0 

    assert test_cell.total_charge(np.array([10, 10, 10, 5, 20])) == 0 

def test_charge_conservation():
    """Tests that no charge disappear.
       Tests charge symmetry."""
   
    alpha = 1.0
    
    EPS = 1e-12

    V_sn0 = 1.437e-9             # [cm**3]
    V_se0 = 7.185e-10            # [cm**3]
    V_sg0 = 1.437e-9             # [cm**3]
    V_dn0 = 1.437e-9             # [cm**3]
    V_de0 = 7.185e-10            # [cm**3]
    V_dg0 = 1.437e-9             # [cm**3]
    Na_sn0 = 19*V_sn0 * 1e3
    Na_se0 = 145*V_se0 * 1e3
    Na_sg0 = 14.6*V_sg0 * 1e3
    K_sn0 = 97*V_sn0 * 1e3
    K_se0 = 3.45*V_se0 * 1e3
    K_sg0 = 101*V_sg0 * 1e3
    Cl_sn0 = 6*V_sn0 * 1e3
    Cl_se0 = 138*V_se0 * 1e3
    Cl_sg0 = 6*V_sg0 * 1e3
    Ca_sn0 = 0.01*V_sn0 * 1e3
    Ca_se0 = 1.1*V_se0 * 1e3

    Na_dn0 = 19*V_dn0 * 1e3
    Na_de0 = 145*V_de0 * 1e3
    Na_dg0 = 14.6*V_dg0 * 1e3
    K_dn0 = 97*V_dn0 * 1e3
    K_de0 = 3.45*V_de0 * 1e3
    K_dg0 = 101*V_dg0 * 1e3
    Cl_dn0 = 6*V_dn0 * 1e3
    Cl_de0 = 138*V_de0 * 1e3
    Cl_dg0 = 6*V_dg0 * 1e3
    Ca_dn0 = 0.01*V_dn0 * 1e3
    Ca_de0 = 1.1*V_de0 * 1e3

    resi = -72*3e-2*616e-12/9.648e4 *1e6  
    resg = -83*3e-2*616e-12/9.648e4 *1e6  
    rese =  resi+resg

    X_sn = Na_sn0 + K_sn0 - Cl_sn0 + 2*Ca_sn0 - resi
    X_se = Na_se0 + K_se0 - Cl_se0 + 2*Ca_se0 + rese
    X_sg = Na_sg0 + K_sg0 - Cl_sg0 - resg
    X_dn = Na_dn0 + K_dn0 - Cl_dn0 + 2*Ca_dn0 - resi
    X_de = Na_de0 + K_de0 - Cl_de0 + 2*Ca_de0 + rese
    X_dg = Na_dg0 + K_dg0 - Cl_dg0 - resg

    cM_sn = 122.01 * 1e3
    cM_se = 287.55 * 1e3
    cM_sg = 121.6 * 1e3
    cM_dn = 122.01 * 1e3
    cM_de = 287.55 * 1e3
    cM_dg = 121.6 * 1e3

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01
    z0 = 1.0

    I_stim = 50e-6 # [uA]

    def dkdt(t,k):

        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k

        my_cell = edNEGmodel(279, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_sn0/V_sn0, Ca_dn0/V_dn0, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)

        dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()

        if t > 8e3 and t < 9e3:
            dKdt_sn += I_stim / my_cell.F
            dKdt_se -= I_stim / my_cell.F

        return dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt 

    def model_jacobian(t,k):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        my_cell = edNEGmodel(279, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_sn0/V_sn0, Ca_dn0/V_dn0, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)
        
        return my_cell.edNEG_jacobian(dense=True)
    
    t_span = (0, 10e3)
    k0 = [Na_sn0, Na_se0, Na_sg0, Na_dn0, Na_de0, Na_dg0, K_sn0, K_se0, K_sg0, K_dn0, K_de0, K_dg0, Cl_sn0, Cl_se0, Cl_sg0, Cl_dn0, Cl_de0, Cl_dg0, Ca_sn0, Ca_se0, Ca_dn0, Ca_de0, n0, h0, s0, c0, q0, z0, V_sn0, V_se0, V_sg0, V_dn0, V_de0, V_dg0]
    sol = solve_ivp(dkdt, t_span, k0, method='Radau', max_step=1e1, jac=model_jacobian)
    
    Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg  = sol.y
    
    test_cell = edNEGmodel(279, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_sn0/V_sn0, Ca_dn0/V_dn0, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)

    q_sn = test_cell.total_charge(np.array([test_cell.Na_sn[-1], test_cell.K_sn[-1], test_cell.Cl_sn[-1], test_cell.Ca_sn[-1], test_cell.X_sn]))
    q_se = test_cell.total_charge(np.array([test_cell.Na_se[-1], test_cell.K_se[-1], test_cell.Cl_se[-1], test_cell.Ca_se[-1], test_cell.X_se]))        
    q_sg = test_cell.total_charge(np.array([test_cell.Na_sg[-1], test_cell.K_sg[-1], test_cell.Cl_sg[-1], 0, test_cell.X_sg]))        
    q_dn = test_cell.total_charge(np.array([test_cell.Na_dn[-1], test_cell.K_dn[-1], test_cell.Cl_dn[-1], test_cell.Ca_dn[-1], test_cell.X_dn]))
    q_de = test_cell.total_charge(np.array([test_cell.Na_de[-1], test_cell.K_de[-1], test_cell.Cl_de[-1], test_cell.Ca_de[-1], test_cell.X_de]))
    q_dg = test_cell.total_charge(np.array([test_cell.Na_dg[-1], test_cell.K_dg[-1], test_cell.Cl_dg[-1], 0, test_cell.X_dg]))

    total_q = abs(q_sn + q_se + q_sg + q_dn + q_de + q_dg)

    assert total_q < EPS
    assert abs(q_sn + q_sg + q_se) < EPS
    assert abs(q_dn + q_dg + q_de) < EPS

def test_volume_conservation():
   
    alpha = 2.0
    
    EPS = 1e-12

    V_sn0 = 1.437e-9             # [cm**3]
    V_se0 = 7.185e-10            # [cm**3]
    V_sg0 = 1.437e-9             # [cm**3]
    V_dn0 = 1.437e-9             # [cm**3]
    V_de0 = 7.185e-10            # [cm**3]
    V_dg0 = 1.437e-9             # [cm**3]
    Na_sn0 = 19*V_sn0 * 1e3
    Na_se0 = 145*V_se0 * 1e3
    Na_sg0 = 14.6*V_sg0 * 1e3
    K_sn0 = 97*V_sn0 * 1e3
    K_se0 = 3.45*V_se0 * 1e3
    K_sg0 = 101*V_sg0 * 1e3
    Cl_sn0 = 6*V_sn0 * 1e3
    Cl_se0 = 138*V_se0 * 1e3
    Cl_sg0 = 6*V_sg0 * 1e3
    Ca_sn0 = 0.01*V_sn0 * 1e3
    Ca_se0 = 1.1*V_se0 * 1e3

    Na_dn0 = 19*V_dn0 * 1e3
    Na_de0 = 145*V_de0 * 1e3
    Na_dg0 = 14.6*V_dg0 * 1e3
    K_dn0 = 97*V_dn0 * 1e3
    K_de0 = 3.45*V_de0 * 1e3
    K_dg0 = 101*V_dg0 * 1e3
    Cl_dn0 = 6*V_dn0 * 1e3
    Cl_de0 = 138*V_de0 * 1e3
    Cl_dg0 = 6*V_dg0 * 1e3
    Ca_dn0 = 0.01*V_dn0 * 1e3
    Ca_de0 = 1.1*V_de0 * 1e3

    resi = -72*3e-2*616e-12/9.648e4 * 1e6
    resg = -83*3e-2*616e-12/9.648e4 * 1e6
    rese =  resi+resg

    X_sn = Na_sn0 + K_sn0 - Cl_sn0 + 2*Ca_sn0 - resi
    X_se = Na_se0 + K_se0 - Cl_se0 + 2*Ca_se0 + rese
    X_sg = Na_sg0 + K_sg0 - Cl_sg0 - resg
    X_dn = Na_dn0 + K_dn0 - Cl_dn0 + 2*Ca_dn0 - resi
    X_de = Na_de0 + K_de0 - Cl_de0 + 2*Ca_de0 + rese
    X_dg = Na_dg0 + K_dg0 - Cl_dg0 - resg

    cM_sn = 122.01 * 1e3
    cM_se = 287.55 * 1e3
    cM_sg = 121.6 * 1e3
    cM_dn = 122.01 * 1e3
    cM_de = 287.55 * 1e3
    cM_dg = 121.6 * 1e3

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01
    z0 = 1.0

    I_stim = 50e-6 # [uA]

    def dkdt(t,k):

        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k

        my_cell = edNEGmodel(279, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_sn0/V_sn0, Ca_dn0/V_dn0, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)

        dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()

        if t > 8e3 and t < 9e3:
            dKdt_sn += I_stim / my_cell.F
            dKdt_se -= I_stim / my_cell.F

        return dNadt_sn, dNadt_se, dNadt_sg, dNadt_dn, dNadt_de, dNadt_dg, dKdt_sn, dKdt_se, dKdt_sg, dKdt_dn, dKdt_de, dKdt_dg, \
            dCldt_sn, dCldt_se, dCldt_sg, dCldt_dn, dCldt_de, dCldt_dg, dCadt_sn, dCadt_se, dCadt_dn, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt 

    def model_jacobian(t,k):
        Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg = k
        my_cell = edNEGmodel(279, Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, X_sn, X_se, X_sg, X_dn, X_de, X_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_sn0/V_sn0, Ca_dn0/V_dn0, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg, cM_sn, cM_se, cM_sg, cM_dn, cM_de, cM_dg)
        
        return my_cell.edNEG_jacobian(dense=True)
    
    t_span = (0, 10e3)
    k0 = [Na_sn0, Na_se0, Na_sg0, Na_dn0, Na_de0, Na_dg0, K_sn0, K_se0, K_sg0, K_dn0, K_de0, K_dg0, Cl_sn0, Cl_se0, Cl_sg0, Cl_dn0, Cl_de0, Cl_dg0, Ca_sn0, Ca_se0, Ca_dn0, Ca_de0, n0, h0, s0, c0, q0, z0, V_sn0, V_se0, V_sg0, V_dn0, V_de0, V_dg0]
    sol = solve_ivp(dkdt, t_span, k0, method='Radau', max_step=1e1, jac=model_jacobian)
    
    Na_sn, Na_se, Na_sg, Na_dn, Na_de, Na_dg, K_sn, K_se, K_sg, K_dn, K_de, K_dg, Cl_sn, Cl_se, Cl_sg, Cl_dn, Cl_de, Cl_dg, Ca_sn, Ca_se, Ca_dn, Ca_de, n, h, s, c, q, z, V_sn, V_se, V_sg, V_dn, V_de, V_dg  = sol.y
   
    V_s_tot0 = V_sn0 + V_se0 + V_sg0
    V_d_tot0 = V_dn0 + V_de0 + V_dg0
    V_s_tot = V_sn[-1] + V_se[-1] + V_sg[-1]
    V_d_tot = V_dn[-1] + V_de[-1] + V_dg[-1]

    assert abs(V_s_tot0 - V_s_tot) < EPS
    assert abs(V_d_tot0 - V_d_tot) < EPS