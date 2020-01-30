import pytest
from scipy.integrate import solve_ivp
from pinsky_rinzel_pump_astrocyte.swella import *

def test_modules():

    alpha = 1.0
    
    V_si = 1437e-18              # [m**3]
    V_se = 718.5e-18             # [m**3]
    V_sg = 1437e-18              # [m**3]
    V_di = 1437e-18              # [m**3]
    V_de = 718.5e-18             # [m**3]
    V_dg = 1437e-18              # [m**3]
    Na_si = 14*V_si
    Na_se = 145*V_se
    Na_sg = 14.6*V_sg
    K_si = 100*V_si
    K_se = 3.45*V_se
    K_sg = 101*V_sg
    Cl_si = 115*V_si
    Cl_se = 138*V_se
    Cl_sg = 6*V_sg
    Ca_si = 0.01*V_si
    Ca_se = 1.1*V_se

    Na_di = 16*V_di
    Na_de = 145*V_de
    Na_dg = 14.6*V_dg
    K_di = 97*V_di
    K_de = 3.45*V_de
    K_dg = 101*V_dg
    Cl_di = 6*V_di
    Cl_de = 138*V_de
    Cl_dg = 6*V_dg
    Ca_di = 0.01*V_di
    Ca_de = 1.1*V_de

    res = -68e-3*3e-2*616e-12/9.648e4
    resi = -72e-3*3e-2*616e-12/9.648e4 
    resg = -83e-3*3e-2*616e-12/9.648e4 
    rese =  resi+resg

    k_res_si = -Na_si - K_si + Cl_si - 2*Ca_si + resi
    k_res_se = -Na_se - K_se + Cl_se - 2*Ca_se - rese
    k_res_sg = -Na_sg - K_sg + Cl_sg + resg
    k_res_di = -Na_di - K_di + Cl_di - 2*Ca_di + resi
    k_res_de = -Na_de - K_de + Cl_de - 2*Ca_de - rese
    k_res_dg = -Na_dg - K_dg + Cl_dg + resg

    c_res_si = 122.01
    c_res_se = 287.55
    c_res_sg = 121.6
    c_res_di = 122.01
    c_res_de = 287.55
    c_res_dg = 121.6

    n = 0.001
    h = 0.999
    s = 0.009
    c = 0.007
    q = 0.01
    z = 1.0

    test_cell = Swella(279, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, K_se/V_se, K_sg/V_sg, K_de/V_de, K_dg/V_dg, Ca_si/V_si, Ca_di/V_di, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg, c_res_si, c_res_se, c_res_sg, c_res_di, c_res_de, c_res_dg)

    assert round(test_cell.nernst_potential(1., 400., 20.), 4) == -0.072

    assert round(test_cell.conductivity_k(test_cell.D_Na, test_cell.Z_Na, 3.2, test_cell.cNa_si, test_cell.cNa_di), 4) == 0.0078

    assert test_cell.total_charge( \
        [test_cell.cNa_si, test_cell.cK_si, test_cell.cCl_si, 0], 0, 1e-1) == -9648
    
    assert test_cell.total_charge( \
        [10, 10, 20, 0], 0, 10) == 0 

    assert test_cell.total_charge( \
        [10, 10, 10, 5], -20, 10) == 0 

def test_charge_conservation():
    """Tests that no charge disappear.
       Tests charge symmetry."""
   
    alpha = 1.0
    
    EPS = 1e-14

    V_si0 = 1437e-18              # [m**3]
    V_se0 = 718.5e-18             # [m**3]
    V_sg0 = 1437e-18              # [m**3]
    V_di0 = 1437e-18              # [m**3]
    V_de0 = 718.5e-18             # [m**3]
    V_dg0 = 1437e-18              # [m**3]
    Na_si0 = 19*V_si0
    Na_se0 = 145*V_se0
    Na_sg0 = 14.6*V_sg0
    K_si0 = 97*V_si0
    K_se0 = 3.45*V_se0
    K_sg0 = 101*V_sg0
    Cl_si0 = 6*V_si0
    Cl_se0 = 138*V_se0
    Cl_sg0 = 6*V_sg0
    Ca_si0 = 0.01*V_si0
    Ca_se0 = 1.1*V_se0

    Na_di0 = 19*V_di0
    Na_de0 = 145*V_de0
    Na_dg0 = 14.6*V_dg0
    K_di0 = 97*V_di0
    K_de0 = 3.45*V_de0
    K_dg0 = 101*V_dg0
    Cl_di0 = 6*V_di0
    Cl_de0 = 138*V_de0
    Cl_dg0 = 6*V_dg0
    Ca_di0 = 0.01*V_di0
    Ca_de0 = 1.1*V_de0

    res = -68e-3*3e-2*616e-12/9.648e4
    resi = -72e-3*3e-2*616e-12/9.648e4 
    resg = -83e-3*3e-2*616e-12/9.648e4 
    rese =  resi+resg

    k_res_si = -Na_si0 - K_si0 + Cl_si0 - 2*Ca_si0 + resi
    k_res_se = -Na_se0 - K_se0 + Cl_se0 - 2*Ca_se0 - rese
    k_res_sg = -Na_sg0 - K_sg0 + Cl_sg0 + resg
    k_res_di = -Na_di0 - K_di0 + Cl_di0 - 2*Ca_di0 + resi
    k_res_de = -Na_de0 - K_de0 + Cl_de0 - 2*Ca_de0 - rese
    k_res_dg = -Na_dg0 - K_dg0 + Cl_dg0 + resg

    c_res_si = 122.01
    c_res_se = 287.55
    c_res_sg = 121.6
    c_res_di = 122.01
    c_res_de = 287.55
    c_res_dg = 121.6

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01
    z0 = 1.0

    I_stim = 50e-12 # [A]

    def dkdt(t,k):

        Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg = k

        my_cell = Swella(279, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_si0/V_si0, Ca_di0/V_di0, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg, c_res_si, c_res_se, c_res_sg, c_res_di, c_res_de, c_res_dg)

        my_cell.G_n = 7e-18
        my_cell.G_g = 7e-18

        dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()

        if t > 10 and t < 20:
            dnKdt_si += I_stim / my_cell.F
            dnKdt_se -= I_stim / my_cell.F

        return dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, \
            dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt 

    t_span = (0, 30)
    k0 = [Na_si0, Na_se0, Na_sg0, Na_di0, Na_de0, Na_dg0, K_si0, K_se0, K_sg0, K_di0, K_de0, K_dg0, Cl_si0, Cl_se0, Cl_sg0, Cl_di0, Cl_de0, Cl_dg0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, n0, h0, s0, c0, q0, z0, V_si0, V_se0, V_sg0, V_di0, V_de0, V_dg0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)
    
    Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg  = sol.y
    
    test_cell = Swella(279, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_si0/V_si0, Ca_di0/V_di0, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg, c_res_si, c_res_se, c_res_sg, c_res_di, c_res_de, c_res_dg)

    q_si = test_cell.total_charge([test_cell.cNa_si[-1], test_cell.cK_si[-1], test_cell.cCl_si[-1], test_cell.cCa_si[-1]], test_cell.ck_res_si, test_cell.V_si)
    q_se = test_cell.total_charge([test_cell.cNa_se[-1], test_cell.cK_se[-1], test_cell.cCl_se[-1], test_cell.cCa_se[-1]], test_cell.ck_res_se, test_cell.V_se)        
    q_sg = test_cell.total_charge([test_cell.cNa_sg[-1], test_cell.cK_sg[-1], test_cell.cCl_sg[-1], 0], test_cell.ck_res_sg, test_cell.V_sg)        
    q_di = test_cell.total_charge([test_cell.cNa_di[-1], test_cell.cK_di[-1], test_cell.cCl_di[-1], test_cell.cCa_di[-1]], test_cell.ck_res_di, test_cell.V_di)
    q_de = test_cell.total_charge([test_cell.cNa_de[-1], test_cell.cK_de[-1], test_cell.cCl_de[-1], test_cell.cCa_de[-1]], test_cell.ck_res_de, test_cell.V_de)
    q_dg = test_cell.total_charge([test_cell.cNa_dg[-1], test_cell.cK_dg[-1], test_cell.cCl_dg[-1], 0], test_cell.ck_res_dg, test_cell.V_dg)

    total_q = abs(q_si + q_se + q_sg + q_di + q_de + q_dg)

    assert total_q < EPS
    assert abs(q_si + q_sg + q_se) < EPS
    assert abs(q_di + q_dg + q_de) < EPS

def test_volume_conservation():
   
    alpha = 1.0
    
    EPS = 1e-14

    V_si0 = 1437e-18              # [m**3]
    V_se0 = 718.5e-18             # [m**3]
    V_sg0 = 1437e-18              # [m**3]
    V_di0 = 1437e-18              # [m**3]
    V_de0 = 718.5e-18             # [m**3]
    V_dg0 = 1437e-18              # [m**3]
    Na_si0 = 19*V_si0
    Na_se0 = 145*V_se0
    Na_sg0 = 14.6*V_sg0
    K_si0 = 97*V_si0
    K_se0 = 3.45*V_se0
    K_sg0 = 101*V_sg0
    Cl_si0 = 6*V_si0
    Cl_se0 = 138*V_se0
    Cl_sg0 = 6*V_sg0
    Ca_si0 = 0.01*V_si0
    Ca_se0 = 1.1*V_se0

    Na_di0 = 19*V_di0
    Na_de0 = 145*V_de0
    Na_dg0 = 14.6*V_dg0
    K_di0 = 97*V_di0
    K_de0 = 3.45*V_de0
    K_dg0 = 101*V_dg0
    Cl_di0 = 6*V_di0
    Cl_de0 = 138*V_de0
    Cl_dg0 = 6*V_dg0
    Ca_di0 = 0.01*V_di0
    Ca_de0 = 1.1*V_de0

    res = -68e-3*3e-2*616e-12/9.648e4
    resi = -72e-3*3e-2*616e-12/9.648e4 
    resg = -83e-3*3e-2*616e-12/9.648e4 
    rese =  resi+resg

    k_res_si = -Na_si0 - K_si0 + Cl_si0 - 2*Ca_si0 + resi
    k_res_se = -Na_se0 - K_se0 + Cl_se0 - 2*Ca_se0 - rese
    k_res_sg = -Na_sg0 - K_sg0 + Cl_sg0 + resg
    k_res_di = -Na_di0 - K_di0 + Cl_di0 - 2*Ca_di0 + resi
    k_res_de = -Na_de0 - K_de0 + Cl_de0 - 2*Ca_de0 - rese
    k_res_dg = -Na_dg0 - K_dg0 + Cl_dg0 + resg

    c_res_si = 122.01
    c_res_se = 287.55
    c_res_sg = 121.6
    c_res_di = 122.01
    c_res_de = 287.55
    c_res_dg = 121.6

    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01
    z0 = 1.0

    I_stim = 50e-12 # [A]

    def dkdt(t,k):

        Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg = k

        my_cell = Swella(279, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, K_se0/V_se0, K_sg0/V_sg0, K_de0/V_de0, K_dg0/V_dg0, Ca_si0/V_si0, Ca_di0/V_di0, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg, c_res_si, c_res_se, c_res_sg, c_res_di, c_res_de, c_res_dg)

        my_cell.G_n = 7e-18
        my_cell.G_g = 7e-18

        dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()
        dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt = my_cell.dVdt()

        if t > 10 and t < 20:
            dnKdt_si += I_stim / my_cell.F
            dnKdt_se -= I_stim / my_cell.F

        return dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, \
            dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt, dVsidt, dVsedt, dVsgdt, dVdidt, dVdedt, dVdgdt 

    t_span = (0, 30)
    k0 = [Na_si0, Na_se0, Na_sg0, Na_di0, Na_de0, Na_dg0, K_si0, K_se0, K_sg0, K_di0, K_de0, K_dg0, Cl_si0, Cl_se0, Cl_sg0, Cl_di0, Cl_de0, Cl_dg0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, n0, h0, s0, c0, q0, z0, V_si0, V_se0, V_sg0, V_di0, V_de0, V_dg0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)
    
    Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z, V_si, V_se, V_sg, V_di, V_de, V_dg  = sol.y
   
    V_s_tot0 = V_si0 + V_se0 + V_sg0
    V_d_tot0 = V_di0 + V_de0 + V_dg0
    V_s_tot = V_si + V_se + V_sg
    V_d_tot = V_di + V_de + V_dg

    assert abs(V_s_tot0 - V_s_tot) < EPS
    assert abs(V_d_tot0 - V_d_tot) < EPS
