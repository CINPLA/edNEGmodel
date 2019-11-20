import pytest
from pinsky_rinzel_pump_astrocyte.buffy import *
from pinsky_rinzel_pump_astrocyte.somatic_injection_current import *

def test_modules():

    test_cell = (279.3, 14., 145., 16., 145., 100., 3., 100., 3., 115., 148., 115., 148., 1, 1, 1, 1, 0, 0, 0, 0, 2)

    test_cell = Buffy(279.3, 14., 145., 14., 16., 145., 16., 100., 3., 100., 100., 3., 100., 115., 148., 115., 115., 148., 115., \
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0)

    assert round(test_cell.nernst_potential(1., 400., 20.), 4) == -0.0721

    assert round(test_cell.conductivity_k(test_cell.D_Na, test_cell.Z_Na, 3.2, test_cell.Na_si, test_cell.Na_di), 4) == 0.0078

    assert test_cell.total_charge( \
        [test_cell.Na_si, test_cell.K_si, test_cell.Cl_si, 0], 0, 1e-1) == -9648
    
    assert test_cell.total_charge( \
        [10, 10, 20, 0], 0, 10) == 0 

    assert test_cell.total_charge( \
        [10, 10, 10, 5], -20, 10) == 0 

def test_charge_conservation():
    """Tests that no charge disappear.
       Tests charge symmetry."""
   
    alpha = 1.0
    
    EPS = 1e-14

    Na_si0 = 18.
    Na_se0 = 139.
    Na_sg0 = 18.
    K_si0 = 99.
    K_se0 = 5.
    K_sg0 = 99.
    Cl_si0 = 7.
    Cl_se0 = 131.
    Cl_sg0 = 7.
    Ca_si0 = 0.01
    Ca_se0 = 1.1

    Na_di0 = 20.
    Na_de0 = 141.
    Na_dg0 = 18.
    K_di0 = 96.
    K_de0 = 4.
    K_dg0 = 99.
    Cl_di0 = 7.
    Cl_de0 = 131.
    Cl_dg0 = 7.
    Ca_di0 = 0.01
    Ca_de0 = 1.1

    k_res_si = Cl_si0 - Na_si0 - K_si0 - 2*Ca_si0
    k_res_se = Cl_se0 - Na_se0 - K_se0 - 2*Ca_se0
    k_res_sg = Cl_sg0 - Na_sg0 - K_sg0
    k_res_di = Cl_di0 - Na_di0 - K_di0 - 2*Ca_di0
    k_res_de = Cl_de0 - Na_de0 - K_de0 - 2*Ca_de0
    k_res_dg = Cl_dg0 - Na_dg0 - K_dg0
    
    n0 = 0.001
    h0 = 0.999
    s0 = 0.009
    c0 = 0.007
    q0 = 0.01
    z0 = 1.0

    I_stim = 10e-12 # [A]

    def dkdt(t,k):

        Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z = k

        my_cell = Buffy(279, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

        dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de = my_cell.dkdt()
        dndt, dhdt, dsdt, dcdt, dqdt, dzdt = my_cell.dmdt()

        if t > 3 and t < 3.5:
            dKdt_si, dKdt_se = somatic_injection_current(my_cell, dKdt_si, dKdt_se, my_cell.Z_K, I_stim)

        return dNadt_si, dNadt_se, dNadt_sg, dNadt_di, dNadt_de, dNadt_dg, dKdt_si, dKdt_se, dKdt_sg, dKdt_di, dKdt_de, dKdt_dg, \
            dCldt_si, dCldt_se, dCldt_sg, dCldt_di, dCldt_de, dCldt_dg, dCadt_si, dCadt_se, dCadt_di, dCadt_de, \
            dndt, dhdt, dsdt, dcdt, dqdt, dzdt

    t_span = (0, 10)
    k0 = [Na_si0, Na_se0, Na_sg0, Na_di0, Na_de0, Na_dg0, K_si0, K_se0, K_sg0, K_di0, K_de0, K_dg0, Cl_si0, Cl_se0, Cl_sg0, Cl_di0, Cl_de0, Cl_dg0, Ca_si0, Ca_se0, Ca_di0, Ca_de0, n0, h0, s0, c0, q0, z0]
    sol = solve_ivp(dkdt, t_span, k0, max_step=1e-4)
    
    Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, Ca_si, Ca_se, Ca_di, Ca_de, n, h, s, c, q, z  = sol.y
    
    test_cell = Buffy(279, Na_si, Na_se, Na_sg, Na_di, Na_de, Na_dg, K_si, K_se, K_sg, K_di, K_de, K_dg, Cl_si, Cl_se, Cl_sg, Cl_di, Cl_de, Cl_dg, \
        Ca_si, Ca_se, Ca_di, Ca_de, k_res_si, k_res_se, k_res_sg, k_res_di, k_res_de, k_res_dg, alpha, Ca_si0, Ca_di0, n, h, s, c, q, z)

    q_si = test_cell.total_charge([test_cell.Na_si[-1], test_cell.K_si[-1], test_cell.Cl_si[-1], test_cell.Ca_si[-1]], test_cell.k_res_si, test_cell.V_si)
    q_se = test_cell.total_charge([test_cell.Na_se[-1], test_cell.K_se[-1], test_cell.Cl_se[-1], test_cell.Ca_se[-1]], test_cell.k_res_se, test_cell.V_se)        
    q_sg = test_cell.total_charge([test_cell.Na_sg[-1], test_cell.K_sg[-1], test_cell.Cl_sg[-1], 0], test_cell.k_res_sg, test_cell.V_sg)        
    q_di = test_cell.total_charge([test_cell.Na_di[-1], test_cell.K_di[-1], test_cell.Cl_di[-1], test_cell.Ca_di[-1]], test_cell.k_res_di, test_cell.V_di)
    q_de = test_cell.total_charge([test_cell.Na_de[-1], test_cell.K_de[-1], test_cell.Cl_de[-1], test_cell.Ca_de[-1]], test_cell.k_res_de, test_cell.V_de)
    q_dg = test_cell.total_charge([test_cell.Na_dg[-1], test_cell.K_dg[-1], test_cell.Cl_dg[-1], 0], test_cell.k_res_dg, test_cell.V_dg)

    total_q = abs(q_si + q_se + q_sg + q_di + q_de + q_dg)

    assert total_q < EPS
    assert abs(q_si + q_sg + q_se) < EPS
    assert abs(q_di + q_dg + q_de) < EPS
