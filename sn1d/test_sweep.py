from sweep import ldSolve


def test_vacuum():
    [psi_l, psi_r] = ldSolve(   mu=-0.5, 
            dx=0.1,
            psi_in=1,
            xs_t=0,
            xs_s=0,
            xs_f=0,
            nu_f=0,
            q =0,
            phi =(0, 0))
    assert psi_l == 1 and psi_r == 1

def test_mirror():
    [psi_l_neg, psi_r_neg] = ldSolve(   mu=-0.5, 
            dx=0.1,
            psi_in=1,
            xs_t=0.23,
            xs_s=0.11,
            xs_f=0.04,
            nu_f=2,
            q =1,
            phi =(1, 1.2))
    [psi_l_pos, psi_r_pos] = ldSolve(   mu=0.5, 
            dx=0.1,
            psi_in=1,
            xs_t=0.23,
            xs_s=0.11,
            xs_f=0.04,
            nu_f=2,
            q =1,
            phi =(1.2, 1))
    assert (psi_l_pos == psi_r_neg) and (psi_r_pos == psi_l_neg)