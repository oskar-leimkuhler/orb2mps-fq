from pyscf import gto, scf
import scipy
import numpy as np


def loop_cart(l):
    """
    Number of unique angular momentum combinations for
    cartesian combinations equals
    the triangular numbers shifted by one
    (l+1)*(l+2)/2

    adapted from PySCF source code for the gto module
    https://github.com/pyscf/pyscf/

    inputs:
    l = angular momentum
    0,1,2,3... -> s,p,d,f,...

    """
    ang_list = []
    for ix in reversed(range(l+1)):
        for iy in reversed(range(l-ix+1)):
            iz = l - ix - iy
            ang_list.append((ix, iy, iz))

    return ang_list


def unpack_basis_direct(pmol):
    a_vec = []
    l_vec = []
    gamma_vec = []
    coeff_vec = []
    
    for bs in pmol._bas:
        
        coords = pmol._atom[bs[0]][1] # Atomic coordinates
        ang_list = loop_cart(bs[1]) # Angular momenta
        expo = pmol._env[bs[5]] # Exponent
        coeff = pmol._env[bs[6]] # Coefficient
        
        for momentum in ang_list:
            a_vec.append(coords)
            l_vec.append(np.array(momentum))
            gamma_vec.append(expo)
            coeff_vec.append(coeff)
    
    return a_vec, l_vec, gamma_vec, coeff_vec
    

def generate_pyscf_mol(atom, basis, spin):
    """
    Wrapper for a routine to run an RHF calculation in
    PySCF and get MO coefficients

    * Requires Bohr units input (TODO: Allow both)

    returns:
        mol: PySCF molecular data object
        C: Canonical MO coefficients from RHF (N by N)
        occ: occupation number list (length N)

    """

    # Build PySCF mol object

    # Center the XYZ coordinates in the box and convert from Angstrom to Bohr:
    mol0 = gto.Mole()
    mol0.unit = 'A'
    mol0.build(atom=atom, basis=basis, spin=spin, cart=True)
    #atom0 = mol0._atom
    atom1 = mol0._atom

    # Convert from Angstrom to Bohr:
    for A in range(len(atom1)):
        for ax in range(3):
            atom1[A][1][ax] *= 1.88973

    #atom1 = center_box(atom0, L)

    mol = gto.Mole()
    mol.unit = 'B'
    mol.symmetry = False
    #mol.charge=1
    mol.build(atom=atom1, basis=basis, spin=spin, cart=True)
    
    print("Running SCF")
    # Run RHF optimization in uncontracted basis:
    mf = scf.RHF(mol)
    mf.kernel()

    print("Restarting SCF with second order method")
    # This will help the trickier cases converge.
    mo_init = mf.mo_coeff
    mocc_init = mf.mo_occ

    mf = scf.RHF(mol).newton()
    mf.kernel(mo_init, mocc_init)
    print("SCF complete")

    # Get RHF canonical molecular
    # orbital matrix C, and occupation number
    C   = mf.mo_coeff
    occ = mf.mo_occ
    
    return mol, C, occ


def run_pyscf(atom, basis, spin):
    
    # Run PySCF to get the basis info and C matrix:
    mol, C, occ = generate_pyscf_mol(atom, basis, spin)
    print(f"HOMO index: {np.where(occ==0)[0][0]}") # One-indexed for use with Julia...

    # Unpack the primitive Gaussian basis:
    pmol, ctr_coeff = mol.to_uncontracted_cartesian_basis() 
    a_vec, l_vec, gamma_vec, coeff_vec = unpack_basis_direct(pmol)
    S_prim = pmol.intor('int1e_ovlp')

    # Unpack the C matrix:
    ctr_mat = scipy.linalg.block_diag(*ctr_coeff)
    C_prim = ctr_mat @ C

    # Modify the C_prim matrix to work with normalized primitive Gaussians:
    norm_factors = np.sqrt(np.diagonal(S_prim))
    
    for i, norm_factor in enumerate(norm_factors):
        C_prim[i,:] *= norm_factor

    G, nmo = C_prim.shape

    return a_vec, l_vec, gamma_vec, C_prim, norm_factors, G, nmo, S_prim