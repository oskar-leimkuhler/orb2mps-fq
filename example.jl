using PyCall
include("./libfqmps.jl") 
pushfirst!(pyimport("sys")."path", ".") # Get current directory path

# Load pyscf_funcs
pf = pyimport("pyscf_funcs")
scipy = pyimport("scipy")

# Define molecule (H2O)
atom = """
        O  0.000000   0.000000   0.000000
        H  0.758602   0.000000   0.504284
        H -0.758602   0.000000   0.504284
       """
basis = "sto-3g"
spin = 0

# Run RHF in pyscf and get data
a_vec, l_vec, gamma_vec, C_prim, norm_vec, G, nmo = pf.run_pyscf(atom, basis, spin)

println("Number of MOs: $(nmo)")
 
# MPS truncation parameters
E = 40 # Planewave energy cutoff (Hartree)
# E = 400 # Planewave energy cutoff (Hartree)
# A relatively high kinetic energy is required to capture some of the localized atomic
# orbitals. If the final state vector fidelity is lower than expected, make sure
# that a sufficiently high cutoff has been used.

L = 20 # Length of simulation cube (Bohr)
# For accurate results this should be much larger than the spatial extent of the molecule.
# This is because the calculations implicitly assume that the atomic orbitals vanish
# outside of the simulation cube.

# Set truncation thresholds
ds_trunc = 1e-20
# This number controls the truncation threshold for a bunch of intermediate steps.
# Making it smaller increases the accuracy of the calculation and one should
# check convergence. Making it smaller can also lead to longer runtimes.

final_trunc = 1e-3
# This number will perform one final truncation of the matrix product state. Setting
# it too small will increase the bond dimension dramatically without contributing
# much to the fidelity.

# Choose the MO number
# mo_number = 1 # Core orbital, will require high kinetic energy cutoff.
mo_number = 5 # HOMO, lower kinetic energy cutoff should suffice.

# Choose MO number --> column of C_prim
c_col = C_prim[:,mo_number]

println("Calculating MPS representation for MO #$(mo_number)")

# Flag that decides if we should print a bunch of intermediate information.
verbose = true

# Get MPS representation of selected Gaussian basis MO in the planewave basis,
# number of planewaves, number of qubits per dimension, and the statevector fidelity
MPS_mo, N_pw, num_qubits_dim, sv_fidelity = StateVecMPS(a_vec, l_vec, gamma_vec,
    c_col, L=L, E=E, verbose=verbose, ds_trunc=ds_trunc, final_trunc=final_trunc);

println("Num. plane waves: $(N_pw)")
println("Qubits per dimension: $(num_qubits_dim)")
println("State vector fidelity: $(sv_fidelity)\n")
println("Bond dimensions: $(linkdims(MPS_mo))")