using ITensors, ITensorMPS
using LinearAlgebra
using Random
using Polynomials
using SpecialPolynomials


# Fidelity / trace distance functions:
function Infidelity(fid)
    return abs(1.0-fid)
end

function NLI(fid)
    return -log10(Infidelity(fid))
end

function TraceDist(fid)
    return sqrt(abs(1.0-fid^2))
end


# Simple plane wave function:
function planewave(x_vals, k, L)
    
    ck = 1/sqrt(L)
    
    return [ck*exp(-1im*k*x) for x in x_vals]
    
end


# Hermite polynomial expansion coefficients for Gaussian-weighted x^l:
function hn_coeffs(l)
    
    hn_vec = []
    
    for n=0:l
        
        if iseven(l-n)
        
            num = 2^(n/2)
            
            denom = factorial(Int((l-n)/2)) * sqrt(factorial(n))
            
            push!(hn_vec, num/denom)
            
        else
            
            push!(hn_vec, 0.0)
            
        end
        
    end
    
    normalize!(hn_vec)
    
    return hn_vec
    
end


# Computes a Gaussian-planewave overlap integral in one dimension:
function ovlp_1d(l, gamma, a, k, L)
    
    phase = exp(1im*k*a)
    
    pref = 2^(1/4)*pi^(1/2)/(gamma^(1/4)*L^(1/2))
    
    gfac = exp(-k^2/(4*gamma))
    
    u = variable(Polynomial{Rational{Int}})
    pols = [Polynomials.basis(Hermite, n)(u) for n in 0:l]
    
    hn = hn_coeffs(l)
    cn = [1.0/sqrt(2^n * factorial(n) * sqrt(pi)) for n=0:l]
    
    psum = sum([1im^n*hn[n+1]*cn[n+1]*pols[n+1] for n in 0:l])
    
    pfac = psum(k/sqrt(2*gamma))
    
    return phase*pref*pfac*gfac

end


# Combine two MPSs into a new MPS with connective bond dim. = 1 (pure product)
function CombineMPS(M1, M2; newsites=nothing)
    sites1 = siteinds(M1)
    sites2 = siteinds(M2)
    sites = vcat(sites1, sites2)
    
    if newsites == nothing
        newsites = [Index(dim(sites[n]), tags="Site,n=$(n)") for n=1:length(sites)]
    end
    
    M3 = MPS(sites)
    
    q = length(sites1)
    link = Index(1, "Link,n=$(q)")
    
    for p=1:q-1
        M3[p] = M1[p]
        replaceind!(M3[p], sites1[p], newsites[p])
    end
    
    # Sites q, q+1:
    inds_q = vcat([id for id in inds(M1[q])], [link])
    T_q = ITensor(inds_q)
    for i=1:dim(inds_q[1]), j=1:dim(inds_q[2])
        T_q[inds_q[1]=>i,inds_q[2]=>j,inds_q[3]=>1] = M1[q][inds_q[1]=>i,inds_q[2]=>j]
    end
    M3[q] = T_q
    replaceind!(M3[q], sites1[q], newsites[q])
    
    inds_qp1 = vcat([i for i in inds(M2[1])], [link])
    T_qp1 = ITensor(inds_qp1)
    for i=1:dim(inds_qp1[1]), j=1:dim(inds_qp1[2])
        T_qp1[inds_qp1[1]=>i,inds_qp1[2]=>j,inds_qp1[3]=>1] = M2[1][inds_qp1[1]=>i,inds_qp1[2]=>j]
    end
    M3[q+1] = T_qp1
    replaceind!(M3[q+1], sites2[1], newsites[q+1])
    
    settags!(M3[q+1], "Site,n=$(q+1)", tags="Site,n=1")
    settags!(M3[q+1], "Link,n=$(q+1)", tags="Link,n=1")
    
    for p=2:length(sites2)
        M3[p+q] = M2[p]
        settags!(M3[p+q], "Link,n=$(p+q-1)", tags="Link,n=$(p-1)")
        settags!(M3[p+q], "Link,n=$(p+q)", tags="Link,n=$(p)")
        replaceind!(M3[p+q], sites2[p], newsites[p+q])
    end
        
    return M3
    
end


# Construct a state vector MPS from primitive Gaussian information:
function StateVecMPS(
        a_list,
        l_list,
        gamma_list,
        c_col;
        L=20.0,
        E=20.0,
        # Practical truncation threshold \\
        # for the direct summation:
        ds_trunc=1e-16,
        # Final truncation threshold:
        final_trunc=1e-12,
        verbose=false
    )
    # Maximal momentum quantum number for the plane waves in each dimension:
    pmax = Int(floor(L/pi*sqrt(E/2)))
    
    # Number of qubits per dimension and array offset to pad with zeros:
    nqb = Int(ceil(log2(2*pmax+1)))
    pshift = 2^(nqb-1)
    
    sites = siteinds(2, nqb)

    full_sites = siteinds(2, 3*nqb)
    
    # Initialize the molecular orbital MPS to be constructed by direct sum \\
    # over Gaussian primitive MPSs:
    Mmo = 0.0 * MPS(full_sites, [1 for n=1:(3*nqb)])
    
    verbose && println("\nComputing state vector MPS:")
    
    M_list = []

    for g = 1:length(gamma_list)

        ovlp_xyz = []

        for i=1:3

            gamma = gamma_list[g]
            l = l_list[g][i]
            a = a_list[g,i]

            ovlp_coeffs = [ovlp_1d(l,gamma,a+L/2,2*pi*p/L,L) for p=-pmax:pmax]

            ovlp_pad = zeros(ComplexF64, 2^nqb)

            ovlp_pad[(pshift-pmax):(pshift+pmax)] = ovlp_coeffs

            push!(ovlp_xyz, ovlp_pad)

        end

        sites = siteinds(2, nqb)

        @disable_warn_order Mx = MPS(ovlp_xyz[1], sites)
        @disable_warn_order My = MPS(ovlp_xyz[2], sites)
        @disable_warn_order Mz = MPS(ovlp_xyz[3], sites)

        M = CombineMPS(Mx, CombineMPS(My, Mz), newsites=full_sites)
        ITensors.truncate!(M, cutoff=ds_trunc)

	push!(M_list, M)
        
        Mmo += c_col[g]*M
        ITensors.truncate!(Mmo, cutoff=ds_trunc)
        
        if verbose
            println("Progress: $(g)/$(length(gamma_list))")
            println("Mx fidelity: $(norm(Mx))")
            println("x vec norm: $(norm(ovlp_xyz[1]))\n")
            println("My fidelity: $(norm(My))")
            println("y vec norm: $(norm(ovlp_xyz[2]))\n")
            println("Mz fidelity: $(norm(Mz))")
            println("z vec norm: $(norm(ovlp_xyz[3]))\n")
            println("M fidelity: $(norm(M))\n")
            println("ls = $(l_list[g,:]), gamma = $(gamma_list[g]), as = $(a_list[g,:]), pmax = $(pmax), L = $(L)")
            println("c_col[g] = $(c_col[g])")
            println("---------------------------------------------------------------")
            flush(stdout)
        end
        
    end

    ITensors.truncate!(Mmo, cutoff=final_trunc)
    
    N_pw = (2*pmax+1)^3

    verbose && println("\nDone!\n")
    
    verbose && println("Num. plane waves: $(N_pw)")
    verbose && println("Qubits per dimension: $(nqb)")
    verbose && println("State vector fidelity: $(norm(Mmo))\n")
    
    return Mmo, N_pw, nqb, norm(Mmo), M_list
    
end


# Convert a plane-wave MPS representation to a real-space vector:
function MPS2RealSpace(M, L, xmax; L_view=L, verbose=false)
    
    x_pts = ((collect(0:xmax) .- floor(xmax/2)) .* (L_view/xmax)) .+ L/2
    
    nqb = Int(length(M)/3)
    pshift = 2^(nqb-1)
    
    sites = siteinds(M)
    
    # Initialize the MO real-space vector:
    mo_vec = zeros((xmax+1)^3)
    
    if verbose
        println("\nComputing real-space vector:")
        prog = 0
    end
    
    for px=0:2^nqb-1, py=0:2^nqb-1, pz=0:2^nqb-1
        
        # Convert p-values to binary:
        bx = (digits(px, base=2, pad=nqb)) .+ 1
        by = (digits(py, base=2, pad=nqb)) .+ 1
        bz = (digits(pz, base=2, pad=nqb)) .+ 1
        bvec = reduce(vcat, [bx,by,bz])
        
        T = ITensor(1.)
        for j=1:length(M)
          T *= (M[j]*state(sites[j],bvec[j]))
        end
        M_coeff = scalar(T)
        
        # Convert p-values to k-values:
        kx = 2*pi*(px-pshift)/L
        ky = 2*pi*(py-pshift)/L
        kz = 2*pi*(pz-pshift)/L
        
        phi_x = planewave(x_pts, kx, L)
        phi_y = planewave(x_pts, ky, L)
        phi_z = planewave(x_pts, kz, L)
        
        mo_vec += M_coeff*kron(phi_x, kron(phi_y, phi_z))
        
        if verbose
            nprog = Int(floor((2^(2*nqb)*px+2^(nqb)*py+pz+1) / (2^(3*nqb)) * 100))
            if nprog > prog
                print("Progress: $(nprog)%  \r")
                flush(stdout)
                prog = nprog
            end
        end
        
    end
    
    verbose && println("\nDone!\n")
    
    return x_pts, mo_vec
    
end
