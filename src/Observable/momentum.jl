# generate uniform momentum space discretization within a cuboid
function get_momenta(
    rx  :: NTuple{2, Float64}, 
    ry  :: NTuple{2, Float64}, 
    rz  :: NTuple{2, Float64}, 
    num :: NTuple{3, Int64}
    )   :: Matrix{Float64}

    # allocate mesh
    momenta = zeros(Float64, 3, (num[1] + 1) * (num[2] + 1) * (num[3] + 1))

    # fill mesh 
    for nx in 0 : num[1]
        for ny in 0 : num[2]
            for nz in 0 : num[3]
                # compute kx
                if num[1] > 0
                    momenta[1, nz + 1 + (num[3] + 1) * ny + (num[3] + 1) * (num[2] + 1) * nx] = rx[1] + nx * (rx[2] - rx[1]) / num[1]
                end

                # compute ky
                if num[2] > 0
                    momenta[2, nz + 1 + (num[3] + 1) * ny + (num[3] + 1) * (num[2] + 1) * nx] = ry[1] + ny * (ry[2] - ry[1]) / num[2]
                end 

                # compute kz
                if num[3] > 0
                    momenta[3, nz + 1 + (num[3] + 1) * ny + (num[3] + 1) * (num[2] + 1) * nx] = rz[1] + nz * (rz[2] - rz[1]) / num[3]
                end
            end 
        end 
    end

    return momenta
end

# compute structure factor of given real space correlations
function compute_structure_factor(
    χ :: Vector{Float64},
    k :: Matrix{Float64},
    l :: lattice,
    r :: reduced_lattice
    ) :: Vector{Float64}

    # allocate structure factor
    s = zeros(Float64, size(k, 2))

    for a in eachindex(s)
        q = k[:, a]

        for b in eachindex(l.uc.basis)
            # compute all contributions from a reference site in the unitcell
            vec = l.uc.basis[b]
            int = get_site(vec, l)

            for c in eachindex(l.sites)
                # consider only those sites in range of reference site
                index = r.project[int, c]

                if index > 0
                    s[a] += cos(dot(q, vec .- l.sites[c].vec)) * χ[index]
                end 
            end 
        end 
    end 

    s ./= length(l.uc.basis)

    return s 
end