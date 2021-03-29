"""
    get_momenta(
        rx  :: NTuple{2, Float64},
        ry  :: NTuple{2, Float64},
        rz  :: NTuple{2, Float64},
        num :: NTuple{3, Int64}
        )   :: Matrix{Float64}

Generate a uniform momentum space discretization within a cuboid.
rx, ry and rz are the respective cartesian boundaries.
num (i.e num = num_x, num_y, num_z) contains the desired number of points along the respective axis.
"""
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

"""
    get_momentumpath(
        vertices :: Vector{Vector{Float64}},
        nums     :: Vector{Int64}
        )        :: Matrix{Float64}

Generate a path in momentum space linearly connecting the given vertices.
vertices contains 3 dimensional coordinates in momentum space (kx, ky, kz)
nums[i] is the desired number of points between vertex[i] and vertex[i+1], including vertex[i] and excluding vertex[i+1].
"""
function get_momentumpath(
    vertices :: Vector{Vector{Float64}},
    nums     :: Vector{Int64}
    )        :: Matrix{Float64}

    #Need number of points for each path increment between to vertices
    @assert length(nums) == length(vertices) - 1 "For N vertices 'nums' must be of length N-1"

    #allocate path in momentum space
    momentumpath = zeros(3, sum(nums)+1)

    #Iterate over path increments between vertices
    startidx = 0
    for i in 1:length(nums)

        #Compute stepsize on path between vertex i and i+1 (excluding vertex i+1)
        dif = vertices[i+1]-vertices[i]
        path_length = norm(dif)
        step = dif/(nums[i])

        #Fill path from vertex i to vertex i+1 (excluding vertex i+1)
        for j in 1:nums[i]
            momentumpath[:, startidx + j] .= vertices[i] .+ step .* (j-1)
        end
        startidx += nums[i]
    end

    #Set last point in the path to be the last vertex
    momentumpath[:, end] = vertices[end]

    return momentumpath
end

"""
    compute_structure_factor(
        χ :: Vector{Float64},
        k :: Matrix{Float64},
        l :: lattice,
        r :: reduced_lattice
        ) :: Vector{Float64}

Compute the static structure factor for given real space correlations χ on irreducible lattice sites.
The momentum space discretization should be formatted such that k[:, n] is the n-th momentum.
"""
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
