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
    get_path(
        nodes :: Vector{Vector{Float64}},
        nums  :: Vector{Int64}
        )     :: Tuple{Vector{Float64}, Matrix{Float64}}

Generate a discrete path in momentum space linearly connecting the given nodes (passed via their cartesian coordinates (kx, ky, kz)).
nums[i] is the desired number of points between node[i] and node[i + 1], including node[i] and excluding node[i + 1].
"""
function get_path(
    nodes :: Vector{Vector{Float64}},
    nums  :: Vector{Int64}
    )     :: Tuple{Vector{Float64}, Matrix{Float64}}

    # sanity check: need some number of points for each path increment between two nodes
    @assert length(nums) == length(nodes) - 1 "For N nodes 'nums' must be of length N - 1."

    # allocate output arrays
    num  = sum(nums) + 1
    dist = zeros(num)
    path = zeros(3, num)

    # iterate over path increments between nodes
    idx = 0

    for i in 1 : length(nums)
        dif   = nodes[i + 1] .- nodes[i]
        step  = dif ./ nums[i]
        width = norm(step)

        # fill path from node i to node i + 1 (excluding node i + 1)
        for j in 1 : nums[i]
            dist[idx + j + 1] = dist[idx + 1] + width * j
            path[:, idx + j] .= nodes[i] .+ step .* (j - 1)
        end

        idx += nums[i]
    end

    # set last point in the path to be the last node
    path[:, end] = nodes[end]

    return dist, path
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
