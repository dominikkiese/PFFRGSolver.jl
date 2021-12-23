"""
    get_momenta(
        rx  :: NTuple{2, Float64},
        ry  :: NTuple{2, Float64},
        rz  :: NTuple{2, Float64},
        num :: NTuple{3, Int64}
        )   :: Matrix{Float64}

Generate a uniform momentum space discretization within a cuboid. rx, ry and rz are the respective cartesian boundaries.
num = (num_x, num_y, num_z) contains the desired number of steps (spacing h = (r[2] - r[1]) / num) along the respective axis.
Returns a matrix k, with k[:, n] being the n-th momentum vector.
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
                # compute linear index 
                idx = nz + 1 + (num[3] + 1) * ny + (num[3] + 1) * (num[2] + 1) * nx

                # compute kx
                if num[1] > 0
                    momenta[1, idx] = rx[1] + nx * (rx[2] - rx[1]) / num[1]
                end

                # compute ky
                if num[2] > 0
                    momenta[2, idx] = ry[1] + ny * (ry[2] - ry[1]) / num[2]
                end

                # compute kz
                if num[3] > 0
                    momenta[3, idx] = rz[1] + nz * (rz[2] - rz[1]) / num[3]
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

Generate a discrete path in momentum space linearly connecting the given nodes (passed via their cartesian coordinates [kx, ky, kz]).
nums[i] is the desired number of points between node[i] and node[i + 1], including node[i] and excluding node[i + 1].
Returns a tuple (l, k) where k[:, n] is the n-th momentum vector and l[n] is the distance to node[1] along the generated path.
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
            dist[idx + j + 1]  = dist[idx + 1] + width * j
            path[:, idx + j]  .= nodes[i] .+ step .* (j - 1)
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
        l :: Lattice,
        r :: Reduced_lattice
        ) :: Vector{Float64}

Compute the structure factor for given real space correlations χ on irreducible lattice sites.
k[:, n] is the n-th discrete momentum vector.
Return structure factor s, where s[n] is the value for the n-th momentum.
"""
function compute_structure_factor(
    χ :: Vector{Float64},
    k :: Matrix{Float64},
    l :: Lattice,
    r :: Reduced_lattice
    ) :: Vector{Float64}

    # allocate structure factor
    s = zeros(Float64, size(k, 2))

    # compute all contributions from reference sites in the unitcell
    for b in eachindex(l.uc.basis)
        vec = l.uc.basis[b]
        int = get_site(vec, l)

        # compute structure factor for all momenta
        Threads.@threads for n in eachindex(s)
            @inbounds @fastmath for i in eachindex(l.sites)
                # consider only those sites in range of reference site
                index = r.project[int, i]

                if index > 0
                    val   = k[1, n] * (vec[1] - l.sites[i].vec[1])
                    val  += k[2, n] * (vec[2] - l.sites[i].vec[2])
                    val  += k[3, n] * (vec[3] - l.sites[i].vec[3])
                    s[n] += cos(val) * χ[index]
                end
            end
        end
    end

    s ./= length(l.uc.basis)

    return s
end