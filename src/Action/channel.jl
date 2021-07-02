"""
    Channel 

Struct containing asymptotic kernels for a channel.
* `q1   :: Matrix{Float64}`   : kernel with both fermionic frequencies -> Inf
* `q2_1 :: Array{Float64, 3}` : kernel with second fermionic frequency -> Inf
* `q2_2 :: Array{Float64, 3}` : kernel with first fermionic frequency -> Inf
* `q3   :: Array{Float64, 4}` : full channel
"""
struct Channel
    q1   :: Matrix{Float64}
    q2_1 :: Array{Float64, 3}
    q2_2 :: Array{Float64, 3} 
    q3   :: Array{Float64, 4}
end

# generate channel dummy
function get_channel_empty(
    r :: Reduced_lattice,
    m :: Mesh    
    ) :: Channel

    num_sites = length(r.sites)

    # init kernels
    q1   = zeros(Float64, num_sites, m.num_Ω)
    q2_1 = zeros(Float64, num_sites, m.num_Ω, m.num_ν)
    q2_2 = zeros(Float64, num_sites, m.num_Ω, m.num_ν)
    q3   = zeros(Float64, num_sites, m.num_Ω, m.num_ν, m.num_ν)

    # build channel 
    ch = Channel(q1, q2_1, q2_2, q3)

    return ch 
end





# get interpolated value of q1
function get_q1(
    site :: Int64, 
    p    :: Param,
    ch   :: Channel
    )    :: Float64 

    val  = p.lower_weight * ch.q1[site, p.lower_index] 
    val += p.upper_weight * ch.q1[site, p.upper_index] 

    return val 
end

# get interpolated value of q1 on all lattice sites 
function get_q1_avx!(
    r        :: Reduced_lattice,
    p        :: Param,
    ch       :: Channel,
    temp     :: SubArray{Float64, 1, Array{Float64, 3}},
    exchange :: Bool,
    sgn      :: Float64
    )        :: Nothing 

    # deref channel
    q1 = ch.q1 

    # deref param
    lower_index  = p.lower_index
    upper_index  = p.upper_index 
    lower_weight = p.lower_weight
    upper_weight = p.upper_weight

    # check for site exchange
    if exchange
        indices = r.exchange

        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight * q1[indices[i], lower_index]
            val     += upper_weight * q1[indices[i], upper_index]
            temp[i] += sgn * val
        end 
    else 
        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight * q1[i, lower_index]
            val     += upper_weight * q1[i, upper_index]
            temp[i] += sgn * val
        end 
    end

    return nothing 
end

# get interpolated value of q2_1
function get_q2_1(
    site :: Int64, 
    p1   :: Param, 
    p2   :: Param,
    ch   :: Channel
    )    :: Float64 

    val  = p1.lower_weight * p2.lower_weight * ch.q2_1[site, p1.lower_index, p2.lower_index] 
    val += p1.upper_weight * p2.lower_weight * ch.q2_1[site, p1.upper_index, p2.lower_index]
    val += p1.lower_weight * p2.upper_weight * ch.q2_1[site, p1.lower_index, p2.upper_index] 
    val += p1.upper_weight * p2.upper_weight * ch.q2_1[site, p1.upper_index, p2.upper_index]

    return val 
end

# get interpolated value of q2_1 on all lattice sites 
function get_q2_1_avx!(
    r        :: Reduced_lattice,
    p1       :: Param,
    p2       :: Param,
    ch       :: Channel,
    temp     :: SubArray{Float64, 1, Array{Float64, 3}},
    exchange :: Bool,
    sgn      :: Float64
    )        :: Nothing 

    # deref channel 
    q2_1 = ch.q2_1 

    # deref param 
    lower_index1  = p1.lower_index
    upper_index1  = p1.upper_index 
    lower_weight1 = p1.lower_weight
    upper_weight1 = p1.upper_weight

    lower_index2  = p2.lower_index
    upper_index2  = p2.upper_index 
    lower_weight2 = p2.lower_weight
    upper_weight2 = p2.upper_weight

    # check for site exchange
    if exchange
        indices = r.exchange

        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight1 * lower_weight2 * q2_1[indices[i], lower_index1, lower_index2]  
            val     += upper_weight1 * lower_weight2 * q2_1[indices[i], upper_index1, lower_index2] 
            val     += lower_weight1 * upper_weight2 * q2_1[indices[i], lower_index1, upper_index2]  
            val     += upper_weight1 * upper_weight2 * q2_1[indices[i], upper_index1, upper_index2] 
            temp[i] += sgn * val
        end 
    else 
        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight1 * lower_weight2 * q2_1[i, lower_index1, lower_index2]  
            val     += upper_weight1 * lower_weight2 * q2_1[i, upper_index1, lower_index2] 
            val     += lower_weight1 * upper_weight2 * q2_1[i, lower_index1, upper_index2]  
            val     += upper_weight1 * upper_weight2 * q2_1[i, upper_index1, upper_index2] 
            temp[i] += sgn * val
        end 
    end

    return nothing 
end
    
# get interpolated value of q2_2
function get_q2_2(
    site :: Int64, 
    p1   :: Param,
    p2   :: Param,
    ch   :: Channel
    )    :: Float64 

    val  = p1.lower_weight * p2.lower_weight * ch.q2_2[site, p1.lower_index, p2.lower_index] 
    val += p1.upper_weight * p2.lower_weight * ch.q2_2[site, p1.upper_index, p2.lower_index]
    val += p1.lower_weight * p2.upper_weight * ch.q2_2[site, p1.lower_index, p2.upper_index]
    val += p1.upper_weight * p2.upper_weight * ch.q2_2[site, p1.upper_index, p2.upper_index]

    return val 
end

# get interpolated value of q2_2 on all lattice sites 
function get_q2_2_avx!(
    r        :: Reduced_lattice,
    p1       :: Param,
    p2       :: Param,
    ch       :: Channel,
    temp     :: SubArray{Float64, 1, Array{Float64, 3}},
    exchange :: Bool,
    sgn      :: Float64
    )        :: Nothing 

    # deref channel 
    q2_2 = ch.q2_2 

    # deref param 
    lower_index1  = p1.lower_index
    upper_index1  = p1.upper_index 
    lower_weight1 = p1.lower_weight
    upper_weight1 = p1.upper_weight

    lower_index2  = p2.lower_index
    upper_index2  = p2.upper_index 
    lower_weight2 = p2.lower_weight
    upper_weight2 = p2.upper_weight

    # check for site exchange
    if exchange
        indices = r.exchange

        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight1 * lower_weight2 * q2_2[indices[i], lower_index1, lower_index2]  
            val     += upper_weight1 * lower_weight2 * q2_2[indices[i], upper_index1, lower_index2] 
            val     += lower_weight1 * upper_weight2 * q2_2[indices[i], lower_index1, upper_index2]  
            val     += upper_weight1 * upper_weight2 * q2_2[indices[i], upper_index1, upper_index2] 
            temp[i] += sgn * val
        end 
    else 
        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight1 * lower_weight2 * q2_2[i, lower_index1, lower_index2]  
            val     += upper_weight1 * lower_weight2 * q2_2[i, upper_index1, lower_index2] 
            val     += lower_weight1 * upper_weight2 * q2_2[i, lower_index1, upper_index2]  
            val     += upper_weight1 * upper_weight2 * q2_2[i, upper_index1, upper_index2] 
            temp[i] += sgn * val
        end 
    end

    return nothing 
end

# get interpolated value of q3
function get_q3(
    site :: Int64, 
    p1   :: Param,
    p2   :: Param,
    p3   :: Param,
    ch   :: Channel
    )    :: Float64 

    val  = p1.lower_weight * p2.lower_weight * p3.lower_weight * ch.q3[site, p1.lower_index, p2.lower_index, p3.lower_index] 
    val += p1.upper_weight * p2.lower_weight * p3.lower_weight * ch.q3[site, p1.upper_index, p2.lower_index, p3.lower_index]
    val += p1.lower_weight * p2.upper_weight * p3.lower_weight * ch.q3[site, p1.lower_index, p2.upper_index, p3.lower_index] 
    val += p1.upper_weight * p2.upper_weight * p3.lower_weight * ch.q3[site, p1.upper_index, p2.upper_index, p3.lower_index]
    val += p1.lower_weight * p2.lower_weight * p3.upper_weight * ch.q3[site, p1.lower_index, p2.lower_index, p3.upper_index] 
    val += p1.upper_weight * p2.lower_weight * p3.upper_weight * ch.q3[site, p1.upper_index, p2.lower_index, p3.upper_index]
    val += p1.lower_weight * p2.upper_weight * p3.upper_weight * ch.q3[site, p1.lower_index, p2.upper_index, p3.upper_index] 
    val += p1.upper_weight * p2.upper_weight * p3.upper_weight * ch.q3[site, p1.upper_index, p2.upper_index, p3.upper_index]

    return val 
end

# get interpolated value of q3 on all lattice sites 
function get_q3_avx!(
    r        :: Reduced_lattice,
    p1       :: Param,
    p2       :: Param,
    p3       :: Param,
    ch       :: Channel,
    temp     :: SubArray{Float64, 1, Array{Float64, 3}},
    exchange :: Bool,
    sgn      :: Float64
    )        :: Nothing 

    # deref channel 
    q3 = ch.q3 

    # deref param 
    lower_index1  = p1.lower_index
    upper_index1  = p1.upper_index 
    lower_weight1 = p1.lower_weight
    upper_weight1 = p1.upper_weight

    lower_index2  = p2.lower_index
    upper_index2  = p2.upper_index 
    lower_weight2 = p2.lower_weight
    upper_weight2 = p2.upper_weight

    lower_index3  = p3.lower_index
    upper_index3  = p3.upper_index 
    lower_weight3 = p3.lower_weight
    upper_weight3 = p3.upper_weight

    # check for site exchange
    if exchange
        indices = r.exchange

        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight1 * lower_weight2 * lower_weight3 * q3[indices[i], lower_index1, lower_index2, lower_index3] 
            val     += upper_weight1 * lower_weight2 * lower_weight3 * q3[indices[i], upper_index1, lower_index2, lower_index3] 
            val     += lower_weight1 * upper_weight2 * lower_weight3 * q3[indices[i], lower_index1, upper_index2, lower_index3]  
            val     += upper_weight1 * upper_weight2 * lower_weight3 * q3[indices[i], upper_index1, upper_index2, lower_index3] 
            val     += lower_weight1 * lower_weight2 * upper_weight3 * q3[indices[i], lower_index1, lower_index2, upper_index3]  
            val     += upper_weight1 * lower_weight2 * upper_weight3 * q3[indices[i], upper_index1, lower_index2, upper_index3] 
            val     += lower_weight1 * upper_weight2 * upper_weight3 * q3[indices[i], lower_index1, upper_index2, upper_index3]  
            val     += upper_weight1 * upper_weight2 * upper_weight3 * q3[indices[i], upper_index1, upper_index2, upper_index3]
            temp[i] += sgn * val
        end  
    else 
        @turbo unroll = 1 for i in eachindex(temp)
            val      = lower_weight1 * lower_weight2 * lower_weight3 * q3[i, lower_index1, lower_index2, lower_index3] 
            val     += upper_weight1 * lower_weight2 * lower_weight3 * q3[i, upper_index1, lower_index2, lower_index3] 
            val     += lower_weight1 * upper_weight2 * lower_weight3 * q3[i, lower_index1, upper_index2, lower_index3]  
            val     += upper_weight1 * upper_weight2 * lower_weight3 * q3[i, upper_index1, upper_index2, lower_index3] 
            val     += lower_weight1 * lower_weight2 * upper_weight3 * q3[i, lower_index1, lower_index2, upper_index3]  
            val     += upper_weight1 * lower_weight2 * upper_weight3 * q3[i, upper_index1, lower_index2, upper_index3] 
            val     += lower_weight1 * upper_weight2 * upper_weight3 * q3[i, lower_index1, upper_index2, upper_index3]  
            val     += upper_weight1 * upper_weight2 * upper_weight3 * q3[i, upper_index1, upper_index2, upper_index3]
            temp[i] += sgn * val
        end  
    end

    return nothing 
end

# get interpolated value of channel for a given frequency buffer
function get_channel(
    site :: Int64, 
    b    :: Buffer,
    ch   :: Channel
    )    :: Float64

    val = 0.0
    
    if b.kernel == 1
        val = get_q1(site, b.p1, ch)
    elseif b.kernel == 2
        val = get_q2_1(site, b.p1, b.p2, ch)
    elseif b.kernel == 3
        val = get_q2_2(site, b.p1, b.p3, ch)
    elseif b.kernel == 4
        val = get_q3(site, b.p1, b.p2, b.p3, ch)
    end 

    return val 
end

# get interpolated value of channel for a given frequency buffer on all lattice sites
function get_channel_avx!(
    r        :: Reduced_lattice,
    b        :: Buffer,
    ch       :: Channel,
    temp     :: SubArray{Float64, 1, Array{Float64, 3}},
    exchange :: Bool,
    sgn      :: Float64
    )        :: Nothing 
    
    if b.kernel == 1
        get_q1_avx!(r, b.p1, ch, temp, exchange, sgn)
    elseif b.kernel == 2
        get_q2_1_avx!(r, b.p1, b.p2, ch, temp, exchange, sgn)
    elseif b.kernel == 3
        get_q2_2_avx!(r, b.p1, b.p3, ch, temp, exchange, sgn)
    elseif b.kernel == 4
        get_q3_avx!(r, b.p1, b.p2, b.p3, ch, temp, exchange, sgn)
    end 

    return nothing
end





# replace channel with another channel 
function replace_with!(
    ch1 :: Channel,
    ch2 :: Channel
    )   :: Nothing 

    ch1.q1   .= ch2.q1 
    ch1.q2_1 .= ch2.q2_1
    ch1.q2_2 .= ch2.q2_2
    ch1.q3   .= ch2.q3

    return nothing 
end

# multiply channel with factor 
function mult_with!(
    ch  :: Channel,
    fac :: Float64
    )   :: Nothing 

    ch.q1   .*= fac
    ch.q2_1 .*= fac
    ch.q2_2 .*= fac 
    ch.q3   .*= fac 

    return nothing 
end 

# multiply channel with factor and add to other channel
function mult_with_add_to!(
    ch2 :: Channel,
    fac :: Float64,
    ch1 :: Channel
    )   :: Nothing 

    ch1.q1   .+= fac .* ch2.q1 
    ch1.q2_1 .+= fac .* ch2.q2_1
    ch1.q2_2 .+= fac .* ch2.q2_2
    ch1.q3   .+= fac .* ch2.q3

    return nothing 
end

"""
    get_abs_max(
        ch :: Channel
        )  :: Float64

Returns maximum absolute value of a channel.
"""
function get_abs_max(
    ch :: Channel
    )  :: Float64

    max_ch = maximum(abs.(ch.q3))

    return max_ch 
end

# set asymptotic limits by scanning the boundaries of q3
function limits!(
    ch :: Channel
    )  :: Nothing

    # get dimensions 
    num_sites = size(ch.q2_1, 1)
    num_Ω     = size(ch.q2_1, 2)
    num_ν     = size(ch.q2_1, 3)

    # set q1
    for w in 1 : num_Ω
        for site in 1 : num_sites
            ch.q1[site, w] = ch.q3[site, w, end, end]
        end 
    end 

    # set q2_1 and q2_2
    for v in 1 : num_ν 
        for w in 1 : num_Ω
            for site in 1 : num_sites 
                ch.q2_1[site, w, v] = ch.q3[site, w, v, end]
                ch.q2_2[site, w, v] = ch.q3[site, w, end, v]
            end 
        end 
    end 
    
    return nothing 
end

# resample a channel to new meshes via trilinear interpolation
function resample_from_to!(
    Ω_old  :: Vector{Float64},
    ν_old  :: Vector{Float64},
    ch_old :: Channel,
    Ω_new  :: Vector{Float64},
    ν_new  :: Vector{Float64},
    ch_new :: Channel
    )      :: Nothing 

    # get dimensions 
    num_sites = size(ch_old.q2_1, 1)
    num_Ω     = size(ch_old.q2_1, 2)
    num_ν     = size(ch_old.q2_1, 3)

    # resample q3 
    for vp in 1 : num_ν
        for v in 1 : num_ν
            for w in 1 : num_Ω
                # get interpolation parameters 
                p1 = get_param(Ω_new[w], Ω_old)
                p2 = get_param(ν_new[v], ν_old)
                p3 = get_param(ν_new[vp], ν_old)

                for site in 1 : num_sites
                    ch_new.q3[site, w, v, vp] = get_q3(site, p1, p2, p3, ch_old)
                end 
            end 
        end 
    end 

    # set asymptotic limits
    limits!(ch_new)

    return nothing 
end