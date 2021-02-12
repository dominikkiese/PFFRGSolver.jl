# full BSE computation
function compute_Γ!(
    Λ      :: Float64,
    r      :: reduced_lattice,
    m      :: mesh,
    a1     :: action,
    a2     :: action,
    tbuffs :: Vector{NTuple{3, Matrix{Float64}}},
    temps  :: Vector{Array{Float64, 3}},
    eval   :: Int64
    )      :: Nothing

    @sync begin
        for w1 in eachindex(m.Ω)
            for w3 in eachindex(m.ν)
                for w2 in w3 : length(m.ν)
                    Threads.@spawn begin
                        compute_channel_s_BSE!(Λ, w1, w2, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        compute_channel_t_BSE!(Λ, w1, w2, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        compute_channel_u_BSE!(Λ, w1, w2, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                    end
                end 
            end 
        end 
    end

    symmetrize!(r, a2) 

    return nothing
end

# full 1l computation
function compute_dΓ_1l!(
    Λ      :: Float64,
    r      :: reduced_lattice,
    m      :: mesh,
    a      :: action,
    da     :: action,
    tbuffs :: Vector{NTuple{3, Matrix{Float64}}},
    temps  :: Vector{Array{Float64, 3}},
    eval   :: Int64
    )      :: Nothing

    @sync begin
        for w1 in eachindex(m.Ω)
            for w3 in eachindex(m.ν)
                for w2 in w3 : length(m.ν)
                    Threads.@spawn begin 
                        compute_channel_s_kat!(Λ, w1, w2, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        compute_channel_t_kat!(Λ, w1, w2, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        compute_channel_u_kat!(Λ, w1, w2, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                    end
                end 
            end
        end 
    end

    symmetrize!(r, da)

    return nothing
end

# full 2l computation
function compute_dΓ_2l!(
    Λ      :: Float64,
    r      :: reduced_lattice,
    m      :: mesh,
    a      :: action,
    da     :: action,
    da_l   :: action,
    tbuffs :: Vector{NTuple{3, Matrix{Float64}}},
    temps  :: Vector{Array{Float64, 3}},
    eval   :: Int64
    )      :: Nothing

    # compute one loop
    compute_dΓ_1l!(Λ, r, m, a, da, tbuffs, temps, eval)

    @sync begin 
        for w1 in eachindex(m.Ω)
            for w3 in eachindex(m.ν)
                for w2 in eachindex(m.ν)
                    Threads.@spawn begin 
                        compute_channel_s_left!(Λ, w1, w2, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        compute_channel_t_left!(Λ, w1, w2, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        compute_channel_u_left!(Λ, w1, w2, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                    end
                end 
            end 
        end      
    end 

    symmetrize_add_to!(r, da_l, da)

    return nothing 
end

# full ml computation
function compute_dΓ_ml!(
    Λ       :: Float64,
    r       :: reduced_lattice,
    m       :: mesh,
    loops   :: Int64,
    a       :: action,
    da      :: action,
    da_l    :: action,
    da_c    :: action,
    da_temp :: action,
    da_Σ    :: action,
    tbuffs  :: Vector{NTuple{3, Matrix{Float64}}},
    temps   :: Vector{Array{Float64, 3}},
    eval    :: Int64
    )       :: Nothing

    # compute two loop
    compute_dΓ_2l!(Λ, r, m, a, da, da_l, tbuffs, temps, eval)

    # update temporary buffer and reset terms for self energy corrections
    reset_Γ!(da_temp)
    reset_Γ!(da_Σ)
    symmetrize_add_to!(r, da_l, da_temp)    

    # init loop order
    o = 2

    while o < loops
        @sync begin 
            for w1 in eachindex(m.Ω)
                for w3 in eachindex(m.ν)
                    for w2 in w3 : length(m.ν)
                        Threads.@spawn begin 
                            compute_channel_s_central!(Λ, w1, w2, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                            compute_channel_t_central!(Λ, w1, w2, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                            compute_channel_u_central!(Λ, w1, w2, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        end
                    end 
                end 
            end 
        end

        @sync begin 
            for w1 in eachindex(m.Ω)
                for w3 in eachindex(m.ν)
                    for w2 in eachindex(m.ν)
                        Threads.@spawn begin 
                            compute_channel_s_left!(Λ, w1, w2, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                            compute_channel_t_left!(Λ, w1, w2, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                            compute_channel_u_left!(Λ, w1, w2, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], eval)
                        end
                    end 
                end 
            end              
        end
        
        # update temporary buffer
        symmetrize!(r, da_c)
        replace_with_Γ!(da_temp, da_c)
        symmetrize_add_to!(r, da_l, da_temp)

        # update self energy corrections and flow
        add_to_Γ!(da_c, da_Σ)
        add_to_Γ!(da_temp, da)

        # increment loop order
        o += 1
    end

    return nothing
end