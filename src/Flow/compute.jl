# compute the full right side of the BSEs for all channels
function compute_Γ!(
    Λ      :: Float64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a1     :: Action,
    a2     :: Action,
    tbuffs :: Vector{NTuple{3, Matrix{Float64}}},
    temps  :: Vector{Array{Float64, 3}},
    corrs  :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    @sync begin
        for w1 in 1 : m.num_Ω
            # compute q1
            Threads.@spawn begin
                compute_channel_s_BSE!(Λ, 1, w1, -1, -1, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                compute_channel_t_BSE!(Λ, 1, w1, -1, -1, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                compute_channel_u_BSE!(Λ, 1, w1, -1, -1, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
            end

            for w3 in 1 : m.num_ν
                # compute q2_1
                Threads.@spawn begin
                    compute_channel_s_BSE!(Λ, 2, w1, w3, -1, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_BSE!(Λ, 2, w1, w3, -1, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_BSE!(Λ, 2, w1, w3, -1, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                # compute q2_2
                Threads.@spawn begin
                    compute_channel_s_BSE!(Λ, 3, w1, -1, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_BSE!(Λ, 3, w1, -1, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_BSE!(Λ, 3, w1, -1, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                for w2 in w3 : m.num_ν
                    # compute q3
                    Threads.@spawn begin
                        compute_channel_s_BSE!(Λ, 4, w1, w2, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_t_BSE!(Λ, 4, w1, w2, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_u_BSE!(Λ, 4, w1, w2, w3, r, m, a1, a2, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    end 
                end 
            end 
        end 
    end

    symmetrize!(r, a2) 

    return nothing
end

# compute the full right side of the Katanin truncated flow equations for all channels
function compute_dΓ_1l!(
    Λ      :: Float64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a      :: Action,
    da     :: Action,
    tbuffs :: Vector{NTuple{3, Matrix{Float64}}},
    temps  :: Vector{Array{Float64, 3}},
    corrs  :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # compute boundary corrections
    compute_corrs_kat!(Λ, m, a, da, corrs, Γ_tol)

    @sync begin
        for w1 in 1 : m.num_Ω
            # compute q1
            Threads.@spawn begin 
                compute_channel_s_kat!(Λ, 1, w1, -1, -1, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                compute_channel_t_kat!(Λ, 1, w1, -1, -1, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                compute_channel_u_kat!(Λ, 1, w1, -1, -1, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
            end

            for w3 in 1 : m.num_ν
                # compute q2_1
                Threads.@spawn begin 
                    compute_channel_s_kat!(Λ, 2, w1, w3, -1, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_kat!(Λ, 2, w1, w3, -1, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_kat!(Λ, 2, w1, w3, -1, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                # compute q2_2
                Threads.@spawn begin 
                    compute_channel_s_kat!(Λ, 3, w1, -1, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_kat!(Λ, 3, w1, -1, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_kat!(Λ, 3, w1, -1, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                for w2 in w3 : m.num_ν
                    # compute q3
                    Threads.@spawn begin 
                        compute_channel_s_kat!(Λ, 4, w1, w2, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_t_kat!(Λ, 4, w1, w2, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_u_kat!(Λ, 4, w1, w2, w3, r, m, a, da, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    end
                end 
            end
        end 
    end

    symmetrize!(r, da)

    return nothing
end

# compute the full right side of the two loop truncated flow equations for all channels
function compute_dΓ_2l!(
    Λ      :: Float64,
    r      :: Reduced_lattice,
    m      :: Mesh,
    a      :: Action,
    da     :: Action,
    da_l   :: Action,
    tbuffs :: Vector{NTuple{3, Matrix{Float64}}},
    temps  :: Vector{Array{Float64, 3}},
    corrs  :: Array{Float64, 3},
    eval   :: Int64,
    Γ_tol  :: NTuple{2, Float64}
    )      :: Nothing

    # compute one loop
    compute_dΓ_1l!(Λ, r, m, a, da, tbuffs, temps, corrs, eval, Γ_tol)

    # compute boundary corrections
    compute_corrs!(Λ, m, a, corrs, Γ_tol)

    @sync begin 
        for w1 in 1 : m.num_Ω
            # compute q1
            Threads.@spawn begin 
                compute_channel_s_left!(Λ, 1, w1, -1, -1, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                compute_channel_t_left!(Λ, 1, w1, -1, -1, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                compute_channel_u_left!(Λ, 1, w1, -1, -1, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
            end

            for w3 in 1 : m.num_ν
                # compute q2_1
                Threads.@spawn begin 
                    compute_channel_s_left!(Λ, 2, w1, w3, -1, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_left!(Λ, 2, w1, w3, -1, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_left!(Λ, 2, w1, w3, -1, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                # compute q2_2
                Threads.@spawn begin 
                    compute_channel_s_left!(Λ, 3, w1, -1, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_left!(Λ, 3, w1, -1, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_left!(Λ, 3, w1, -1, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                for w2 in 1 : m.num_ν
                    # compute q3
                    Threads.@spawn begin 
                        compute_channel_s_left!(Λ, 4, w1, w2, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_t_left!(Λ, 4, w1, w2, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_u_left!(Λ, 4, w1, w2, w3, r, m, a, da, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    end
                end 
            end 
        end      
    end 

    symmetrize_add_to!(r, da_l, da)

    return nothing 
end

# compute the full right side of the multiloop truncated flow equations for all channels
function compute_dΓ_ml!(
    Λ       :: Float64,
    r       :: Reduced_lattice,
    m       :: Mesh,
    loops   :: Int64,
    a       :: Action,
    da      :: Action,
    da_l    :: Action,
    da_c    :: Action,
    da_temp :: Action,
    da_Σ    :: Action,
    tbuffs  :: Vector{NTuple{3, Matrix{Float64}}},
    temps   :: Vector{Array{Float64, 3}},
    corrs   :: Array{Float64, 3},
    eval    :: Int64,
    Γ_tol   :: NTuple{2, Float64}
    )       :: Nothing

    # compute two loop
    compute_dΓ_2l!(Λ, r, m, a, da, da_l, tbuffs, temps, corrs, eval, Γ_tol)

    # update temporary buffer and reset terms for self energy corrections
    reset_Γ!(da_temp)
    reset_Γ!(da_Σ)
    symmetrize_add_to!(r, da_l, da_temp)    

    for loop in 3 : loops
        @sync begin 
            for w1 in 1 : m.num_Ω
                # compute q1
                Threads.@spawn begin 
                    compute_channel_s_central!(Λ, 1, w1, -1, -1, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_central!(Λ, 1, w1, -1, -1, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_central!(Λ, 1, w1, -1, -1, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                for w3 in 1 : m.num_ν
                     # compute q2_1
                    Threads.@spawn begin 
                        compute_channel_s_central!(Λ, 2, w1, w3, -1, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_t_central!(Λ, 2, w1, w3, -1, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_u_central!(Λ, 2, w1, w3, -1, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    end

                     # compute q2_2
                    Threads.@spawn begin 
                        compute_channel_s_central!(Λ, 3, w1, -1, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_t_central!(Λ, 3, w1, -1, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_u_central!(Λ, 3, w1, -1, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    end

                    for w2 in w3 : m.num_ν
                         # compute q3
                        Threads.@spawn begin 
                            compute_channel_s_central!(Λ, 4, w1, w2, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                            compute_channel_t_central!(Λ, 4, w1, w2, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                            compute_channel_u_central!(Λ, 4, w1, w2, w3, r, m, a, da_l, da_c, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        end
                    end 
                end 
            end 
        end

        @sync begin 
            for w1 in 1 : m.num_Ω
                # compute q1
                Threads.@spawn begin 
                    compute_channel_s_left!(Λ, 1, w1, -1, -1, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_t_left!(Λ, 1, w1, -1, -1, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    compute_channel_u_left!(Λ, 1, w1, -1, -1, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                end

                for w3 in 1 : m.num_ν
                    # compute q2_1
                    Threads.@spawn begin 
                        compute_channel_s_left!(Λ, 2, w1, w3, -1, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_t_left!(Λ, 2, w1, w3, -1, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_u_left!(Λ, 2, w1, w3, -1, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    end

                    # compute q2_2
                    Threads.@spawn begin 
                        compute_channel_s_left!(Λ, 3, w1, -1, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_t_left!(Λ, 3, w1, -1, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                        compute_channel_u_left!(Λ, 3, w1, -1, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                    end

                    # compute q3
                    for w2 in 1 : m.num_ν
                        Threads.@spawn begin 
                            compute_channel_s_left!(Λ, 4, w1, w2, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                            compute_channel_t_left!(Λ, 4, w1, w2, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
                            compute_channel_u_left!(Λ, 4, w1, w2, w3, r, m, a, da_temp, da_l, tbuffs[Threads.threadid()], temps[Threads.threadid()], corrs, eval, Γ_tol)
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
    end

    return nothing
end