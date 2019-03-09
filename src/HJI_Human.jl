# export HJICache, GriddedInterpolation, Gridded, Linear, SVector    # TODO: I shouldn't have to do this

@maintain_type struct SimpleCarState{T} <: FieldVector{4,T}
    E::T    # world frame "x" position of CM
    N::T    # world frame "y" position of CM
    ψ::T    # world frame heading of vehicle
    V::T    # speed of vehicle (velocity assumed to be in the heading direction)
end

@maintain_type struct HJIRelativeState_Human{T} <: FieldVector{5,T}
    E_rel::T #relative 'x' position in Robot frame
    N_rel::T #relative 'y' position in Robot frame
    ψ_us::T # world frame heading of robot (from E) (need converter)
    ψ_them::T # world frame heading of human (from E)
    Ux::T # robot longitudinal velocity
end

function HJIRelativeState_Human(us::BicycleState, them::SimpleCarState)
    us_ψ = pi/2 + us.ψ
    sψ, cψ = sincos(us_ψ)
    E_rel, N_rel = @SMatrix([cψ sψ; -sψ cψ])*SVector(them.E - us.E, them.N - us.N)
    HJIRelativeState_Human(E_rel, N_rel, us_ψ, them.ψ, us.Ux)
end

struct HJICache_Human
    grid_knots::NTuple{5,Vector{Float32}}
    V ::GriddedInterpolation{Float32,5,Float32,Gridded{Linear},NTuple{5,Vector{Float32}}}
    ∇V::GriddedInterpolation{SVector{5,Float32},5,SVector{5,Float32},Gridded{Linear},NTuple{5,Vector{Float32}}}
end

function placeholder_HJICache_Human()
    grid_knots = tuple((Float32[-1000., 1000.] for i in 1:5)...)
    V  = interpolate(Float32, Float32, grid_knots, zeros(Float32,2,2,2,2,2), Gridded(Linear()))
    ∇V = interpolate(Float32, SVector{5,Float32}, grid_knots, zeros(SVector{5,Float32},2,2,2,2,2), Gridded(Linear()))
    HJICache_Human(grid_knots, V, ∇V)
end

function load_HJICache_Human(fname::String)
    if endswith(fname, ".jld2")
        @load fname mgrid mV_raw m∇V_raw
        grid_knots = mgrid
        V  = interpolate(Float32, Float32, mgrid, mV_raw, Gridded(Linear()))
        ∇V = interpolate(Float32, SVector{5,Float32}, mgrid, m∇V_raw, Gridded(Linear()))
        HJICache_Human(grid_knots, V, ∇V)
    else
        error("Unknown file type for loading HJICache_Human")
    end
end

function save(fname::String, cache::HJICache_Human)
    grid_knots = cache.grid_knots
    V_raw  = cache.V.coefs
    ∇V_raw = Array(reinterpret(Float32, cache.∇V.coefs))
    @save fname grid_knots V_raw ∇V_raw
end

function Base.getindex(cache::HJICache_Human, x::HJIRelativeState_Human{T}) where {T}
    if all(cache.grid_knots[i][1] <= x[i] <= cache.grid_knots[i][end] for i in 1:length(cache.grid_knots))
        (V=cache.V(x[1], x[2], x[3], x[4], x[5]), ∇V=cache.∇V(x[1], x[2], x[3], x[4], x[5]))    # avoid splatting penalty
    else
        (V=T(Inf), ∇V=zeros(SVector{5,T}))
    end
end

function simple_car_dynamics((E, N, ψ, V)::StaticVector{4}, ωH::Float64)
    sψ, cψ = sincos(ψ)
    SVector(
        V*cψ,
        V*sψ,
        ωH,
        0.
        )
end

function relative_dynamics(X::VehicleModel, (E_rel, N_rel, ψ_us, ψ_them, Ux)::StaticVector{5},    # relative state
                                            uR::StaticVector{2},                       # robot control
                                            ω::Float64)                        # human control
    Vel = 4.
    m = X.bicycle_model.m
    δ = uR[1]
    Fx = uR[2]
    sΔψ, cΔψ = sincos(ψ_them - ψ_us)
    SVector(
        Vel*cΔψ - Ux + N_rel*δ,
        Vel*sΔψ      - E_rel*δ,
        δ,
        ω,
        Fx / m
    )
end

function optimal_disturbance(relative_state::HJIRelativeState_Human, ∇V::StaticVector{5}, dMode=:min)
    sgn = (dMode == :max ? 1 : -1)
    lam  = ∇V[4]    # θh dot
    dlimit = deg2rad(60)

    if abs(lam) < 1e-3
        return (0.)
    else
        ω_opt = ifelse(lam >= 0, sgn*dlimit, -sgn*dlimit)
        return (ω_opt)
    end
end

function optimal_control(X::VehicleModel, relative_state::HJIRelativeState_Human, ∇V::StaticVector{5}, uMode=:max)
    m = X.bicycle_model.m
    aMax = 2.
    aMin = -3.5
    ωLimit = deg2rad(60)

    sgn = (uMode == :max ? 1 : -1)

    A = ∇V[1] * relative_state.N_rel - ∇V[2] * relative_state.E_rel + ∇V[3] #steer
    B = ∇V[5] # accel
    
    δ_opt  = ifelse(A >= 0, sgn*ωLimit, -sgn*ωLimit)
    if uMode == :max
        a_opt = ifelse(B >= 0, aMax, aMin)
    else
        a_opt = ifelse(B >= 0, aMin, aMax)
    end
    Fx_opt = a_opt * m
    BicycleControl2(δ_opt, Fx_opt)
end

function compute_reachability_constraint(X::VehicleModel, cache::HJICache_Human, relative_state::HJIRelativeState_Human, ϵ = 0.,
                                         uR_lin=optimal_control(X, relative_state, cache[relative_state].∇V))    # definitely not the correct choice...
    V, ∇V = cache[relative_state]
    if V >= ϵ
        (M=SVector{2,Float64}(0, 0), b=1.0)
    else
        uH_opt = optimal_disturbance(relative_state, ∇V)
        f = uR -> dot(∇V, relative_dynamics(X, SVector(relative_state), uR, uH_opt))

        ∇H_uR = ForwardDiff.gradient(f, SVector(uR_lin))
        (M=∇H_uR, b=dot(∇V, relative_dynamics(X, relative_state, uR_lin, uH_opt)) - dot(∇H_uR, uR_lin)) # so that H = dot(∇V, uR) ≈ ∇H_uR*uR + c
    end
end
