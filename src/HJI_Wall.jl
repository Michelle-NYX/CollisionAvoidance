# export HJICache, GriddedInterpolation, Gridded, Linear, SVector    # TODO: I shouldn't have to do this


@maintain_type struct HJIRelativeState_Wall{T} <: FieldVector{3,T}
    dN::T #relative 'y' position in Robot frame
    ψ::T # world frame heading of robot (from E) (need converter)
    Ux::T # robot longitudinal velocity
end

function HJIRelativeState_Wall(us::BicycleState, y::Float64)
    ψ = pi/2 + us.ψ
    dN = us.N - y
    HJIRelativeState_Wall(dN, ψ, us.Ux)
end

struct HJICache_Wall
    grid_knots::NTuple{3,Vector{Float32}}
    V ::GriddedInterpolation{Float32,3,Float32,Gridded{Linear},NTuple{3,Vector{Float32}}}
    ∇V::GriddedInterpolation{SVector{3,Float32},3,SVector{3,Float32},Gridded{Linear},NTuple{3,Vector{Float32}}}
end

function placeholder_HJICache_Wall()
    grid_knots = tuple((Float32[-1000., 1000.] for i in 1:3)...)
    V  = interpolate(Float32, Float32, grid_knots, zeros(Float32,2,2,2), Gridded(Linear()))
    ∇V = interpolate(Float32, SVector{3,Float32}, grid_knots, zeros(SVector{3,Float32},2,2,2), Gridded(Linear()))
    HJICache_Wall(grid_knots, V, ∇V)
end

function load_HJICache_Wall(fname::String)
    if endswith(fname, ".jld2")
        @load fname mgrid mV_raw m∇V_raw
        grid_knots = mgrid
        V  = interpolate(Float32, Float32, mgrid, mV_raw, Gridded(Linear()))
        ∇V = interpolate(Float32, SVector{3,Float32}, mgrid, m∇V_raw, Gridded(Linear()))
        HJICache_Wall(grid_knots, V, ∇V)
    else
        error("Unknown file type for loading HJICache_Wall")
    end
end

function save(fname::String, cache::HJICache_Wall)
    grid_knots = cache.grid_knots
    V_raw  = cache.V.coefs
    ∇V_raw = Array(reinterpret(Float32, cache.∇V.coefs))
    @save fname grid_knots V_raw ∇V_raw
end

function Base.getindex(cache::HJICache_Wall, x::HJIRelativeState_Wall{T}) where {T}
    if all(cache.grid_knots[i][1] <= x[i] <= cache.grid_knots[i][end] for i in 1:length(cache.grid_knots))
        (V=cache.V(x[1], x[2], x[3]), ∇V=cache.∇V(x[1], x[2], x[3]))    # avoid splatting penalty
    else
        (V=T(Inf), ∇V=zeros(SVector{3,T}))
    end
end

function relative_dynamics(X::VehicleModel, (dN, ψ, Ux)::StaticVector{3},    # relative state
                                            uR::StaticVector{2})                      # robot control
    m = X.bicycle_model.m
    δ = uR[1]
    Fx = uR[2]
    sΔψ, cΔψ = sincos(ψ)
    SVector(
        Ux*sΔψ,
        δ,
        Fx / m
    )
end

function optimal_control(X::VehicleModel, relative_state::HJIRelativeState_Wall, ∇V::StaticVector{3}, uMode=:max)
    m = X.bicycle_model.m
    aMax = 2.
    aMin = -3.5
    ωLimit = deg2rad(60)

    sgn = (uMode == :max ? 1 : -1)

    A = ∇V[2] #steer
    B = ∇V[3] # accel
    
    δ_opt  = ifelse(A >= 0, sgn*ωLimit, -sgn*ωLimit)
    if uMode == :max
        a_opt = ifelse(B >= 0, aMax, aMin)
    else
        a_opt = ifelse(B >= 0, aMin, aMax)
    end
    Fx_opt = a_opt * m
    BicycleControl2(δ_opt, Fx_opt)
end

function compute_reachability_constraint(X::VehicleModel, cache::HJICache_Wall, relative_state::HJIRelativeState_Wall, ϵ = 0.,
                                         uR_lin=optimal_control(X, relative_state, cache[relative_state].∇V))    # definitely not the correct choice...
    V, ∇V = cache[relative_state]
    if V >= ϵ
        (M=SVector{2,Float64}(0, 0), b=1.0)
    else
        f = uR -> dot(∇V, relative_dynamics(X, SVector(relative_state), uR))

        ∇H_uR = ForwardDiff.gradient(f, SVector(uR_lin))
        (M=∇H_uR, b=dot(∇V, relative_dynamics(X, relative_state, uR_lin)) - dot(∇H_uR, uR_lin)) # so that H = dot(∇V, uR) ≈ ∇H_uR*uR + c
    end
end
