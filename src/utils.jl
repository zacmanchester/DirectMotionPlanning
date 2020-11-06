"""
    linear interpolation between two vectors over horizon T
"""
function linear_interp(x0, xf, T)
    n = length(x0)
    X = [copy(Array(x0)) for t = 1:T]
    for t = 1:T
        for i = 1:n
            X[t][i] = (xf[i] - x0[i]) / (T - 1) * (t - 1) + x0[i]
        end
    end
    return X
end

function row_col!(row, col, r, c)
    for cc in c
        for rr in r
            push!(row, convert(Int, rr))
            push!(col, convert(Int, cc))
        end
    end
    return row, col
end

function row_col_cartesian!(row, col, r, c)
    for i = 1:length(r)
        push!(row, convert(Int, r[i]))
        push!(col, convert(Int, c[i]))
    end
    return row, col
end

"""
    random controls
"""
function random_controls(model, T, r = 1.0)
	[r * ones(model.m) for t = 1:T-1]
end

"""
    control bounds trajectories
"""
function control_bounds(model, T, l, u)
    l_traj = [l * ones(model.m) for t = 1:T-1]
    u_traj = [u * ones(model.m) for t = 1:T-1]

    return l_traj, u_traj
end

function control_bounds(model, T,
        l::AbstractVector{S} = -Inf * ones(model.m),
        u::AbstractVector{S} = Inf * ones(model.m)) where S

    l_traj = [copy(l) for t = 1:T-1]
    u_traj = [copy(u) for t = 1:T-1]

    return l_traj, u_traj
end

"""
    state bounds trajectories
"""
function state_bounds(model, T,
        l::AbstractVector{S} = -Inf * ones(model.n),
        u::AbstractVector{S} = Inf * ones(model.n);
        x1 = Inf * ones(model.n),
        xT = Inf * ones(model.n)) where S

    l_traj = [copy(l) for t = 1:T]
    u_traj = [copy(u) for t = 1:T]

    # initial and final conditions
    x1_mask = isfinite.(x1)
    xT_mask = isfinite.(xT)

    if length(x1_mask) > 0
        l_traj[1][x1_mask] = copy(x1[x1_mask])
        u_traj[1][x1_mask] = copy(x1[x1_mask])
    end

    if length(xT_mask) > 0
        l_traj[T][xT_mask] = copy(xT[xT_mask])
        u_traj[T][xT_mask] = copy(xT[xT_mask])
    end

    return l_traj, u_traj
end

function check_slack(Z, prob)
    model = prob.prob.model
    idx = prob.prob.idx
    S̄ = [Z[idx.u[t]][model.idx_s][1] for t = 1:T-1]
    @show norm(S̄, Inf)
end

function state_to_configuration(X)
    T = length(X)
    nq = convert(Int, floor(length(X[1])/2))
    [X[1][1:nq], [X[t][nq .+ (1:nq)] for t = 1:T]...]
end

function configuration_to_state(Q)
    T = length(Q)
    [[t == 1 ? Q[1] : Q[t-1]; Q[t]] for t = 1:T]
end

function free_time_model(model::T) where T
	model_ft = typeof(model)([f == :m ? getfield(model,f) + 1 : getfield(model,f)
	 	for f in fieldnames(typeof(model))]...)
	return model_ft
end

function additive_noise_model(model::T) where T
	model_ft = typeof(model)([f == :d ? model.n : getfield(model,f)
	 	for f in fieldnames(typeof(model))]...)
	return model_ft
end
