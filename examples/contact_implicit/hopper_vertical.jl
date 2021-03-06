include(joinpath(pwd(), "src/models/hopper.jl"))
include(joinpath(pwd(), "src/constraints/contact.jl"))
include(joinpath(pwd(), "src/constraints/free_time.jl"))

# Free-time model
model_ft = free_time_model(model)

function fd(model::Hopper, x⁺, x, u, w, h, t)
	q3 = view(x⁺, model.nq .+ (1:model.nq))
	q2⁺ = view(x⁺, 1:model.nq)
	q2⁻ = view(x, model.nq .+ (1:model.nq))
	q1 = view(x, 1:model.nq)
	u_ctrl = view(u, model.idx_u)
	λ = view(u, model.idx_λ)
	b = view(u, model.idx_b)
	h = u[end]

	[q2⁺ - q2⁻;
	((1.0 / h) * (M_func(model, q1) * (SVector{4}(q2⁺) - SVector{4}(q1))
	- M_func(model, q2⁺) * (SVector{4}(q3) - SVector{4}(q2⁺)))
	+ transpose(B_func(model, q3)) * SVector{2}(u_ctrl)
	+ transpose(N_func(model, q3)) * SVector{1}(λ)
	+ transpose(P_func(model, q3)) * SVector{2}(b)
	- h * G_func(model, q2⁺))]
end

function maximum_dissipation(model::Hopper, x⁺, u, h)
	q3 = x⁺[model.nq .+ (1:model.nq)]
	q2 = x⁺[1:model.nq]
	ψ = u[model.idx_ψ]
	ψ_stack = ψ[1] * ones(model.nb)
	η = u[model.idx_η]
	h = u[end]
	return P_func(model, q3) * (q3 - q2) / h + ψ_stack - η
end

# Horizon
T = 11

# Time step
tf = 1.0
h = tf / (T - 1)

# Bounds
_uu = Inf * ones(model_ft.m)
_uu[end] = 5.0 * h
_ul = zeros(model_ft.m)
_ul[model_ft.idx_u] .= -Inf
_ul[end] = 0.25 * h
ul, uu = control_bounds(model_ft, T, _ul, _uu)

# Initial and final states
z_h = 0.0
q1 = [0.0, 0.5 + z_h, 0.0, 0.5]

xl, xu = state_bounds(model_ft, T,
		[model_ft.qL; model_ft.qL],
		[model_ft.qU; model_ft.qU],
        x1 = [q1; q1])

# Objective
Qq = Diagonal([1.0, 1.0, 1.0, 1.0])
Q = cat(0.5 * Qq, 0.5 * Qq, dims = (1, 2))
QT = cat(0.5 * Qq, 100.0 * Diagonal(ones(model_ft.nq)), dims = (1, 2))
R = Diagonal([1.0e-1, 1.0e-3, zeros(model_ft.m - model_ft.nu)...])

obj_tracking = quadratic_time_tracking_objective(
    [t < T ? Q : QT for t = 1:T],
    [R for t = 1:T-1],
    [[q1; q1] for t = 1:T],
    [zeros(model_ft.m) for t = 1:T],
    1.0)
obj_contact_penalty = PenaltyObjective(100.0, model_ft.m - 1)

obj = MultiObjective([obj_tracking, obj_contact_penalty])

# Constraints
con_free_time = free_time_constraints(T)
con_contact = contact_constraints(model_ft, T)
con = multiple_constraints([con_free_time, con_contact])

# Problem
prob = trajectory_optimization_problem(model_ft,
               obj,
               T,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con
               )

# Trajectory initialization
X0 = [[q1; q1] for t = 1:T] # linear interpolation on state
U0 = [[1.0e-5 * rand(model_ft.m-1); h] for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0, U0, prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
include("/home/taylor/.julia/dev/SNOPT7/src/SNOPT7.jl")

@time Z̄ = solve(prob, copy(Z0),
	nlp = :SNOPT7,
	tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5)

check_slack(Z̄, prob)
X̄, Ū = unpack(Z̄, prob)

@show Ū[4][end]
@show check_slack(Z̄, prob)

using Plots
plot(hcat(Ū...)[end,:], linetype=:steppost)

include(joinpath(pwd(), "src/models/visualize.jl"))
vis = Visualizer()
open(vis)
visualize!(vis, model_ft, state_to_configuration(X̄), Δt = Ū[1][end])
