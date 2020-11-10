struct NonlinearObjective <: Objective
    stage_cost_func #a nonlinear stage cost function to evaluate at each time step
    terminal_cost_func #a nonlinear terminal cost to evaluate at the final time step
end

function objective(Z, obj::NonlinearObjective, idx, T)
    J = 0.0

    for t = 1:(T-1)
        x = view(Z, idx.x[t])
        u = view(Z, idx.u[t])

        J += stage_cost_func(x,u)
    end
    x = view(Z, idx.x[T])
    J += obj.terminal_cost_func(x)

    return J
end

function objective_gradient!(∇J, Z, obj::NonlinearObjective, idx, T)

    for t = 1:T
        x = view(Z, idx.x[t])
        Q = obj.Q[t]
        q = obj.q[t]
        c = obj.c[t]

        ∇J[idx.x[t]] += 2.0 * Q * x + q

        t == T && continue

        u = view(Z, idx.u[t])
        R = obj.R[t]
        r = obj.r[t]

        ∇J[idx.u[t]] += 2.0 * R * u + r
    end

    return nothing
end
