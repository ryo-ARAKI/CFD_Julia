#=
Julia program to solve 1D inviscid Burgers equation by WENO-5 (5th-order Weighted Essentially Non-Oscillatory) scheme with Dirichlet boundry condition
=#


"""
Module for parameters and variables
"""
module ParamVar
    struct Parameters
        nx::Int64
        ns::Int64
        dt::Float64
        tm::Float64
        dx::Float64
        nt::Int64
        ds::Float64
    end

    mutable struct Variables
        x::Array{Float64}  # Global coordinate
        u::Array{Float64, 2}  # Numerical solution
    end
end  # ParamVar


"""
Module for 1D inviscid Burgers equation by WENO-5 (5th-order Weighted Essentially Non-Oscillatory) scheme with Dirichlet boundry condition
"""
module InviscidBergers1D_WENO5_Dirichlet
    """
    Compute numerical solution
        - Time integration using Runge-Kutta third order
        - 5th-order Compact WENO scheme for spatial terms
    """
    function numerical(nx,ns,nt,dx,dt,u)
        x = Array{Float64}(undef, nx+1)
        un = Array{Float64}(undef, nx+1) # numerical solsution at every time step
        ut = Array{Float64}(undef, nx+1) # temporary array during RK3 integration
        r = Array{Float64}(undef, nx)

        k = 1 # record index
        freq = Int64(nt/ns)

        for i = 1:nx+1
            x[i] = dx*(i-1)
            un[i] = sin(2.0*pi*x[i])
            u[i,k] = un[i] # store solution at t=0
        end

        # dirichlet boundary condition
        u[1,k], u[nx+1,k] = 0.0, 0.0
        un[1] = 0.0
        un[nx+1] = 0.0

        # dirichlet boundary condition for temporary array
        ut[1] = 0.0
        ut[nx+1] = 0.0

        for j = 1:nt
            rhs(nx,dx,un,r)

            for i = 2:nx
                ut[i] = un[i] + dt*r[i]
            end

            rhs(nx,dx,ut,r)

            for i = 2:nx
                ut[i] = 0.75*un[i] + 0.25*ut[i] + 0.25*dt*r[i]
            end

            rhs(nx,dx,ut,r)

            for i = 2:nx
                un[i] = (1.0/3.0)*un[i] + (2.0/3.0)*ut[i] + (2.0/3.0)*dt*r[i]
            end

            if (mod(j,freq) == 0)
                u[:,k] = un[:]
                k = k+1
            end
        end
    end

    """
    Calculate right hand term of the inviscid Burgers equation
    r = -u∂u/∂x
    """
    function rhs(nx,dx,u,r)
        uL = Array{Float64}(undef, nx)
        uR = Array{Float64}(undef, nx+1)

        wenoL(nx,u,uL)

        wenoR(nx,u,uR)

        for i = 2:nx
            if (u[i] >= 0.0)
                r[i] = -u[i]*(uL[i] - uL[i-1])/dx
            else
                r[i] = -u[i]*(uR[i+1] - uR[i])/dx
            end
        end
    end

    """
    WENO reconstruction for upwind direction (positive; left to right)
    u(i): solution values at finite difference grid nodes i = 1,...,N+1
    f(j): reconstructed values at nodes j = i+1/2; j = 1,...,N
    """
    function wenoL(n,u,f)
        a = Array{Float64}(undef, n)
        b = Array{Float64}(undef, n)
        c = Array{Float64}(undef, n)
        r = Array{Float64}(undef, n)

        i = 1
        v1 = 3.0*u[i] - 2.0*u[i+1]
        v2 = 2.0*u[i] - u[i+1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = u[i+2]
        f[i] = wcL(v1,v2,v3,v4,v5)

        i = 2
        v1 = 2.0*u[i-1] - u[i]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = u[i+2]
        f[i] = wcL(v1,v2,v3,v4,v5)

        for i = 3:n-1
            v1 = u[i-2]
            v2 = u[i-1]
            v3 = u[i]
            v4 = u[i+1]
            v5 = u[i+2]
            f[i] = wcL(v1,v2,v3,v4,v5)
        end

        i = n
        v1 = u[i-2]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = 2.0*u[i+1]-u[i]
        f[i] = wcL(v1,v2,v3,v4,v5)

    end

    """
    CRWENO reconstruction for downwind direction (negative; right to left)
    u(i): solution values at finite difference grid nodes i = 1,...,N+1
    f(j): reconstructed values at nodes j = i-1/2; j = 2,...,N+1
    """
    function wenoR(n,u,f)
        a = Array{Float64}(undef, n+1)
        b = Array{Float64}(undef, n+1)
        c = Array{Float64}(undef, n+1)
        r = Array{Float64}(undef, n+1)

        i = 2
        v1 = 2.0*u[i-1] - u[i]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = u[i+2]
        f[i] = wcR(v1,v2,v3,v4,v5)


        for i = 3:n-1
            v1 = u[i-2]
            v2 = u[i-1]
            v3 = u[i]
            v4 = u[i+1]
            v5 = u[i+2]
            f[i] = wcR(v1,v2,v3,v4,v5)
        end

        i = n
        v1 = u[i-2]
        v2 = u[i-1]
        v3 = u[i]
        v4 = u[i+1]
        v5 = 2.0*u[i+1] - u[i]
        f[i] = wcR(v1,v2,v3,v4,v5)

        i = n+1
        v1 = u[i-2]
        v2 = u[i-1]
        v3 = u[i]
        v4 = 2.0*u[i] - u[i-1]
        v5 = 3.0*u[i] - 2.0*u[i-1]
        f[i] = wcR(v1,v2,v3,v4,v5)

    end

    """
    nonlinear weights for upwind direction
    """
    function wcL(v1,v2,v3,v4,v5)
        eps = 1.0e-6

        # smoothness indicators
        s1 = (13.0/12.0)*(v1-2.0*v2+v3)^2 + 0.25*(v1-4.0*v2+3.0*v3)^2
        s2 = (13.0/12.0)*(v2-2.0*v3+v4)^2 + 0.25*(v2-v4)^2
        s3 = (13.0/12.0)*(v3-2.0*v4+v5)^2 + 0.25*(3.0*v3-4.0*v4+v5)^2

        # computing nonlinear weights w1,w2,w3
        c1 = 1.0e-1/((eps+s1)^2)
        c2 = 6.0e-1/((eps+s2)^2)
        c3 = 3.0e-1/((eps+s3)^2)

        w1 = c1/(c1+c2+c3)
        w2 = c2/(c1+c2+c3)
        w3 = c3/(c1+c2+c3)

        # candiate stencils
        q1 = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3
        q2 =-v2/6.0 + 5.0/6.0*v3 + v4/3.0
        q3 = v3/3.0 + 5.0/6.0*v4 - v5/6.0

        # reconstructed value at interface
        f = (w1*q1 + w2*q2 + w3*q3)

        return f

    end

    """
    nonlinear weights for downwind direction
    """
    function wcR(v1,v2,v3,v4,v5)
        eps = 1.0e-6

        s1 = (13.0/12.0)*(v1-2.0*v2+v3)^2 + 0.25*(v1-4.0*v2+3.0*v3)^2
        s2 = (13.0/12.0)*(v2-2.0*v3+v4)^2 + 0.25*(v2-v4)^2
        s3 = (13.0/12.0)*(v3-2.0*v4+v5)^2 + 0.25*(3.0*v3-4.0*v4+v5)^2

        c1 = 3.0e-1/(eps+s1)^2
        c2 = 6.0e-1/(eps+s2)^2
        c3 = 1.0e-1/(eps+s3)^2

        w1 = c1/(c1+c2+c3)
        w2 = c2/(c1+c2+c3)
        w3 = c3/(c1+c2+c3)

        # candiate stencils
        q1 =-v1/6.0      + 5.0/6.0*v2 + v3/3.0
        q2 = v2/3.0      + 5.0/6.0*v3 - v4/6.0
        q3 = 11.0/6.0*v3 - 7.0/6.0*v4 + v5/3.0

        # reconstructed value at interface
        f = (w1*q1 + w2*q2 + w3*q3)

        return f
    end
end


"""
Module to analyse result
"""
module Analysis
    """
    Compute error of final field
    """
    function calc_finalerror(par,var)
        var.error = var.uₙ₊₁[:] - var.uₑ
    end

    """
    Compute L-2 norm of a vector
    """
    function calc_l2norm(nx,r)
        rms = 0.0
        for ix = 2:nx
            rms += r[ix]^2
        end
        rms = sqrt(rms/(nx-1))
        return rms
    end
end  # Analysis


"""
Module to handle output
"""
module Output
using Printf

    """
    Output final field
    """
    function out_finalfield(par,var)
        solution = open("solution_d.dat", "w")

        for i = 1:par.nx+1
            write(solution, string(var.x[i]), " ",)
            for j = 1:par.ns
                write(solution, string(var.u[i,j]), " ")
            end
            write(solution, "\n",)
        end

        close(solution)
    end
end  # Output


# ====================
# Main
# ====================

## Declare modules
using .ParamVar
using .InviscidBergers1D_WENO5_Dirichlet:
numerical
using .Analysis
using .Output

# --------------------
## Set parameters
# --------------------
nx = 200
ns = 10
dt = 0.0001
tm = 0.25
dx = 1.0/nx
nt = Int64(tm/dt)
ds = tm/ns
param_ = ParamVar.Parameters(nx,ns,dt,tm,dx,nt,ds)

# --------------------
## Set variables
# --------------------
x = Array(0:param_.dx:1.0)
u = Array{Float64}(undef, param_.nx+1, param_.ns+1)

var_ = ParamVar.Variables(x,u)

# --------------------
## Compute Time iteration
# --------------------
numerical(param_.nx,param_.ns,param_.nt,param_.dx,param_.dt,var_.u)

# --------------------
## Output final state
# --------------------
Output.out_finalfield(param_,var_)
