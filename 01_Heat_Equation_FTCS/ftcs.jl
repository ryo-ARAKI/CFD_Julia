#=
Julia program to solve 1D Heat equation by FTCS (Forward Time Central Space) scheme
=#


"""
Module for parameters and variables
"""
module ParamVar
    struct Parameters
        xₗ::Float64  # Left end of the region
        xᵣ::Float64  # Right end of the region
        dx::Float64  # Space discretisation
        nₓ::Int64  # DoF of space
        dt::Float64  # Time discretisation
        t::Float64  # Time discretisation
        nₜ::Int64  # DoF of space
        α::Float64  # #####TBA Comment#####
        β::Float64  # #####TBA Comment#####
    end

    mutable struct Variables
        x::Array{Float64}  # Global coordinate
        uₑ::Array{Float64}  # Exact (analytical) solution at time=t
        uₙ::Array{Float64, 2}  # Numerical solution
        error::Array{Float64}  # L2 norm error
    end
end  # ParamVar


"""
Module for 1D heat equation with FTCS (Forward Time Central Space) scheme
"""
module HeatEq1D_FTCS

    """
    Set coordinates and initial conditions
    """
    function set_initial_condition(par,var)
        for ix = 1:par.nₓ+1
            var.x[ix] = par.xₗ + par.dx * (ix-1)
            var.uₙ[1, ix] = -sin(pi*var.x[ix])
            var.uₑ[ix] = -exp(-par.t) * sin(pi*var.x[ix])
        end
    end

    """
    Set boundary conditions at arbitrary time step
    """
    function set_boundary_condition(par,var,it)
        var.uₙ[it,1] = 0.0
        var.uₙ[it,par.nₓ+1] = 0.0
    end

    """
    Time march
    """
    function time_march(par,var)
        for it = 2:par.nₜ+1  # Time integration
            for ix = 2:par.nₓ  # FTCS scheme
                var.uₙ[it,ix] = var.uₙ[it-1,ix] + par.β * (
                    var.uₙ[it-1,ix+1] - 2.0 * var.uₙ[it-1,ix] + var.uₙ[it-1,ix-1]
                )
            end
            set_boundary_condition(par,var,it)
        end
    end
end  # HeatEq1D_FTCS


"""
Module to analyse result
"""
module Analysis
    """
    Compute error of final field
    """
    function calc_finalerror(par,var)
        var.error = var.uₙ[par.nₜ+1,:] - var.uₑ
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
    Output L2 norm error
    """
    function out_l2norm(rms_error)
        out = open("l2_error.dat", "w")
        write(out, "Error data \n")
        write(out, "L-2 norm = ", string(rms_error), " \n")
        write(out, "Maximum norm = ", string(maximum(abs.(rms_error))), " \n")
        close(out)
    end

    """
    Output final field
    """
    function out_finalfield(par,var)
        out = open("final_field.dat", "w")
        write(out, "x, u_exact, u_numerical, u_error \n")
        for ix = 1:par.nₓ+1
            write(out,
                @sprintf("%.16f", var.x[ix]), " ",
                @sprintf("%.16f", var.uₑ[ix]), " ",
                @sprintf("%.16f", var.uₙ[par.nₜ+1, ix]), " ",
                @sprintf("%.16f", var.error[ix]), " \n"
                )
        end
        close(out)
    end
end  # Output


# ====================
# Main
# ====================

## Declare modules
using .ParamVar
using .HeatEq1D_FTCS
using .Analysis
using .Output

# --------------------
## Set parameters
# --------------------
### Spatial parameters
xₗ = -1.0
xᵣ = 1.0
dx = 0.025
nₓ = Int64((xᵣ-xₗ)/dx)

### Temporal parameters
dt = 0.0025
t = 1.0
nₜ = Int64(t/dt)

### Problem-specific parameters
α = 1/(pi*pi)
β = α*dt/(dx*dx)

### Declare parameter module
param_ = ParamVar.Parameters(xₗ,xᵣ,dx,nₓ,dt,t,nₜ,α,β)

# --------------------
## Set variables
# --------------------
### Arrays
x = Array{Float64}(undef, param_.nₓ+1)
uₑ = Array{Float64}(undef, param_.nₓ+1)
uₙ = Array{Float64}(undef, param_.nₜ+1, param_.nₓ+1)
error = Array{Float64}(undef, param_.nₓ+1)

### Declare variable module
var_ = ParamVar.Variables(x,uₑ,uₙ,error)

# --------------------
## Compute coordinates, analytical final-state solution and initial condition
# --------------------
HeatEq1D_FTCS.set_initial_condition(param_, var_)
HeatEq1D_FTCS.set_boundary_condition(param_, var_,1)

# --------------------
## Compute Time iteration
# --------------------
HeatEq1D_FTCS.time_march(param_,var_)

# --------------------
## Compute L-2 norm error
# --------------------
Analysis.calc_finalerror(param_,var_)
rms_error = Analysis.calc_l2norm(param_.nₓ,var_.error)

# --------------------
## Output final state and error
# --------------------
Output.out_l2norm(rms_error)
Output.out_finalfield(param_,var_)
