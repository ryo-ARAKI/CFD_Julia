#=
Julia program to solve 2D Poisson equation by Finite difference fast Fourier transform (FFT) based direct solver
=#


"""
Module for parameters and variables
"""
module ParamVar
    struct Parameters
        nx::Int64
        ny::Int64
        x_l::Float64
        x_r::Float64
        y_b::Float64
        y_t::Float64
        dx::Float64
        dy::Float64
    end

    mutable struct Variables
        x::Array{Float64}
        y::Array{Float64}
        ue::Array{Float64, 2}
        f::Array{Float64, 2}
        un::Array{Float64, 2}
    end
end  # ParamVar


"""
Module for calculation
"""
module Poisson2D_FFT
    """
    Set 2D coordinates
    """
    function set_coordinates(par, x, y)
        for ix = 1:par.nx+1
            x[ix] = par.x_l + par.dx*(ix-1)
        end
        for iy = 1:par.ny+1
            y[iy] = par.y_b + par.dy*(iy-1)
        end
    end

    """
    Calculate exact solution
    """
    function compute_exact_solution(par, x, y, ue, f, un)
        km = 16
        c1 = (1.0/km)^2
        c2 = -8.0*pi*pi

        for j = 1:par.ny+1 for i = 1:par.nx+1
            ue[i,j] = sin(2.0*pi*x[i])*sin(2.0*pi*y[j]) +
                c1*sin(km*2.0*pi*x[i])*sin(km*2.0*pi*y[j])

            f[i,j] = c2*sin(2.0*pi*x[i])*sin(2.0*pi*y[j]) +
                    c2*sin(km*2.0*pi*x[i])*sin(km*2.0*pi*y[j])

            un[i,j] = 0.0
        end end
    end

    """
    Ensure periodic boundary condition
    """
    function compute_periodic_solution(par, un)
        un[par.nx+1,:] = un[1,:]
        un[:,par.ny+1] = un[:,1]
    end
end  # Poisson2D_FFT


"""
Module to analyse result
"""
module Analysis

    """
    Compute L-2 norm of a vector
    """
    function compute_l2norm(nx, ny, r)
        rms = 0.0
        # println(residual)
        for j = 1:ny+1 for i = 1:nx+1
            rms = rms + r[i,j]^2
        end end
        rms = sqrt(rms/((nx+1)*(ny+1)))
        return rms
    end

    """
    Compute errors
    """
    function compute_error(par,var)
        uerror = Array{Float64}(undef, par.nx+1, par.ny+1)
        rms_error = 0.0

        uerror = var.un - var.ue

        rms_error = compute_l2norm(par.nx, par.ny, uerror)
        max_error = maximum(abs.(uerror))

        return rms_error, max_error
    end
end  # Analysis


"""
Module to handle output
"""
module Output
using Printf
    """
    Output field
    """
    function out_field(par, var, filename)
        field_initial = open(filename, "w")
        for j = 1:par.ny+1 for i = 1:par.nx+1
            write(field_initial, string(var.x[i]), " ",string(var.y[j]), " ", string(var.f[i,j]),
                " ", string(var.un[i,j]), " ", string(var.ue[i,j]), " \n")
        end end
        close(field_initial)
    end

    """
    Output various data
    """
    function out_data(rms_error,max_error,t)
        # STDOUT
        println("Error details:");
        println("L-2 Norm = ", rms_error);
        println("Maximum Norm = ", max_error);
        print("CPU Time = ", t);

        # txt file
        output = open("output_512.txt", "w");

        write(output, "Error details: \n");
        write(output, "L-2 Norm = ", string(rms_error), " \n");
        write(output, "Maximum Norm = ", string(max_error), " \n");
        write(output, "CPU Time = ", string(t), " \n");
        close(output)
    end
end  # Output


"""
Solve 2D Poisson equation by FFT
"""
function ps_spectral(nx,ny,dx,dy,f)
    eps = 1.0e-6

    kx = Array{Float64}(undef,nx)
    ky = Array{Float64}(undef,ny)

    data = Array{Complex{Float64}}(undef,nx,ny)
    data1 = Array{Complex{Float64}}(undef,nx,ny)
    e = Array{Complex{Float64}}(undef,nx,ny)

    u = Array{Complex{Float64}}(undef,nx,ny)

    #wave number indexing
    hx = 2.0*pi/(nx*dx)

    for i = 1:Int64(nx/2)
        kx[i] = hx*(i-1.0)
        kx[i+Int64(nx/2)] = hx*(i-Int64(nx/2)-1)
    end
    kx[1] = eps

    ky = kx

    for j = 1:ny for i = 1:nx
        data[i,j] = complex(f[i,j],0.0)
    end end

    e = fft(data)
    e[1,1] = 0.0
    for j = 1:ny for i = 1:nx
        data1[i,j] = e[i,j]/(-kx[i]^2 -ky[j]^2)
    end end

    u = real(ifft(data1))

    return u
end


# ====================
# Main
# ====================

## Declare modules
using FFTW
using CPUTime
using .ParamVar
using .Poisson2D_FFT
using .Analysis
using .Output

# --------------------
## Set parameters
# --------------------
nx = 512
ny = 512

x_l = 0.0
x_r = 1.0
y_b = 0.0
y_t = 1.0

dx = (x_r-x_l)/nx
dy = (y_t-y_b)/ny

par_ = ParamVar.Parameters(nx,ny,x_l,x_r,y_b,y_t,dx,dy)

# --------------------
## Set variables
# --------------------
x = Array{Float64}(undef, par_.nx+1)
y = Array{Float64}(undef, par_.ny+1)
ue = Array{Float64}(undef, par_.nx+1, par_.ny+1)
f  = Array{Float64}(undef, par_.nx+1, par_.ny+1)
un = Array{Float64}(undef, par_.nx+1, par_.ny+1)

var_ = ParamVar.Variables(x,y,ue,f,un)

# --------------------
## Define coordinates
# --------------------
Poisson2D_FFT.set_coordinates(par_, var_.x, var_.y)

# --------------------
## Define exact solution
# --------------------
Poisson2D_FFT.compute_exact_solution(par_, var_.x, var_.y, var_.ue, var_.f, var_.un)

# --------------------
## Output initial field
# --------------------
Output.out_field(par_, var_, "field_initial.txt")

# --------------------
## Begin time count
# --------------------
val, t, bytes, gctime, memallocs = @timed begin

# --------------------
## Solve Poisson equation
# --------------------
var_.un[1:par_.nx,1:par_.ny] = ps_spectral(par_.nx,par_.ny,par_.dx,par_.dy,var_.f)

end

# --------------------
## Ensure periodic boundary condition
# --------------------
Poisson2D_FFT.compute_periodic_solution(par_, var_.un)

# --------------------
## Calculate errors
# --------------------
rms_error, max_error = Analysis.compute_error(par_, var_)

# --------------------
## Output data
# --------------------
Output.out_data(rms_error,max_error,t)

# --------------------
## Output final field
# --------------------
Output.out_field(par_, var_, "field_final_512.txt")
