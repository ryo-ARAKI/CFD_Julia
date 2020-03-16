#=
Julia program to solve 2D Poisson equation by conjugate gradient method based iterative solver
=#


"""
Module for parameters and variables
"""
module ParamVar
    struct Parameters
        ipr::Int64
        nx::Int64
        ny::Int64
        tolerance::Float64
        max_iter::Float64
        tiny::Float64
        #
        x_l::Float64
        x_r::Float64
        y_b::Float64
        y_t::Float64
        #
        dx::Float64
        dy::Float64
        #
        c1::Float64
        c2::Float64
    end

    mutable struct Variables
        x::Array{Float64}
        y::Array{Float64}
        u_e::Array{Float64, 2}
        f::Array{Float64, 2}
        u_n::Array{Float64, 2}
    end
end  # ParamVar


"""
Module for calculation
"""
module Poisson2D_CGM
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
    function compute_exact_solution(par, var)

        for i = 1:par.nx+1 for j = 1:par.ny+1

            if par.ipr == 1
                var.u_e[i,j] = (var.x[i]^2 - 1.0)*(var.y[j]^2 - 1.0)

                var.f[i,j]  = -2.0*(2.0 - var.x[i]^2 - var.y[j]^2)

                var.u_n[i,j] = 0.0
            end

            if par.ipr == 2
                var.u_e[i,j] = sin(2.0*pi*var.x[i]) * sin(2.0*pi*var.y[j]) +
                        par.c1*sin(16.0*pi*var.x[i]) * sin(16.0*pi*var.y[j])

                var.f[i,j] = 4.0*par.c2*sin(2.0*pi*var.x[i]) * sin(2.0*pi*var.y[j]) +
                        par.c2*sin(16.0*pi*var.x[i]) * sin(16.0*pi*var.y[j])

                var.u_n[i,j] = 0.0
            end
        end end
    end

    """
    Ensure periodic boundary condition
    """
    function compute_periodic_solution(nx, ny, u_e, u_n)
        u_n[:,1] = u_e[:,1]
        u_n[:, ny+1] = u_e[:, ny+1]

        u_n[1,:] = u_e[1,:]
        u_n[nx+1,:] = u_e[nx+1,:]
    end


    """
    Compute residual
    """
    function compute_residual(nx, ny, dx, dy, f, u_n, r)

        for j = 2:ny for i = 2:nx
            d2udx2 = (u_n[i+1,j] - 2*u_n[i,j] + u_n[i-1,j])/(dx^2)
            d2udy2 = (u_n[i,j+1] - 2*u_n[i,j] + u_n[i,j-1])/(dy^2)
            r[i,j] = f[i,j]  - d2udx2 - d2udy2
        end end

    end

    """
    Compute L-2 norm of a vector
    """
    function compute_l2norm(nx, ny, r)

        rms = 0.0
        # println(residual)
        for j = 2:ny for i = 2:nx
            rms = rms + r[i,j]^2
        end end
        # println(rms)
        rms = sqrt(rms/((nx-1)*(ny-1)))
        return rms
    end

    """
    Compute errors
    """
    function compute_error(par,var)
        uerror = Array{Float64}(undef, par.nx+1, par.ny+1)
        rms_error = 0.0

        uerror = var.u_n - var.u_e

        rms_error = compute_l2norm(par.nx, par.ny, uerror)
        max_error = maximum(abs.(uerror))

        return rms_error, max_error
    end

    """
    Solve 2D Poisson equation by conjugate gradient method
    """
    function conjugate_gradient(dx, dy, nx, ny, r, f, u_n, rms,
                        init_rms, max_iter, tolerance, tiny, output)

        # create text file for writing residual history
        residual_plot = open("cg_residual.txt", "w")
        #write(residual_plot, "k"," ","rms"," ","rms/rms0"," \n")

        count = 0.0

        compute_residual(nx, ny, dx, dy, f, u_n, r)

        rms = compute_l2norm(nx, ny, r)

        initial_rms = rms
        iteration_count = 0
        println(iteration_count, " ", initial_rms, " ", rms/initial_rms)
        # allocate the matric for direction and set the initial direction (conjugate vector)
        p = zeros(Float64, nx+1, ny+1)

        # asssign conjugate vector to initial residual
        for j = 1:ny+1 for i = 1:nx+1
            p[i,j] = r[i,j]
        end end

        del_p    = zeros(Float64, nx+1, ny+1)

        # start calculation
        for iteration_count = 1:max_iter

            # calculate ∇^2(residual)
            for j = 2:ny for i = 2:nx
                del_p[i,j] = (p[i+1,j] - 2.0*p[i,j] + p[i-1,j])/(dx^2) +
                            (p[i,j+1] - 2.0*p[i,j] + p[i,j-1])/(dy^2)
            end end

            aa = 0.0
            bb = 0.0
            # calculate aa, bb, cc. cc is the distance parameter(α_n)
            for j = 2:ny for i = 2:nx
                aa = aa + r[i,j]*r[i,j]
                bb = bb + del_p[i,j]*p[i,j]
            end end
            # cc = <r,r>/<d,p>
            cc = aa/(bb + tiny)

            # update the numerical solution by adding some component of conjugate vector
            for j = 2:ny for i = 2:nx
                u_n[i,j] = u_n[i,j] + cc*p[i,j]
            end end

            # bb = <r,r> = aa (calculated in previous loop)
            bb = aa
            aa = 0.0

            # update the residual by removing some component of previous residual
            for j = 2:ny for i = 2:nx
                r[i,j] = r[i,j] - cc*del_p[i,j]
                aa = aa + r[i,j]*r[i,j]
            end end
            # cc = <r-cd, r-cd>/<r,r>
            cc = aa/(bb+tiny)

            # update the conjugate vector
            for j = 1:ny for i = 1:nx
                p[i,j] = r[i,j] + cc*p[i,j]
            end end

            # compute the l2norm of residual
            rms = compute_l2norm(nx, ny, r)

            write(residual_plot, string(iteration_count), " ",string(rms), " ", string(rms/initial_rms)," \n");
            count = iteration_count

            println(iteration_count, " ", rms, " ", rms/initial_rms)

            if (rms/initial_rms) <= tolerance
                break
            end
        end

        write(output, "L-2 Norm = ", string(rms), " \n");
        write(output, "Maximum Norm = ", string(maximum(abs.(r))), " \n");
        write(output, "Iterations = ", string(count), " \n");
        close(residual_plot)
    end
end  # Poisson2D_CGM


"""
Module to handle output
"""
module Output
using Printf
    """
    Output field
    """
    function out_field(nx, ny, var, filename)
        out_field = open(filename, "w")
            for j = 1:ny+1 for i = 1:nx+1
                write(out_field, string(var.x[i]), " ",string(var.y[j]), " ", string(var.f[i,j]),
                    " ", string(var.u_n[i,j]), " ", string(var.u_e[i,j]), " \n")
            end end
        close(out_field)
    end

    """
    Output various data
    """
    function out_data(rms_error,max_error,t, output)
        # STDOUT
        println("Error details:");
        println("L-2 Norm = ", rms_error);
        println("Maximum Norm = ", max_error);
        print("CPU Time = ", t);

        # # txt file
        # output = open("output_512.txt", "w");

        write(output, "Error details: \n");
        write(output, "L-2 Norm = ", string(rms_error), " \n");
        write(output, "Maximum Norm = ", string(max_error), " \n");
        write(output, "CPU Time = ", string(t), " \n");
        # close(output)
    end
end  # Output


# ========================================
# Main
# ========================================

## Declare modules
using CPUTime
using .ParamVar
using .Poisson2D_CGM
using .Output


# ----------------------------------------
## Set parameters
# ----------------------------------------
ipr = 1
nx = 512
ny = 512
tolerance = 1.0e-10
max_iter = 100000
tiny = 1.0e-16

x_l = 0.0
x_r = 1.0
y_b = 0.0
y_t = 1.0

dx = (x_r - x_l)/nx
dy = (y_t - y_b)/ny

c1 = (1.0/16.0)^2
c2 = -2.0*pi*pi

par_ = ParamVar.Parameters(ipr,nx,ny,tolerance,max_iter,tiny,x_l,x_r,y_b,y_t,dx,dy,c1,c2)

# allocate array for x and y position of grids, exact solution and source term
x = Array{Float64}(undef, par_.nx+1)
y = Array{Float64}(undef, par_.ny+1)
u_e = Array{Float64}(undef, par_.nx+1, par_.ny+1)
f = Array{Float64}(undef, par_.nx+1, par_.ny+1)
u_n = Array{Float64}(undef, par_.nx+1, par_.ny+1)

var_ = ParamVar.Variables(x,y,u_e,f,u_n)

# --------------------
## Define coordinates
# --------------------
Poisson2D_CGM.set_coordinates(par_, var_.x, var_.y)

# --------------------
## Define exact solution
# --------------------
Poisson2D_CGM.compute_exact_solution(par_, var_)

# --------------------
## Ensure periodic boundary condition
# --------------------
Poisson2D_CGM.compute_periodic_solution(par_.nx, par_.ny, var_.u_e, var_.u_n)

# --------------------
## Output initial field
# --------------------
Output.out_field(par_.nx, par_.ny, var_, "field_initial.txt")

# create output file for L2-norm
output = open("output.txt", "w");
write(output, "Residual details: \n");


val, t, bytes, gctime, memallocs = @timed begin


r = zeros(Float64, nx+1, ny+1)
init_rms = 0.0
rms = 0.0
Poisson2D_CGM.conjugate_gradient(dx, dy, nx, ny, r, f, u_n, rms,
                      init_rms, max_iter, tolerance, tiny, output)

end

print("CPU time: ", t)

# --------------------
## Calculate errors
# --------------------
rms_error, max_error = Poisson2D_CGM.compute_error(par_, var_)

# --------------------
## Output data
# --------------------
Output.out_data(rms_error,max_error,t, output)

# --------------------
## Output final field
# --------------------
Output.out_field(par_.nx, par_.ny, var_, "field_final.txt")
close(output)
