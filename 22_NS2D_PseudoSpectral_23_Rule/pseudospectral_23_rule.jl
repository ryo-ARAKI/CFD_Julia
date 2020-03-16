#=
Julia program to solve vortex merging problem defined in 2D Incompressible Navier-Stokes equation by pseudospectral method with 2/3 dealiasing rule
=#


"""
Module for parameters and variables
"""
module ParamVar
    struct Parameters
        # Spectral size of the domain
        nx::Int64
        ny::Int64
        # Physical size of domain
        x_l::Float64
        x_r::Float64
        y_b::Float64
        y_t::Float64
        dx::Float64
        dy::Float64
        # Temporal parameters
        dt::Float64
        tf::Float64
        nt::Float64
        # Other parameters
        Re::Float64
        ns::Float64
    end

    mutable struct Variables
        # Global coordinates
        x::Array{Float64}
        y::Array{Float64}
        # Physical values
        wn::Array{Float64, 2}
        un::Array{Float64, 2}
    end
end  # ParamVar


"""
Module for calculation
"""
module NS_2D_PseudoSpectral_23Rule
using FFTW
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
    Set initial conditon for vortex merger problem
    """
    # initial condition
    function compute_vortex_initial_condition(nx,ny,x,y,w)
        sigma = pi
        xc1 = pi-pi/4.0
        yc1 = pi
        xc2 = pi+pi/4.0
        yc2 = pi

        for ix = 2:nx+2 for iy = 2:ny+2
            w[ix,iy] = exp(-sigma*((x[ix-1]-xc1)^2 + (y[iy-1]-yc1)^2)) +
                    exp(-sigma*((x[ix-1]-xc2)^2 + (y[iy-1]-yc2)^2))
        end end
    end

    """
    Ensure periodic boundary condition
    """
    function compute_periodic_solution(nx, ny, w)
        w[1,:] = w[nx+1,:]
        w[:,1] = w[:,ny+1]

        w[nx+2,:] = w[2,:]
        w[:,ny+2] = w[:,2]
    end


    """
    TBA
    """
    function wavespace(nx,ny,dx,dy)
        eps = 1.0e-6

        kx = Array{Float64}(undef,nx)
        ky = Array{Float64}(undef,ny)

        k2 = Array{Float64}(undef,nx,ny)

        #wave number indexing
        hx = 2.0*pi/(nx*dx)
        hy = 2.0*pi/(ny*dy)

        for ix = 1:Int64(nx/2)
            kx[ix] = hx*(ix-1.0)
            kx[ix+Int64(nx/2)] = hx*(ix-Int64(nx/2)-1)
        end
        for iy = 1:Int64(ny/2)
            ky[iy] = hy*(iy-1.0)
            ky[iy+Int64(ny/2)] = hy*(iy-Int64(ny/2)-1)
        end
        kx[1], ky[1] = eps, eps

        for ix = 1:nx for iy = 1:ny
            k2[ix,iy] = kx[ix]^2 + ky[iy]^2
        end end

        return k2
    end


    """
    Calculate Jacobian in fourier space
    jf = -J(w,ψ)
    """
    function jacobian(nx,ny,dx,dy,wf,k2)
        eps = 1.0e-6
        kx = Array{Float64}(undef,nx)
        ky = Array{Float64}(undef,ny)

        #wave number indexing
        hx = 2.0*pi/(nx*dx)

        for ix = 1:Int64(nx/2)
            kx[ix] = hx*(ix-1.0)
            kx[ix+Int64(nx/2)] = hx*(ix-Int64(nx/2)-1)
        end
        kx[1] = eps
        ky = transpose(kx)

        j1f = zeros(ComplexF64,nx,ny)
        j2f = zeros(ComplexF64,nx,ny)
        j3f = zeros(ComplexF64,nx,ny)
        j4f = zeros(ComplexF64,nx,ny)

        # x-derivative
        for ix = 1:nx for iy = 1:ny
            j1f[ix,iy] = 1.0im*wf[ix,iy]*kx[ix]/k2[ix,iy]
            j4f[ix,iy] = 1.0im*wf[ix,iy]*kx[ix]
        end end

        # y-derivative
        for ix = 1:nx for iy = 1:ny
            j2f[ix,iy] = 1.0im*wf[ix,iy]*ky[iy]
            j3f[ix,iy] = 1.0im*wf[ix,iy]*ky[iy]/k2[ix,iy]
        end end

        nxe = Int64(floor(nx*2/3))
        nye = Int64(floor(ny*2/3))

        for ix = Int64(floor(nxe/2)+1):Int64(nx-floor(nxe/2)) for iy = 1:ny
            j1f[ix,iy] = 0.0
            j2f[ix,iy] = 0.0
            j3f[ix,iy] = 0.0
            j4f[ix,iy] = 0.0
        end end

        for ix = 1:nx for iy = Int64(floor(nye/2)+1):Int64(ny-floor(nye/2))
            j1f[ix,iy] = 0.0
            j2f[ix,iy] = 0.0
            j3f[ix,iy] = 0.0
            j4f[ix,iy] = 0.0
        end end

        j1 = real(ifft(j1f))
        j2 = real(ifft(j2f))
        j3 = real(ifft(j3f))
        j4 = real(ifft(j4f))
        jacp = zeros(Float64,nx,ny)

        for ix = 1:nx for iy = 1:ny
            jacp[ix,iy] = j1[ix,iy]*j2[ix,iy] - j3[ix,iy]*j4[ix,iy]
        end end

        jf = fft(jacp)

        return jf
    end


    """
    Compute numerical solution
        - Time integration using Runge-Kutta third order
        - 2nd-order finite difference discretization
    """
    function numerical(nx,ny,nt,dx,dy,dt,x,y,re,wn,ns,Output)

        # Intermidiate vortex field for RK3 scheme
        w1f = Array{Complex{Float64}}(undef,nx,ny)
        w2f = Array{Complex{Float64}}(undef,nx,ny)
        wnf = Array{Complex{Float64}}(undef,nx,ny)

        # Intermidiate Jacobian field for RK3 scheme
        j1f = Array{Complex{Float64}}(undef,nx,ny)
        j2f = Array{Complex{Float64}}(undef,nx,ny)
        jnf = Array{Complex{Float64}}(undef,nx,ny)

        ut = Array{Float64}(undef, nx+1, ny+1)  # Output field
        wm_cmp = Array{Complex{Float64}}(undef, nx, ny)
        d1 = Array{Float64}(undef, nx, ny)
        d2 = Array{Float64}(undef, nx, ny)
        d3 = Array{Float64}(undef, nx, ny)

        k2 = Array{Float64}(undef, nx, ny)

        freq_out = Int64(nt/ns)  # Output frequency
        m = 1 # record index

        for ix = 1:nx for iy = 1:ny
            wm_cmp[ix,iy] = complex(wn[ix+1,iy+1],0.0)
        end end
        # wm_cmp[1,1] = undef, but it will be overlapped by wnf[1,1] = 0.0

        wnf = fft(wm_cmp)
        k2 = wavespace(nx,ny,dx,dy)
        wnf[1,1] = 0.0

        # Constants for RK3 scheme
        alpha1, alpha2, alpha3 = 8.0/15.0, 2.0/15.0, 1.0/3.0
        gamma1, gamma2, gamma3 = 8.0/15.0, 5.0/12.0, 3.0/4.0
        rho2, rho3 = -17.0/60.0, -5.0/12.0

        for ix = 1:nx for iy = 1:ny
            z = 0.5*dt*k2[ix,iy]/re
            d1[ix,iy] = alpha1*z
            d2[ix,iy] = alpha2*z
            d3[ix,iy] = alpha3*z
        end end

        # Time iteration
        for itr_step = 1:nt
            jnf = jacobian(nx,ny,dx,dy,wnf,k2)

            # 1st step
            for ix = 1:nx for iy = 1:ny
                w1f[ix,iy] = ((1.0 - d1[ix,iy])/(1.0 + d1[ix,iy]))*wnf[ix,iy] +
                            (gamma1*dt*jnf[ix,iy])/(1.0 + d1[ix,iy])
            end end

            w1f[1,1] = 0.0
            j1f = jacobian(nx,ny,dx,dy,w1f,k2)

            # 2nd step
            for ix = 1:nx for iy = 1:ny
                w2f[ix,iy] = ((1.0 - d2[ix,iy])/(1.0 + d2[ix,iy]))*w1f[ix,iy] +
                            (rho2*dt*jnf[ix,iy] + gamma2*dt*j1f[ix,iy])/(1.0 + d2[ix,iy])
            end end

            w2f[1,1] = 0.0
            j2f = jacobian(nx,ny,dx,dy,w2f,k2)

            # 3rd step
            for ix = 1:nx for iy = 1:ny
                wnf[ix,iy] = ((1.0 - d3[ix,iy])/(1.0 + d3[ix,iy]))*w2f[ix,iy] +
                            (rho3*dt*j1f[ix,iy] + gamma3*dt*j2f[ix,iy])/(1.0 + d3[ix,iy])
            end end

            if (mod(itr_step,freq_out) == 0)
                println(itr_step)
                ut[1:nx,1:ny] = real(ifft(wnf))
                # periodic BC
                ut[nx+1,:] = ut[1,:]
                ut[:,ny+1] = ut[:,1]
                Output.out_field(
                    nx, ny,
                    x, y,
                    ut,
                    string("vm",string(m),".txt"))
                m = m+1
            end
        end

        # return ut
    end
end  # NS_2D_PseudoSpectral_23Rule


"""
Module to handle output
"""
module Output
using Printf
    """
    Output field
    """
    function out_field(nx, ny, x, y, val, filename)
        out_field = open(filename, "w")
            for iy = 1:ny+1 for ix = 1:nx+1
                write(out_field, string(x[ix]), " ",string(y[iy]), " ", string(val[ix,iy]), " \n")
            end end
        close(out_field)
    end
end  # Output


# ========================================
# Main
# ========================================

## Declare modules
using CPUTime
using Plots
using .ParamVar
using .NS_2D_PseudoSpectral_23Rule:
set_coordinates,
compute_vortex_initial_condition,
compute_periodic_solution,
numerical
using .Output

font = Plots.font("Times New Roman", 18)
pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)

# ----------------------------------------
## Set parameters & variables
# ----------------------------------------
### Spectral size of the domain
nx = 128
ny = 128

### Physical size of domain
x_l = 0.0
x_r = 2.0*pi
y_b = 0.0
y_t = 2.0*pi
dx = (x_r-x_l)/nx
dy = (y_t-y_b)/ny

### Temporal parameters
dt = 0.01
tf = 20.0
nt = tf/dt

### Other parameters
Re = 1000.0
ns = 10  # Number of mid-steps to output .txt files

### Declare parameters
par_ = ParamVar.Parameters(
    nx,ny,
    x_l,x_r,y_b,y_t,dx,dy,
    dt,tf,nt,
    Re,ns)

### Global coordinates
x = Array{Float64}(undef, par_.nx+1)
y = Array{Float64}(undef, par_.ny+1)

### Physical values
wn = Array{Float64}(undef, par_.nx+2, par_.ny+2)
un = Array{Float64}(undef, par_.nx+1, par_.ny+1)

### Declare arrays
var_ = ParamVar.Variables(
    x,y,
    wn,un)


# ----------------------------------------
## Define coordinates
# ----------------------------------------
set_coordinates(par_, var_.x, var_.y)


# ----------------------------------------
## Set initial condition for vortex
# ----------------------------------------
compute_vortex_initial_condition(
    par_.nx, par_.ny,
    var_.x, var_.y,
    var_.wn)


# ----------------------------------------
## Ensure periodic boundary condition
# ----------------------------------------
compute_periodic_solution(
    par_.nx, par_.ny,
    var_.wn)


# ----------------------------------------
## Output initial field
# ----------------------------------------
Output.out_field(
    par_.nx, par_.ny,
    var_.x, var_.y,
    var_.wn[2:par_.nx+2,2:par_.ny+2],
    "vm0.txt")


# ----------------------------------------
## Compute time integration & output data
# ----------------------------------------
numerical(
    par_.nx, par_.ny, par_.nt,
    par_.dx, par_.dy, par_.dt,
    var_.x, var_.y,
    par_.Re, var_.wn, par_.ns,
    Output)
