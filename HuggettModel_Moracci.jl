# #################################################################
# Huggett (1996) model: life cycle with heterogeneous agents
# Elia Moracci - elia.moracci@eui.eu
# Project for the Household Dynamic Choice course - Winter 2020
####################################################################

## Housekeeping

# Set command directory
cd("/Users/eliam/OneDrive - Istituto Universitario Europeo/Coursework/Advanced courses/Household Dynamic Choice/ProjectHuggett")
# Import packages
using QuantEcon, DelimitedFiles, Plots, Interpolations, Random, StatsBase, Parameters, Statistics
# Close plots
Plots.closeall

## Defining parameters

# Structure that assign types to parameters
struct parstruct
    nkk::Integer
    nzz::Integer
    J::Integer
    Jr::Integer
    a_grid::Vector{Float64}
    z_grid::Vector{Float64}
    α::Float64
    β::Float64
    δ::Float64
    σ::Float64
    ω::Float64
    n::Float64
    Π::Array{Float64,2}
    Πstat::Vector{Float64}
    DiscretizedMC::MarkovChain
    ψ::Vector{Float64}
    s::Vector{Float64}
    e::Array{Float64,2}
    L::Float64
    θ::Float64
    λ::Vector{Float64}
end

# Function that sets parameters
function parameters(
                    # Pass all input after the semicolon, position does not matter.
                    ;
                    nkk::Integer   = 100,   # Points in asset grid.
                    nzz::Integer   = 5,    # Points in productivity shock grid
                    J::Integer     = 71,    # number of age groups
                    Jr::Integer    = 41,    # period of retirement
                    α::Float64     = 0.36,  # capital share
                    β::Float64     = 0.96,  # discount factor
                    δ::Float64     = 0.08,  # depreciation rate
                    σ::Float64     = 2.0,   # risk aversion
                    ω::Float64     = 0.5,   # replacement rate
                    n::Float64     = 0.01,  # growth rate of new cohort
                    k_lb::Float64  = 0.0,   # lower bound of asset grid
                    k_ub::Float64  = 100.0, # upper bound of asset grid
                    ρ_z::Float64   = 0.9,   # persistence of productivity shock
                    σ_z::Float64   = 0.2,   # variance of error term of AR1
                    spread::Integer = 2)    # bounds of prodcutivity grid in terms of standard deviations

    # Grid for capital holdings
    a_grid = collect(range(k_lb,k_ub,length=nkk))

    # Construct grid for productivity
    DiscretizedMC = QuantEcon.tauchen(nzz,ρ_z,σ_z,0.0,spread)
    aux           = QuantEcon.stationary_distributions(DiscretizedMC)
    Πstat         = aux[1]
    log𝐳          = DiscretizedMC.state_values
    Π             = DiscretizedMC.p
    𝐳             = exp.(log𝐳)
    mean𝐳         = sum(𝐳.*Πstat)
    z_grid        = 𝐳./mean𝐳;
    #avgz         = sum(𝐳.*Π_stat) # It is equal to one, okay.

    # Load survival probabilities and compute population fractions
    aux2     = readdlm("LifeTables.txt",Float64)   # Read surv. prob. table
    s        = aux2[:,1]                           # Store survival probabilities
    N        = Array{Float64}(undef,J)             # Pop. size by age group
    N[1] = 1.0
    for j in 2:J
        N[j] = s[j-1]*N[j-1]/(1+n)
    end
    ψ = N ./ sum(N)                                # Pop. shares by group

    # Create vector of efficiency units of labor for each (j,z) pair
    λ = zeros(J)                               # Avg. life-cycle income profile
    for j in 1:Jr-1
        λ[j] = 0.195 + 0.107*j - 0.00213*j^2     # Fill it with provided polynomial
    end
    e = zeros(nzz,J)                             # Efficiency units of labor
    for j in 1:Jr-1
        for z in 1:nzz
            e[z,j] = z_grid[z] * λ[j]
        end
    end

    # Labour supply
    L = sum(ψ.*λ)                                # Scalar prod. of pop. shares and avg. inc.

    # Payroll tax
    θ = ω*sum(ψ[Jr:end])/sum(ψ[1:Jr-1])

    # Assign types to relevant parameters (thanks, Andrej)
    return parstruct(nkk, nzz, J, Jr, a_grid, z_grid, α, β, δ, σ, ω, n, Π, Πstat, DiscretizedMC, ψ, s, e, L, θ, λ)
end

## Main functions

# Function that computes K, w and b for given r
function CapitalWagesPensions(param::parstruct,r::Float64)
    @unpack α, δ, L, θ, ψ, Jr, J = param
    K =  ((r+δ)/(α))^(1/(α-1))*L
    w = (1 - α) * (α / (r + δ))^(α / (1 - α))
    b          = zeros(J)
    b[Jr:end] .= θ*w*L/(sum(ψ[Jr:end]))
    return K, w, b
end

# Function that solves the HH problem with bisection method for given r, T, w, b
function HH_Bisec(
    param::parstruct,
    r::Float64,
    T::Float64,
    w::Float64,
    b::Vector{Float64})
    # Extract needed parameters from param. structure
    @unpack a_grid, z_grid, nkk, nzz, J, e, σ, β, Π, s, θ   = param
    # Preallocate household's policy function
    g_pol = zeros(nzz, nkk, J)
    # Loop over (z,k,j)
    for j = J-1:-1:1
        for k = 1:nkk
            for z = 1:nzz
                # Compute income (except asset income) this period and next period
                y  = (1-θ)*e[z,j]*w+b[j]+T                      # 1×1
                y′ = (1-θ)*e[:,j+1]*w .+ b[j+1] .+T             # z×1
                # For each (k,z,j), I have to find g(z,k,j)
                err = 1                     # Set large bisection error.
                mina = 0.0                  # Bisection bounds for the position of a'
                maxa = a_grid[nkk]
                maxiter = 50                # Maximum number of iterations
                iter = 0                    # Initialize iterations
                # Bisection "while" loop
                while abs(err) > tol && iter<maxiter
                    iter += 1                                  # Update iterations
                    a′=0.5*mina+0.5*maxa                       # Update optimal a′
                    # Have to compute the implied vector a′′ via g(z,a′,j+1) [known]
                    a′′ = zeros(nzz,1)
                    for z = 1 : nzz
                        itp = interpolate((a_grid,),g_pol[z,:,j+1],Gridded(Linear()))
                        a′′[z] = itp(a′)
                    end

                    c = y + (1+r)*a_grid[k]-a′
                    if c<0
                        err = 10000
                    else
                    err = c^(-σ)-s[j]*β*(1+r)*(Π[z,:]*((y′ .+ ((1+r)*a′-a′′)).^(-σ)))
                    end
                    # Update bisection error
                    e = abs(euler)
                    # Shift asset upper bound to the left if want to consume more today ---> a' has to go down.
                    if euler > 0
                        maxa = a′
                    # Shift asset lower bound to the right if want to consume less today ---> a' has to go up.
                    elseif euler < 0
                        mina = a′
                    end

                    if a′ > a_grid[nkk]
                       g_pol[z,k,j] = a_grid[nkk]
                    elseif a_guess < 0
                       g_pol[z,k,j] = 0.0
                    else
                    # optimal value for policy function
                    g_pol[z,k,j] = a′
                    end

                end
                # End of bisection "while" loop
            end
        end
    end
    # Return household's saving policy function
    return g_pol
end

# Function that solves the HH problem with EGM for given r, T, w, b
function HH_EGM(
    param::parstruct,
    r::Float64,
    T::Float64,
    w::Float64,
    b::Vector{Float64})
    @unpack a_grid, z_grid, nkk, nzz, J, e, σ, β, Π, s, θ  = param
    g = zeros(nzz,nkk,J)
    # Compute policy function from J-1:-1:1 (nosavings in last period J)
    for j in J-1:-1:1
        a_star = zeros(nzz,nkk)
        for k in 1:nkk
            for z in 1:nzz
                # Compute income (except asset income) this period and next period
                y  = (1-θ)*e[z,j]*w+b[j]+T'                     # 1×1
                y′ = (1-θ)*e[:,j+1]*w .+ b[j+1] .+T             # z×1

                # Calculate a consistent with a′ and g(z′,a′,j+1),
                a_star[z,k] = ((s[j]*β*(1+r)*(Π[z,:]'*((y′ .+ (1+r)*a_grid[k] - g[:,k,j+1]).^(-σ))))^(-1/σ) - y + a_grid[k])/(1+r)
            end
        end

        # interpolate to obtain policy function
        for k in 1:nkk
            for z in 1:nzz
                nodes = (a_star[z,:],)
                itp  = interpolate(nodes,a_grid,Gridded(Linear()))
                etpf = extrapolate(itp,Line())
                g[z,k,j] = etpf(a_grid[k])

                # corner solutions
                if g[z,k,j] < 0
                   g[z,k,j] = 0.0
                end
            end
        end
    end
    return g
end

# Function that simulates assets and bequests of a N×J-panel of individuals (get K and T)
function SimulateAssetsBequests(param::parstruct,g::Array{Float64},N::Integer)
    @unpack nzz, J, a_grid, s, ψ, DiscretizedMC, Πstat  = param
    # Preallocate
    z_path_indices = zeros(J,N)
    a_path = zeros(J,N)
    for n in 1:N
        z0=StatsBase.sample(Weights(Πstat))
        z_path_indices[:,n] = QuantEcon.simulate_indices(DiscretizedMC::MarkovChain,J::Int64;init=z0)
    end
    # Path of shock indices
    z_path_indices= convert(Array{Integer,2}, z_path_indices)
    # Everybody is born with zero assets
    a_path[1,:] .= 0.0
    # Now we iterate using policy function g and path for z.
    # For any age
    for j in 2:J
        # For any person in the panel
        for n in 1:N
            # Interpolate the policy function to extend it to points outside the grid to allow assets to stay outside.
            nodes = (a_grid,)
            itp = interpolate(nodes,g[z_path_indices[j-1,n],:,j-1],Gridded(Linear()))
            assets_itp = itp(a_path[j-1,n])
            a_path[j,n] = assets_itp
        end
    end
    # Compute average assets for each age
    avg_a = [mean(a_path[i,:]) for i in 1:J]
    # Compute total bequests
    # T = ∑(j) (1-s(j))⋅(ψ(j))⋅̄a(j)
    T′ = sum((1 .- s).*ψ.*avg_a)
    K′ = sum(avg_a.* ψ)
    return T′, a_path, K′
end

# Function that computes Gini coeff. for the panel
function GiniCoeff(a_path::Array{Float64})
    naa=length(a_path)
    a_path_sorted = sort(a_path)
    gini = 2*(sum(collect(1:naa).*a_path_sorted))/(naa*sum(a_path_sorted))-1
end

# Function that finds the equilibrium bequest T for any r
function T_bisec(
    param::parstruct,
    r::Float64,
    w::Float64,
    b::Vector{Float64},
    T0::Float64)
    T         = T0
    err_T     = 1.0
    iter_T    = 0
    maxiter_T = 1000
    tol_T     = 0.001
    while err_T > tol_T && iter_T < maxiter_T
        iter_T = iter_T+1
        g     = HH_EGM(param,r,T,w,b)
        T′, a_path, K′=SimulateAssetsBequests(param,g,1000)
        err_T=abs(T′-T)
        T=T′
    end
    g = HH_EGM(param,r,T,w,b)
    T′, a_path, K′=SimulateAssetsBequests(param,g,1000)
    return T′,K′,a_path
end

# Function that finds r as a function of K
function InterestRate(param::parstruct,K::Float64)
    @unpack α, L, δ = param
    r′ = K^(α-1)*(α*L^(1-α)) - δ
    return r′
end

# Function that simulates consumption, asset holdings and earnings of N×J-panel
function SimulatePanel(
    param::parstruct,
    g::Array{Float64},
    N::Integer,
    r::Float64,
    w::Float64,
    b::Vector{Float64},
    T::Float64,
    )
    @unpack nzz, J, a_grid, e, ψ, DiscretizedMC, θ, Πstat  = param

    # Preallocate stuff
    z_path_indices = zeros(J,N)
    a_path         = zeros(J,N)
    y_path         = zeros(J,N)
    c_path         = zeros(J,N)

    for n in 1:N
        z0=StatsBase.sample(Weights(Πstat))
        z_path_indices[:,n] = QuantEcon.simulate_indices(DiscretizedMC::MarkovChain,J::Int64;init=z0)
    end
    # Path of shock indices
    z_path_indices= convert(Array{Integer,2}, z_path_indices)
    # Everybody is born with zero assets
    a_path[1,:] .= 0.0
    # Asset path
    for j in 2:J
        # For any person in the panel
        for n in 1:N
            # Interpolate the policy function to extend it to points outside the grid to allow assets to stay outside.
            nodes = (a_grid,)
            itp = interpolate(nodes,g[z_path_indices[j-1,n],:,j-1],Gridded(Linear()))
            assets_itp = itp(a_path[j-1,n])
            a_path[j,n] = assets_itp
        end
    end
    # Income path
    for j in 1:J
        for n in 1:N
            y_path[j,n] = (1-θ)*w * e[z_path_indices[j,n],j]  + b[j]
        end
    end
    # Consumption path
    for j in 1:J-1
        c_path[j,:] =  y_path[j,:] .+ T + (1+r)*a_path[j,:] - a_path[j+1,:]
    end
    c_path[J,:] = y_path[J,:] .+ T + (1+r)*a_path[J,:]
    return a_path, y_path, c_path
end

# Function that finds equilibrium interest rate
function GenEq(param::parstruct)
    T    = 1.2;
    r    = 0.02;
    err_r       = 1.0
    iter_r      = 0
    maxiter_r   = 1000
    tol_r       = 0.001
    while err_r > tol_r && iter_r < maxiter_r
        iter_r = iter_r+1
        println("N. Iterations for r=", iter_r)
        K, w, b = CapitalWagesPensions(param,r)
        T,K′    = T_bisec(param,r,w,b,T)
        r′      = InterestRate(param,K′)
        err_r=abs(r′-r)
        r=0.5*r+0.5*r′
        println("|r′-r|= ",err_r)
        println("r= ", r)
    end
    K, w, b                 = CapitalWagesPensions(param,r)
    g                       = HH_EGM(param,r,T,w,b)
    a_path, y_path, c_path  = SimulatePanel(param,g,1000,r,w,b,T)
    return r, K, w, b, g, a_path, y_path, c_path, T
end

## MAIN LOOP

@time begin
param = parameters()
r, K, w, b, g, a_path, y_path, c_path, T = GenEq(param)
end

## RESULTS
@unpack θ, L, a_grid, α, δ, J, λ, Πstat, z_grid = param

# Output
Y = (K^(α))*(L^(1-α))
# Capital-output ratio
KY = K/Y
# Gini index
a_Gini=GiniCoeff(a_path[:])

# Showing results:
println("The equilibrium interest rate r is ", r)
println("The equilibrium capital K is ", K)
println("The equilibrium wage w is ", w)
println("The equilibrium bequest T is ", T)
println("The equilibrium pension b is ", b[71])
println("The income tax θ is ", θ)
println("The capital-output ratio is ", KY)
println("The Gini index is ", a_Gini)

# Plot 1 - Saving policy for diff. shocks.
plot1=Plots.plot(a_grid,g[1,:,30]-a_grid, title="Savings policy: g(z,a,j)-a for j=30 and different values of z", xlabel="a", ylabel="a′-a", label="z=1")
Plots.plot!(a_grid,g[2,:,30]-a_grid,label="z=2")
Plots.plot!(a_grid,g[3,:,30]-a_grid,label="z=3")
Plots.plot!(a_grid,g[4,:,30]-a_grid,label="z=4")
Plots.plot!(a_grid,g[5,:,30]-a_grid,label="z=5")


# Plot distribution of assets
a_mean      = zeros(J)
a_50        = zeros(J)
a_25        = zeros(J)
a_75        = zeros(J)
for j in 1:J
    a_mean[j]      = mean(a_path[j,:] )
    a_50[j]        = median(a_path[j,:])
    a_25[j]        = percentile(a_path[j,:],25)
    a_75[j]        = percentile(a_path[j,:],75)
end

# Plot 2 - Distribution of asset holdings by age
plot2=plot([25:J+24],a_mean,title="Distribution of asset holdings by age",label="Assets, mean",xlabel="j", ylabel="a")
plot!([25:J+24],a_25,label="Assets, 25th pctile")
plot!([25:J+24],a_50,label="Assets, median" )
plot!([25:J+24],a_75,label="Assets, 75th pctile" )


y_mean=zeros(J)
y_Gini=zeros(J)
c_mean=zeros(J)
c_Gini=zeros(J)
for j in 1:J
    y_mean[j]      = mean(y_path[j,:])
    c_mean[j]      = mean(c_path[j,:])
    y_Gini[j]      = GiniCoeff(y_path[j,:])
    c_Gini[j]      = GiniCoeff(c_path[j,:])
end

# Plot 3 - Average earnings and consumption
plot3=plot([25:J+24],y_mean,title="Average earnings and consumption by age",label="Earnings y", xlabel="j")
plot!([25:J+24],λ,label="Earnings y (theory)")
plot!([25:J+24],c_mean,label="Consumption c" )

# Plot 4 - Gini index for earnings and consumption
plot4=plot([25:J+24],y_Gini,title="Gini index for earnings and consumption by age",label="Earnings y", xlabel="j")
plot!([25:J+24],c_Gini,label="Consumption c" )
