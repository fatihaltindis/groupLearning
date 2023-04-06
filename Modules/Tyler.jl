#   Module "Tyler.jl" for Julia language
#
#   MIT License
#   Copyright (c) 2020
#   Marco Congedo, CNRS, UGA, Grenoble-INP, Grenoble, France
#   https://sites.google.com/site/marcocongedo/
#
# This module implements Tyler's M Estimator of covariance
# matrix 'shape' (Tyler, 1987) and the normalized regularized version
# of Zhang and Wiesel (2016)
#
#
# ? CONTENT
#
# FUNCTIONS:
# tme   | Tyler M-Estimator fixed point algorithm (Tyler, 1987)
# nrtme | normalized regularized Tyler's M-Estimator (Zhang and Wiesel, 2016)
#
# REFERENCES
# David E. Tyler (1987)
# A Distribution-Free M-Estimator of Multivariate Scatter
# The Annals of Statistics, 15(1), 234-251.
# https://projecteuclid.org/download/pdf_1/euclid.aos/1176350263

# Teng Zhang, Ami Wiesel (2016)
# Automatic diagonal loading for Tyler's robust covariance estimator
# IEEE Statistical Signal Processing Workshop (SSP), 1-5.
# https://sciences.ucf.edu/math/tengz/wp-content/uploads/sites/45/2016/08/automatic-diagonal-loading-3.pdf


module Tyler

using LinearAlgebra

export
   tme,
   nrtme

## Tyler M-Estimator fixed point algorithm (Tyler, 1987)
# `X` (the data) must be a wide matrix (for the sake of efficiency)
# `tol` is the stopping criterion
# `maxiter` is the maximum number of iterations allowed
# if `verbose`, information on convergence will be printed in the REPL.
function tme(   X::AbstractMatrix{T};
                tol::Real = real(T)(1e-6),
                maxiter::Int = 200,
                verbose::Bool = false) where {T<:Union{Real,Complex}}
    n, t = size(X)
    R = Matrix{T}(I, n, n)
    Rnew = Matrix{T}(undef, n, n)
    iter, 😋 = 1, false

    verbose && println("Iterating M-estimator fixed-point algorithm...")
    while true
        C = cholesky(R)
        fill!(Rnew, zero(T))
        @inbounds for i = 1:t
            @views v = C.L \ X[:, i]
            Rnew += (X[:, i] .* X[:, i]') ./ (v ⋅ v)
        end
        Rnew *= inv(tr(Rnew))
        conv = norm(Rnew - R) / norm(R)
        verbose && println("iteration: ", iter, "; convergence: ", conv)
        (overRun = iter == maxiter) && @warn(
            "M-estimator reached the max number of iterations before convergence:",
            iter,
        )
        (😋 = conv <= tol) || overRun == true ? break : (iter += 1; R[:] = Rnew)
    end # while
    verbose && @info("Convergence has " * (😋 ? "" : "not ") * "been attained.\n\n")
    return Rnew
end


## normalized regularized Tyler's M-Estimator (Zhang and Wiesel, 2016)
# `X` (the data) must be a wide matrix (for the sake of efficiency)
# if `reg` is `:rmt` (default) the random matrix theory shrinkage is used.
# 	Any other symbol will use the Ledoit & Wolf shrinkage.
# `tol` is the stopping criterion
# `maxiter` is the maximum number of iterations allowed
# if `verbose`, information on convergence will be printed in the REPL.
function nrtme( X::AbstractMatrix{T};
                reg::Symbol = :rmt,
                tol::Real = real(T)(1e-6),
                maxiter::Int = 500,
                verbose::Bool = false) where {T<:Union{Real,Complex}}
    n, t = size(X)
    R = Matrix{T}(I, n, n)
    Rnew = zeros(T, n, n)
    x = Matrix{T}(undef, n, 1)
    v = Vector{T}(undef, n)
    iter, 😋, α, β, nt⁻¹ = 1, false, 0.0, 0.0, n / t
    @inbounds x² = [x⋅x for x ∈ eachcol(X)]

    if reg == :rmt
        @inbounds for i=1:t
            x[:] = X[:, i]
            BLAS.gemm!('N', 'T', inv(x²[i]), x, x, 1., Rnew) # | instead of Rnew += (X[:, i].*X[:, i]')./x²[i]
        end
        ζ = n * tr((Rnew ./ t)^2) - nt⁻¹ - 1.
    else
        scm = (X * X') .* inv(n)
        ζ = (n * tr(scm^2) / (tr(scm))^2) - 1.
    end
    α = clamp(inv(t) * ((ζ + 1 + n) / (ζ + nt⁻¹)), 0., 1.)
    β = 1. - α
    αn⁻¹ = α / n
    g(x, β) = BLAS.gemm('N', 'T', β, x, x)

    verbose && println("Iterating nrtme-estimator fixed-point algorithm...")
    while true
        # compute tr(R⁻¹) = tr(Diagonal(L⁻¹'*L⁻¹))
        if n<400 BLAS.set_num_threads(1) end
        L = cholesky(R)
        L⁻¹ = inv(L.L)
        trR⁻¹ = T(0)
        for j = 1:n, i = j:n @inbounds trR⁻¹ += abs2(L⁻¹[i, j]) end
        if n<400 BLAS.set_num_threads(Sys.CPU_THREADS) end

        fill!(Rnew, zero(T))
        for i = 1:t
            x[:] = X[:, i]
            v[:] = L \ x
            c = αn⁻¹ * x²[i]
            Rnew += (g(x, β) + c*I) ./ (β*(v⋅v) + c*trR⁻¹)
            #Rnew += (β*(x.*x')+(αn⁻¹*x²[i])*I) ./ (β*(v⋅v)+αn⁻¹*trR⁻¹*x²[i])
        end
        Rnew *= (inv(tr(Rnew)))
        conv = norm(Rnew - R) / norm(R)

        verbose && println("iteration: ", iter, "; convergence: ", conv)
        (overRun = iter == maxiter) && @warn(
            "nrtme-estimator reached the max number of iterations before convergence:",
            iter,
        )
        (😋 = conv <= tol) || overRun == true ? break : (iter += 1; R[:] = Rnew)
    end # while
    verbose && @info("Convergence has " * (😋 ? "" : "not ") * "been attained.\n\n")
    return Rnew
end


end # module


# test the M-estimators
#
# create data drawn randomly from a multivariate t-student
# distribution with `df` degrees of freedom
# and check how far the estimated shape is different from
# `trueC`


#=
using BenchmarkTools, Distributions, PDMats, PosDefManifold, Statistics

n, t, df = 30, 512, 3.0
trueC=randP(n)
trueC=trueC/tr(trueC)
tdist=MvTDist(3., zeros(n), PDMat(Matrix(trueC)))
X=rand(tdist, t)

Tyler.tme(X, verbose=true)
Tyler.nrtme(X, verbose=true)
Tyler.nrtme(X, reg=:lw, verbose=true)

# run 100 simulations
# and check the Fisher distance between true and estimated shape
t=1_000
ntrials=100
d1=Vector{Float64}(undef, ntrials)
d2=Vector{Float64}(undef, ntrials)
for i=1:ntrials
	tdist=MvTDist(df, zeros(n), PDMat(Matrix(trueC)))
	X=rand(tdist, t);
	#println(cov(td))
	scm=cov(X')
	scm=scm/tr(scm)
	M = (tme(X))
	d1[i]=distance(Fisher, Hermitian(trueC), Hermitian(scm))
	d2[i]=distance(Fisher, Hermitian(trueC), Hermitian(M))
end
10*log10(mean(d1))
10*log10(mean(d2))

# benchmark algorithms
@benchmark(tme(X))
@benchmark(tme2(X))
@benchmark(nrtme(X))
=#
