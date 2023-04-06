# Unit covarianceEstimation.jl, part of groupLearning Package for julia
#
# MIT License 
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes

# Estimate super trial covariance matrices for train and test splits
# For ERP
function estimate_cov_mat(trials            :: Vector{Matrix{Float64}},
                          train_splits      :: Vector{Vector{Vector{Int64}}},
                          test_splits       :: Vector{Vector{Vector{Int64}}},
                          prototypes        :: Vector{Matrix{Float64}};
                          det_normalization :: Bool = false,
                          estimator         :: Symbol = :lw)

    train_covs = Vector{Hermitian{}}[]
    test_covs = Vector{Hermitian{}}[]
    for (tr, te, prt) in zip(train_splits, test_splits, prototypes)
        if estimator == :lw
            push!(train_covs, ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X prt])) 
                for X ∈ trials[vcat(tr...)]]));
            push!(test_covs, ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X prt])) 
                for X ∈ trials[vcat(te...)]]));
        elseif estimator == :oas
            push!(train_covs, ℍVector([ℍ(cov(LinearShrinkage(DiagonalCommonVariance(), :oas), [X prt])) 
                for X ∈ trials[vcat(tr...)]]));
            push!(test_covs, ℍVector([ℍ(cov(LinearShrinkage(DiagonalCommonVariance(), :oas), [X prt])) 
                for X ∈ trials[vcat(te...)]]));
        else
            push!(train_covs, ℍVector([ℍ(cov(SimpleCovariance(corrected=false), [X prt])) 
                for X ∈ trials[vcat(tr...)]]));
            push!(test_covs, ℍVector([ℍ(cov(SimpleCovariance(corrected=false), [X prt])) 
                for X ∈ trials[vcat(te...)]]));
        end
    end

    if det_normalization
        for i in eachindex(train_covs)
            map!(x-> x=det1(x),train_covs[i],train_covs[i])
            map!(x-> x=det1(x),test_covs[i],test_covs[i])
        end
        return train_covs, test_covs
    else
        return train_covs, test_covs
    end
end

# Estimate super trial covariance matrices for all the given trials
# For ERP
function estimate_cov_mat(trials            :: Vector{Matrix{Float64}},
                          prototypes        :: Vector{Matrix{Float64}};
                          det_normalization :: Bool = false,
                          estimator         :: Symbol = :lw)

    covs = Vector{Hermitian{}}[]
    for prt in prototypes
        if estimator == :lw
            push!(covs, ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X prt])) 
                for X ∈ trials]));
        elseif estimator == :oas
            push!(covs, ℍVector([ℍ(cov(LinearShrinkage(DiagonalCommonVariance(), :oas), [X prt])) 
                for X ∈ trials]));
        else
            push!(covs, ℍVector([ℍ(cov(SimpleCovariance(corrected=false), [X prt])) 
                for X ∈ trials]));
        end
    end

    if det_normalization
        for i in eachindex(covs)
            map!(x-> x=det1(x),covs[i],covs[i])
        end
        return covs
    else
        return covs
    end
end

# Estimate covaraince matrices for train and test splits
# For MI data
function estimate_cov_mat(trials            :: Vector{Matrix{Float64}},
                          train_splits      :: Vector{Vector{Vector{Int64}}},
                          test_splits       :: Vector{Vector{Vector{Int64}}};
                          estimator         :: Symbol = :lw,
                          det_normalization :: Bool = false)

    train_covs = Vector{Hermitian{}}[]
    test_covs = Vector{Hermitian{}}[]
    for (tr, te) in zip(train_splits, test_splits)
        if estimator == :lw
            push!(train_covs,ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation(), :lw), X)) 
                for X ∈ trials[vcat(tr...)]]));

            push!(test_covs,ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation(), :lw), X)) 
                for X ∈ trials[vcat(te...)]]));
        elseif estimator == :oas
            push!(train_covs,ℍVector([ℍ(cov(LinearShrinkage(DiagonalCommonVariance(), :oas), X)) 
                for X ∈ trials[vcat(tr...)]]));

            push!(test_covs,ℍVector([ℍ(cov(LinearShrinkage(DiagonalCommonVariance(), :oas), X)) 
                for X ∈ trials[vcat(te...)]]));
        else
            push!(train_covs,ℍVector([ℍ(cov(SimpleCovariance(corrected=false), X)) 
                for X ∈ trials[vcat(tr...)]]));

            push!(test_covs,ℍVector([ℍ(cov(SimpleCovariance(corrected=false), X)) 
                for X ∈ trials[vcat(te...)]]));
        end
    end

    if det_normalization
        for i in eachindex(train_covs)
            map!(x-> x=det1(x),train_covs[i],train_covs[i])
            map!(x-> x=det1(x),test_covs[i],test_covs[i])
        end
        return train_covs, test_covs
    else
        return train_covs, test_covs
    end
end

# Estimate covariance matrices for all the given trials
# For MI data
function estimate_cov_mat(trials            :: Vector{Matrix{Float64}};
                          estimator         :: Symbol = :lw,
                          det_normalization :: Bool = false)

    covs = Vector{Hermitian{}}[]
    if estimator == :lw
        push!(covs,ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation(), :lw), X)) 
            for X ∈ trials]));
    elseif estimator == :oas
        push!(covs,ℍVector([ℍ(cov(LinearShrinkage(DiagonalCommonVariance(), :oas), X)) 
            for X ∈ trials]));
    else
        push!(covs,ℍVector([ℍ(cov(SimpleCovariance(corrected=false), X)) 
            for X ∈ trials]));
    end

    if det_normalization
        for i in eachindex(train_covs)
            map!(x-> x=det1(x),covs[i],covs[i])
        end
        return covs
    else
        return covs
    end
end