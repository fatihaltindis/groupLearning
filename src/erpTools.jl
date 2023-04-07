# Unit erpTools.jl, part of groupLearning Package for julia
#
# MIT License 
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes

# This function is for creating prototype trials from each train split,
# separately. 

function createPrototype(raw_eeg         :: Matrix{Float64}, 
                         window_length   :: Int, 
                         train_splits    :: Vector{Vector{Vector{Int64}}},
                         TA_index        :: Int;
                         overlapping     :: Bool = true,
                         weights         :: Union{Symbol, Nothing} = nothing,
                         PCA_dim         :: Union{Int, Nothing} = nothing,
                         verbose         :: Bool = false)

    ch_ = minimum(size(raw_eeg));
    if !isnothing(PCA_dim)
        (ch_ < PCA_dim) ? throw(ArgumentError("PCA dimension cannot be bigger than number of channels!!!")) : 
            verbose && @info("PCA is enabled!") 
            verbose && @info("$(PCA_dim) principal components will be kept out of $(ch_).") 
    end

    prototype = Matrix{Float64}[];
    for s in train_splits
        temp = mean(raw_eeg, window_length, s, overlapping=overlapping;
                    weights=weights)[TA_index];
        if !isnothing(PCA_dim)
            temp *= eigvecs(cov(SimpleCovariance(), temp))[:, ch_-PCA_dim+1 : ch_];
        end
        push!(prototype,temp)
    end
    return prototype    
end

# This function is for creating prototype from all trials
function createPrototype(raw_eeg         :: Matrix{Float64}, 
                         window_length   :: Int, 
                         trials          :: Vector{Vector{Int64}},
                         TA_index        :: Int;
                         overlapping     :: Bool = true,
                         weights         :: Union{Symbol, Nothing} = nothing,
                         PCA_dim         :: Union{Int, Nothing} = nothing,
                         verbose         :: Bool = true)

    ch_ = minimum(size(raw_eeg))
    if !isnothing(PCA_dim)
        (ch_ < PCA_dim) ? throw(ArgumentError("PCA dimension cannot be bigger than number of channels!!!")) : 
            verbose && @info("PCA is enabled!") 
            verbose && @info("$(PCA_dim) principal components will be kept out of $(ch_).") 
    end

    prototype = Matrix{Float64}[]
    temp = mean(raw_eeg, window_length, trials, overlapping=overlapping;
            weights=weights)[TA_index]
    if !isnothing(PCA_dim)
        temp *= eigvecs(cov(SimpleCovariance(), temp))[:, ch_-PCA_dim+1 : ch_];
    end
    push!(prototype,temp)
    return prototype
end

