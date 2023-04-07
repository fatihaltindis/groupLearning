# Unit pipelineObjects.jl, part of groupLearning Package for julia
#
# MIT License 
# version: 10 Sept 2022
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes

# This is structure object to be used in reading raw EEG signals,
# creating train/test splits for reminder steps of the pipeline
Base.@kwdef mutable struct Database
    dbName                  :: String
    n_of_subject            :: Integer = 0
    selection_rule          :: Symbol = :none
    threshold               :: Float64 = 0.7
    turn                    :: Symbol = :local
    random_state            :: Union{Integer, Nothing} = nothing
    filepath                :: String
    bandpass                :: Tuple = (1,16)
    artefact_rej            :: Bool = true
    n_splits                :: Integer = 5
    split_ratio             :: Union{Integer, Nothing} = nothing    
    paradigm                :: Symbol = :ERP
    subject_list            :: Vector{} = Vector[]
end

# Includes all the hyperparameters required for running the code
# Though all have initial values
Base.@kwdef mutable struct Parameters
    ERP_weight              :: Symbol = :a
    overlapping             :: Bool = true
    PCA_dim                 :: Union{Nothing, Integer} = nothing
    cov_estimator           :: Symbol = :lw
    det_normalization       :: Bool = false
    dt                      :: Bool = false
    ch_union                :: Union{Nothing,Vector{String}} = nothing
    vec_mean                :: Bool = true
    vec_norm                :: Bool = true
    whitening               :: Symbol = :smart
    white_dim               :: Integer = 16
    sub_dim                 :: Vector{Integer} = [16]
    smart_subspace          :: Integer = 16
    n_of_boot               :: Integer = 100
    bootsize                :: Integer = 25
    normalize_U             :: Symbol = :white
    initialize_U            :: Symbol = :none
    sort_U                  :: Bool = true
    classifier              :: Symbol = :LinearSVC
    random_state            :: Union{Nothing, Integer} = nothing
    leaveout                :: Bool = false
    repetition              :: Union{Nothing,Int} = nothing
    timer                   :: Union{Nothing,TimerOutput} = nothing
    save_opt                :: Symbol = :v1
    verbose                 :: Bool = false
end

# Train/test splitted TS vectors, corresponding class labels
Base.@kwdef mutable struct TSVectorData
    train_vecs              :: Vector{Vector{Matrix{Float64}}} = []
    test_vecs               :: Vector{Vector{Matrix{Float64}}} = []
    train_labels            :: Vector{Vector{}} = []
    test_labels             :: Vector{Vector{}} = []
    all_vecs                :: Vector{Vector{Matrix{Float64}}} = []
    all_labels              :: Vector{Vector{}} = []
    all_clabels             :: Vector{Any} = []
end

# Group learning algorithm aligment matrices,
# U matrices, whitening matrices (wh),
# whitened matrices (T) and matrices
Base.@kwdef mutable struct GLData
    B                       :: Vector{Vector{Any}} = []
    B_fast                  :: Vector{Vector{Any}} = []
    U                       :: Vector{Vector{Matrix{Float64}}} = []
    U_fast                  :: Vector{Vector{Matrix{Float64}}} = []
    T                       :: Vector{Vector{Matrix{Float64}}} = []
    S                       :: Vector{Vector{Any}} = []
    wh                      :: Vector{Vector{Matrix{Float64}}} = []
end

# Classification results of pipelines, GALIA iteration
# and convergence ratio
Base.@kwdef mutable struct ResultData
    sw_res                  :: Vector{} = Vector{}[]
    gl_res                  :: Vector{} = Vector{}[]
    gl_res_fa               :: Vector{} = Vector{}[]
    fa_res                  :: Vector{} = Vector{}[]
    iter                    :: Vector{Any} = []
    conv                    :: Vector{Any} = []
end