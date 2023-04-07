# Unit tangentSpaceVectors.jl, part of groupLearning Package for julia
#
# MIT License 
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes

# This function handles split structure of the data 
# for using tsMap function from PosDefManifold module.
# Slightly altered version of tsMap is also available for 
# applying dimensionality transcending, that is proposed by [1].
# [1] Rodrigues PLC, Congedo M, Jutten C. Dimensionality Transcending: 
#     A Method for Merging BCI Datasets With Different Dimensionalities. 
#     IEEE Trans Biomed Eng. 2021 Feb;68(2):673-684. 
#     doi: 10.1109/TBME.2020.3010854. Epub 2021 Jan 20. PMID: 32746067.

function tangentVectors(train_covs         :: Vector{Vector{Hermitian}},
                        test_covs          :: Vector{Vector{Hermitian}},
                        vecRange           :: Union{Nothing,UnitRange},
                        paradigm           :: Symbol;
                        metric             :: Metric = Fisher,
                        w                  :: Union{Vector{Vector},Nothing} = nothing,
                        transpose          :: Bool = true,
                        dt                 :: Bool = false,
                        ch_union           :: Union{Nothing, Vector{String}} = nothing,
                        ch_sub             :: Union{Nothing, Vector{String}} = nothing,
                        PCA_dim            :: Union{Nothing, Integer} = nothing)

    isnothing(w) ? w=[] : nothing

    train_ts_vecs, test_ts_vecs = Matrix{Float64}[], Matrix{Float64}[]
    if !dt
        for i in eachindex(train_covs)
            if paradigm == :ERP
                push!(train_ts_vecs, tsMap(metric, train_covs[i]; w=w, transpose=transpose, vecRange=vecRange)[1])
                push!(test_ts_vecs, tsMap(metric, test_covs[i]; w=w, transpose=transpose, vecRange=vecRange)[1])
            elseif paradigm == :MI
                push!(train_ts_vecs, tsMap(metric, train_covs[i]; w=w, transpose=transpose, ⏩=false)[1])
                push!(test_ts_vecs, tsMap(metric, test_covs[i]; w=w, transpose=transpose, ⏩=false)[1])
            end
        end
    else
        isnothing(ch_union) ? throw(ErrorException("Union channel space cannot be empty!!!")) : nothing;
        @info("Dimensionality Transcending is active!!!")
        for i in eachindex(train_covs)
            if paradigm == :ERP
                # Dimensionality Transcending on train data
                temp_ts_mats = dt_tsMap(metric, train_covs[i]; w=w, transpose=transpose)[1];
                push!(train_ts_vecs, dt_promotion(temp_ts_mats, ch_union, ch_sub; PCA_dim = PCA_dim));
                # Dimensionality Transcending on train data
                temp_ts_mats = dt_tsMap(metric, test_covs[i]; w=w, transpose=transpose)[1];
                push!(test_ts_vecs, dt_promotion(temp_ts_mats, ch_union, ch_sub; PCA_dim = PCA_dim));
            elseif paradigm == :MI
                @info("MI version will be added.")
            end
        end
    end
    return train_ts_vecs, test_ts_vecs
end

function tangentVectors(train_covs         :: Vector{Vector{Hermitian}},
                        vecRange           :: UnitRange,
                        modality           :: Symbol;
                        metric             :: Metric = Fisher,
                        w                  :: Union{Vector{Vector},Nothing} = nothing,
                        transpose          :: Bool = true,
                        dt                 :: Bool = false,
                        ch_union           :: Union{Nothing, Vector{String}} = nothing,
                        ch_sub             :: Union{Nothing, Vector{String}} = nothing,
                        PCA_dim            :: Union{Nothing, Integer} = nothing)

    isnothing(w) ? w=[] : nothing

    train_ts_vecs = Matrix{Float64}[]
    if !dt
        for i in eachindex(train_covs)
            if modality == :ERP
                push!(train_ts_vecs, tsMap(metric, train_covs[i]; w=w, transpose=transpose, vecRange=vecRange)[1])
            elseif modality == :MI
                push!(train_ts_vecs, tsMap(metric, train_covs[i]; w=w, transpose=transpose, ⏩=false)[1])
            end
        end
    else
        isnothing(ch_union) ? throw(ErrorException("Union channel space cannot be empty!!!")) : nothing;
        @info("Dimensionality Transcending is active!!!")
        for i in eachindex(train_covs)
            if modality == :ERP
                # Dimensionality Transcending on train data
                temp_ts_mats = dt_tsMap(metric, train_covs[i]; w=w, transpose=transpose)[1];
                push!(train_ts_vecs, dt_promotion(temp_ts_mats, ch_union, ch_sub; PCA_dim = PCA_dim));
            elseif modality == :MI
                @info("MI version will be added.")
            end
        end
    end
    return train_ts_vecs
end