# Unit dataTransformation.jl, part of groupLearning Package for julia
#
# MIT License 
# Copyright (c) - 2023
# Fatih Altindis⁺ꜝ and Marco Congedo ꜝ
# ⁺ Abdullah Gul University, Kayseri
# ꜝ GIPSA-lab, CNRS, University Grenoble Alpes

# SPLIT DATA INTO TRAIN AND TEST SPLITS
# This function takes marker data from NY format data
# It returns two variables: train splits and test splits
# Splits include the time series index number of the beginning 
# of each trial or trial no. 
# return_type sets this option
# :index for raw_data trial index (e.g. trial starts at 2047, or 5247 ....)
# and :trial_no trial returns trial numbers (e.g. 1st trial, 25th trial ...)
import Random.shuffle
function getSplits(trial_idx               :: Vector{Vector{Int}};
                   c_labels                :: Vector{} = [],
                   n_splits                :: Union{Int, Nothing} = nothing,
                   split_ratio             :: Union{Nothing,Real} = nothing,
                   return_type             :: Symbol = :index,
                   random_state            :: Union{Int, Nothing} = nothing)
    # Check if random_state is activated
    isnothing(random_state) ? random_state = 1 : nothing;
    
    # Check if split ratio is in correct interval
    isnothing(split_ratio) ? nothing : (split_ratio > 0 && split_ratio < 100 ? nothing : 
        throw(ErrorException("Split ratio must be between 0 and 100 !!!")));
    
    reverse_split = false;
    if !isnothing(split_ratio)
        split_ratio <= 50 ? reverse_split = true : reverse_split = false;
        isnothing(n_splits) ? n_splits = 5 : n_splits = n_splits;
        # Set split size and initialize split matrices
        reverse_split ? split_sizes = Int.(floor.(size.(trial_idx,1)*(split_ratio/100))) :
            split_sizes = Int.(floor.(size.(trial_idx,1)*((100-split_ratio)/100)));
    else
        isnothing(n_splits) ? n_splits = 5 : n_splits = n_splits;
        split_sizes = Int.(floor.(size.(trial_idx,1)/n_splits));
        n_splits == 1 ? reverse_split = true : nothing;
    end

    seed_repeat = true;
    split_seeds = [];
    while seed_repeat
        split_seeds = deepcopy(rand(MersenneTwister(random_state), 1:100, n_splits));
        # Check if there are same seed numbers
        size(unique(deepcopy(split_seeds)), 1) == size(split_seeds, 1) ? seed_repeat=false : random_state=random_state*2;
    end

    train_splits, test_splits = Vector{Vector{Int}}[], Vector{Vector{Int}}[]
    # Splitting into n_splits of test/train splits for each class
    for n = 1:n_splits
        temp_marker = deepcopy(shuffle.(MersenneTwister(split_seeds[n]), trial_idx));
        temp_test = map((t,s) -> t[1:s], temp_marker, split_sizes)
        pushfirst!(test_splits,temp_test)
        temp_train = map((t,s) -> t[s+1:end], temp_marker, split_sizes)
        pushfirst!(train_splits,temp_train)
    end

    # Return splits indices or trial numbers
    # Return labels for train/test splits
    if return_type == :index
        isnothing(split_ratio) ? (return train_splits, test_splits) : 
            (return test_splits, train_splits)
    elseif return_type == :trial_no
        index_arranger = [0, cumsum(size.(trial_idx,1))[1:end-1]...];
        map!(x -> indexin.(x,trial_idx), train_splits, train_splits)
        map!(x -> indexin.(x,trial_idx), test_splits, test_splits)
        for i in eachindex(test_splits)
            test_splits[i] = map((x,y)-> x.+y, test_splits[i], index_arranger)
            train_splits[i] = map((x,y)-> x.+y, train_splits[i], index_arranger)
        end
        # Create labels for test and train splits
        train_labels, test_labels = [], []
        for i in eachindex(train_splits[1])
            push!(train_labels, Int64.(ones(size(train_splits[1][i],1)).*i))
            push!(test_labels, Int64.(ones(size(test_splits[1][i],1)).*i))
        end
        reverse_split ? (return test_splits, train_splits, 
            vcat(test_labels...), vcat(train_labels...)) : 
            (return train_splits, test_splits, 
            vcat(train_labels...), vcat(test_labels...))
    else
        ArgumentError("Return type must be 'index' or 'trial_no' !!!")
    end
end

function swapDims(train_splits             :: Vector{Vector{Matrix{Float64}}},
                  test_splits              :: Vector{Vector{Matrix{Float64}}})

    train = Vector{Vector{Matrix{Float64}}}(undef,length(train_splits[1]))
    test = Vector{Vector{Matrix{Float64}}}(undef,length(train_splits[1]))

    for sp in eachindex(train)   
        train[sp] = [Z[sp] for (d,Z) ∈ enumerate(train_splits)];
        test[sp] = [Z[sp] for (d,Z) ∈ enumerate(test_splits)];
    end
    return train, test
end
