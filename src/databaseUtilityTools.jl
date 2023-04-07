# Unit databaseUtilityTools.jl, part of groupLearning Package for julia
#
# MIT License 
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes
# 
# DESCRIPTION
# This unit implements required utilities related to caching the filenames,
# loading data from NY files and train/test splitting of the loaded data.
# 

# Cache file names of subjects from the given database.
# There are 4 selection rule available;
# :none     >>  It is possible to select N subjects from the database based 
#           on their filenames order (s01, s02, s03 ... so on)
# :rand     >>  Randomly select subjects and load all sessions of the selected
#           subjects. Random seed can be set using "random_state" argument of the function
# :sort     >>  Data is loaded based on the sorted within session train/test classification 
#           score list (text file) provided in the same folder where database is present.
#           Sorting is descending.
# :sort_r   >>  Same as ":sort" but sorting is ascending.
# :thre     >>  The subjects that have within session train/test classification score that 
#           is above the given threshold will be selected. Default threshold value is 70 %.
# :thre_r   >>  Reverse of the ":thre" option, rejecting subjects above threshold and keeping
#           the ones whose classification accuracy is below the given threshold.

include("dbInfo.jl");

import StatsBase.sample
import Random.MersenneTwister

function selectSubjects(dbName             :: String;
                        n_of_subject       :: Integer = 0,
                        selection_rule     :: Symbol = :none,
                        threshold          :: Float64 = 0.7,
                        random_state       :: Union{Integer, Nothing} = nothing,
                        turn               :: Symbol = :local,
                        filepath           :: Union{Nothing, String} = nothing)

    isnothing(random_state) ? random_state = 1 : nothing;
    # Overruling for sorted selection
    if ((selection_rule == :sort) || (selection_rule == :sort_r)) && (n_of_subject == 0)
        selection_rule = :none;
        @warn("All subjects are selected, sorted selection is disabled!!!")
    end

    # Overruling for threshold selection
    if ((selection_rule == :thre) || (selection_rule == :thre_r))
        (threshold >= 1.0) || (threshold <= 0.0) ? error("Threshold must be between 0-1 !!!") : nothing;
    end

    # Define filepath for locating the databases
    if isnothing(filepath)
        modality = dbModalities[dbName];
        turn == :gricad ? database_path = "/bettik/PROJECTS/pr-bci/COMMON/" : 
                database_path = joinpath(@__DIR__, "DataBases", modality);
    else
        database_path = filepath;
    end
        

    # Variable to keep selected files
    selected_files = [];
    selected_subjects = [];
    dbDir = joinpath(database_path, dbName);
    files = loadNYdb(dbDir);

    ## Create list of subject numbers of given database
    selectedIdx = [findall( x -> occursin(dbListPrefixes[dbName]*lpad(ff,2,"0"), x), 
            files) for ff in 1:250];
    dummy_subject_numbers = map(x-> isempty(selectedIdx[x]) ? [] : x ,1:250);
    filter!(x-> isempty(x) == false, dummy_subject_numbers);
    filter!(x-> isempty(x) == false, selectedIdx);
    selectedIdx = vcat(selectedIdx...);

    (n_of_subject > size(dummy_subject_numbers,1)) || (n_of_subject == 0) ? 
    temp_n_sub = size(dummy_subject_numbers,1) : temp_n_sub = deepcopy(n_of_subject);

    if selection_rule == :rand
        selected_subjects = dummy_subject_numbers[sample(MersenneTwister(random_state),
                            1:length(dummy_subject_numbers), temp_n_sub)];
        selectedIdx = [findall( x -> occursin(dbListPrefixes[dbName]*lpad(ff,2,"0"), x), 
            files) for ff in selected_subjects];
        selectedIdx = vcat(selectedIdx...);
        selected_files = [files[i] for i∈selectedIdx];
    elseif selection_rule == :sort
        cvResults = "$(dbName)_sw_learning.txt";
        CVFileDir = joinpath(dbDir, cvResults);
        cvScores = readdlm(CVFileDir, Float64);

        subject_acc = cvScores[:,3];
        ksort = sortperm(vec(subject_acc), rev=true)
        
        selected_subjects = Int.(repeat(dummy_subject_numbers,inner=dbSessionCount[dbName])[ksort][1:temp_n_sub]);
        selected_files = [files[i] for i ∈ ksort[1:temp_n_sub]];
    elseif selection_rule == :sort_r
        cvResults = "$(dbName)_sw_learning.txt";
        CVFileDir = joinpath(dbDir, cvResults);
        cvScores = readdlm(CVFileDir, Float64);

        subject_acc = cvScores[:,3];
        ksort = sortperm(vec(subject_acc), rev=false)
        
        selected_subjects = Int.(repeat(dummy_subject_numbers,inner=dbSessionCount[dbName])[ksort][1:temp_n_sub]);
        selected_files = [files[i] for i ∈ ksort[1:temp_n_sub]];
    elseif selection_rule == :thre
        cvResults = "$(dbName)_sw_learning.txt";
        CVFileDir = joinpath(dbDir, cvResults);
        cvScores = readdlm(CVFileDir, Float64);

        subject_acc = cvScores[:,3];
        selectedIdx = selectedIdx[subject_acc .> threshold];
        selected_files = [files[i] for i ∈ selectedIdx];
        selected_subjects = repeat(dummy_subject_numbers,inner=dbSessionCount[dbName])[selectedIdx];
    elseif selection_rule == :thre_r
        cvResults = "$(dbName)_sw_learning.txt";
        CVFileDir = joinpath(dbDir, cvResults);
        cvScores = readdlm(CVFileDir, Float64);

        subject_acc = cvScores[:,3];
        selectedIdx = selectedIdx[subject_acc .< threshold];
        selected_files = [files[i] for i ∈ selectedIdx];
        selected_subjects = repeat(dummy_subject_numbers,inner=dbSessionCount[dbName])[selectedIdx];
    else
        selected_subjects = dummy_subject_numbers[1:temp_n_sub];
        selectedIdx = [findall(x -> occursin(dbListPrefixes[dbName]*lpad(ff,2,"0"), x), 
            files) for ff in selected_subjects];
        selectedIdx = vcat(selectedIdx...);
        selected_files = [files[i] for i∈selectedIdx];
    end
    return selected_files, selected_subjects
end
