using Pkg
Pkg.activate("groupL")

# Add packages
using LinearAlgebra, CovarianceEstimation,
PosDefManifold, PosDefManifoldML, Plots,
Diagonalizations, NPZ, YAML, HDF5, EzXML,
BlockDiagonals, DelimitedFiles, TimerOutputs, 
Base.Threads

# Add custom packages 
push!(LOAD_PATH, joinpath(@__DIR__, "Modules"))
using EEGio, System, FileSystem, 
EEGpreprocessing, EEGprocessing, ERPs

include(joinpath(@__DIR__, "src", "pipelineTools.jl"));

# Main folder of the database
filepath = joinpath(@__DIR__, "data");
# Folder name of the selected database
dbName = "bi2015a";

# Parameters
PCA_dim = nothing;
random_state = 1;
n_of_subject = 4;
selection_rule = :none;
threshold = 0.66;
turn = :local;
random_state = 1;
bandpass = (1,16);
artefact_rej = true;
n_splits = 5;
split_ratio = 50;
whitening = :svd;
initialize_U = :smart;
paradigm = :ERP;
verbose = true;

obj_list = initiateObjects(dbName, filepath;
                           n_of_subject=n_of_subject, PCA_dim=PCA_dim, verbose=true,
                           split_ratio=split_ratio, n_splits=n_splits,
                           whitening=whitening, initialize_U=initialize_U);

# Run multiple pipelines
pipeline = [createTSVectors, trainSW, prepareGL, runGL, trainGL];
runPipe!(pipeline, obj_list)

# Subject specific train/test pipeline
pipeline1 = [createTSVectors, trainSW];
runPipe!(pipeline1, obj_list)

# Group learning pipeline
pipeline2 = [createTSVectors, prepareGL, runGL, trainGL];
runPipe!(pipeline2, obj_list)

# Fast alignment pipeline
pipeline3 = [runLeaveOut, trainFA];
runPipe!(pipeline3, obj_list)

# PLot and compare pipelines
plotAcc(obj_list)

# Pipeline parts
# createTSVectors >>> reads NY files, extracts TS vectors
#                     and splits into train/test
# trainSW         >>> runs subject specific train/test classification 
#                     on extracted/splitted data
# prepareGL       >>> pre-steps of the Group Learning. It is a must step
#                     for running Group Learning 
# runGL           >>> runs GALIA
# 
# trainGL         >>> train/test step of the group learning pipeline
#                     
# runLeaveOut     >>> runs leave-one-out stategy for performing fast alignment
#
# trainFA         >>> train/test step of the fast alignment pipeline
