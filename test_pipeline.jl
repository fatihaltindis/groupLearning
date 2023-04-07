using 	LinearAlgebra, CovarianceEstimation,
PosDefManifold, PosDefManifoldML,
Diagonalizations, NPZ, YAML, HDF5, EzXML,
BlockDiagonals, DelimitedFiles, TimerOutputs, Base.Threads

# Include required custom packages 
push!(LOAD_PATH, joinpath(@__DIR__, "Modules"))
using 	EEGio, System, FileSystem, EEGpreprocessing, EEGprocessing, ERPs

include(".\\src\\pipelineTools.jl");

filepath = "G:\\Mon Drive\\PhD\\Codes\\julia_codes\\group-learning\\Group-Learning-BCI\\ERP Analysis\\DataBases\\P300";
dbName = "bi2014b";
PCA_dim = 16;
random_state = 1;

db_obj = Database(dbName = dbName, n_of_subject = 5, filepath = filepath);
param_obj = Parameters(verbose = true, PCA_dim = PCA_dim, random_state = random_state);
ts_obj = TSVectorData();
gl_obj = GLData();
res_obj = ResultData();

createTSVectors(db_obj, param_obj, ts_obj, gl_obj, res_obj);
res_obj.sw_res = swTraining(ts_obj.train_vecs, ts_obj.test_vecs,
                 ts_obj.train_labels, ts_obj.test_labels,
                 classifier = param_obj.classifier)[1];

