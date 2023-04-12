
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
n_of_subject = 5;
selection_rule = :none;
threshold = 0.66;
turn = :local;
random_state = 1;
bandpass = (1,16);
artefact_rej = true;
n_splits = 4;
split_ratio = 40;
paradigm = :ERP;

db_obj, param_obj, ts_obj, gl_obj, res_obj = initiateObjects(dbName, filepath;
                                              n_of_subject=n_of_subject, PCA_dim=PCA_dim);

# db_obj = Database(dbName = dbName, n_of_subject = 5, filepath = filepath);
# param_obj = Parameters(verbose = true, PCA_dim = PCA_dim, random_state = random_state);
# ts_obj = TSVectorData();
# gl_obj = GLData();
# res_obj = ResultData();

createTSVectors(db_obj, param_obj, ts_obj, gl_obj, res_obj);

trainSW(db_obj, param_obj, ts_obj, gl_obj, res_obj);

prepareGL(db_obj, param_obj, ts_obj, gl_obj, res_obj);

runGL(db_obj, param_obj, ts_obj, gl_obj, res_obj);

trainGL(db_obj, param_obj, ts_obj, gl_obj, res_obj);

# initiateObjects(dbName, filepath) |> createTSVectors |> prepareGL |> runGL |> trainGL;


