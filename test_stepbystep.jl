using 	LinearAlgebra, CovarianceEstimation,
PosDefManifold, PosDefManifoldML, Plots,
Diagonalizations, NPZ, YAML, HDF5, EzXML,
BlockDiagonals, DelimitedFiles, TimerOutputs, Base.Threads

# Include required custom packages 
push!(LOAD_PATH, joinpath(@__DIR__, "Modules"))
using 	EEGio, System, FileSystem, EEGpreprocessing, EEGprocessing, ERPs

include(joinpath(@__DIR__, "src", "databaseUtilityTools.jl"));
include(joinpath(@__DIR__, "src", "dataTransformation.jl"));
include(joinpath(@__DIR__, "src", "covarianceEstimation.jl"));
include(joinpath(@__DIR__, "src", "erpTools.jl"));
include(joinpath(@__DIR__, "src", "tangentSpaceVectors.jl"));
include(joinpath(@__DIR__, "src", "utils.jl"));
include(joinpath(@__DIR__, "src", "galia.jl"));
include(joinpath(@__DIR__, "src", "learning.jl"));
include(joinpath(@__DIR__, "src", "dbInfo.jl"));

# Main folder of the database
filepath = joinpath(@__DIR__, "data");
# Folder name of the selected database
dbName = "bi2015a";

#
n_of_subject = 0;		    # number of subjects to be used (set to 0 for using all subjects)
selection_rule = :none;     # selection rule (set :rand for random selection)
n_splits = 5;			    # number of splits for train/test
split_ratio = 50;           # Train/Test ratio of the trials
turn = :local;              # :gricad
random_state = 1;
# Preprocessing parameters
bandpass = (1,16);		    # bandpass filter cut-off frequencies
artefact_rejection = 1;		# set 1 for enable artefact rejection
# ERP Prototype estimation & Covariance Estimation
ERP_weight = :a;		    # adaptive weights 
overlapping = true;		    # overlapping of ERPs
PCA_dim = nothing;			    # set nothing if no PCA is applied
det_normalization = false;	# normalize determinants of cov mats
# Dimensionality transcending parameters
ch_union = nothing;         # Union channel space for Dimensionality Transcending
dt = false;
# Other parameters
verbose = true;
paradigm = dbListParadigm[dbName];
clf = :LinearSVC;

all_clabels = []; 	        # Class labels of all subjects
train_vecs, test_vecs = Vector{Matrix{Float64}}[], Vector{Matrix{Float64}}[];
train_labels, test_labels = Vector{}[], Vector{}[];

# Select files to be loaded and used
selected_files, subject_list = selectSubjects(dbName; n_of_subject = n_of_subject,
                                              selection_rule = selection_rule, turn = :none,
                                              filepath = filepath);
# Start reading data and estimate tangent space vectors
for (n,temp_file) ∈ enumerate(selected_files)
    println("processing file $n of $(length(selected_files))...")
    o=readNY(temp_file;
             bandpass=bandpass, upperLimit=artefact_rejection);
    TAindex = findfirst(isequal("Target"), o.clabels);
    # Split the data
    train_splits, _ = getSplits(o.mark; 
            n_splits = n_splits,
            split_ratio = split_ratio,
            return_type = :index,
            random_state = random_state);
    train_splits_no, test_splits_no, y_train, y_test = getSplits(o.mark; 
            n_splits = n_splits,
            split_ratio = split_ratio,
            return_type = :trial_no,
            random_state = random_state);
    # ERP Prototype estimation
    Y = createPrototype(o.X, o.wl, train_splits, TAindex;
                        overlapping = overlapping,
                        weights = ERP_weight,
                        PCA_dim = PCA_dim,
                        verbose = verbose);

    # Covariance estimation
    train_covs, test_covs = estimateCov(o.trials, train_splits_no, test_splits_no, Y;
                            det_normalization = det_normalization, estimator = :lw);
    train_tsvec, test_tsvec = tangentVectors(train_covs, test_covs, 1:o.ne, paradigm; 
                            transpose=false)

    # append data
    push!(train_vecs, train_tsvec)
    push!(test_vecs, test_tsvec)
    push!(train_labels, y_train)
    push!(test_labels, y_test)
    append!(all_clabels,[o.clabels])
end
train_vecs, test_vecs = swapDims(train_vecs, test_vecs);

# Subject-wise learning
sw_acc, sw_std = swTraining(train_vecs, test_vecs, train_labels, test_labels; classifier = clf);

# Group Learning Parameters
n_of_boots = 150;           # number of bootstraps from each class
bootsize = 25;              # number of features used for each bootstrapped feature vector
vecMean = true;      		# zero mean of each row of bootstraps
vecNorm = true;			    # column normalization of bootstraps
whitening = :svd;		    # 4 types of whitening is avaliable >>> :svd, :pca, :smart, :none
white_dim = 16;			    # Whitening dimension >>> typicall 4 is enough
sub_dim = 16;
sort_U = true;
smart_subspace = 16;		# Subspace dimension for smart whitening >>> typically 32 is enough
normalize_U = :white;	    # columns normalization of U matrices
initialize_U = :none;    	# initialize the joal algorithm if preferred

# Group learning
M = size(train_vecs[1],1);
n_splits = size(train_vecs,1);
𝐁 = Vector{Vector{Matrix{Float64}}}(undef,n_splits);

for sp = 1:n_splits
    𝐖 = Vector{Matrix{Float64}}(undef,M);    #bootstraps
    vsize = size(train_vecs[sp][1],1);
    for m = 1:M
        println("bootstrapping >>> split no: $sp/$n_splits and domain $m/$M")
        𝐖[m] = createBootstrap(train_vecs[sp][m], train_labels[m]; 
                                n_of_boot=n_of_boots, bootsize=bootsize,
                                random_state=random_state, w = :b);     
        # writedlm(joinpath(@__DIR__, "boots$m.txt"), 𝐖[m]);
        # Vector zero mean if enabled
        vecMean ? 𝐖[m] .-= mean(𝐖[m], dims=2) : nothing;
        # Vector normalization if enabled
        vecNorm ? 𝐖[m] ./= mean(norm.(eachcol(𝐖[m]))) : nothing;

        # Normalize feature vectors if enabled
        vecNorm ? train_vecs[sp][m] ./= mean(norm.(eachcol(train_vecs[sp][m]))) : nothing;
        vecNorm ? test_vecs[sp][m] ./= mean(norm.(eachcol(test_vecs[sp][m]))) : nothing;
    end
    # Prewhitening
    T, S, wh = whitenData(𝐖; type = whitening, white_dim = white_dim,
                          smart_subspace = smart_subspace, verbose = verbose);
    
    # JOAL begins
    U_init = joalInitializer(deepcopy(T); type = initialize_U);
    𝐔, _ = joal(deepcopy(T); verbose = verbose, threaded = false, 
                init = deepcopy(U_init), maxiter = 2500, tol = 1e-8);
    
    # Normalization of alignment matrices
    𝐔_ = normU(𝐔; type = normalize_U, 𝐓 = deepcopy(T));

    # Sort 𝐔 matrices before sub dimension selection
    sort_U ? sortU!(𝐔_, T) : nothing;

    # Computation of B matrices
    𝐁[sp] = estimateB(𝐔_, wh; 𝐒 = S, type = whitening, white_dim = white_dim,
                      reverse_selection = false);
end

gl_acc, gl_std = glTraining(train_vecs, test_vecs, train_labels, test_labels, 𝐁;
                            sub_dim = sub_dim, classifier = clf, verbose = verbose);

ksort = sortperm(sw_acc, rev=true);
plot(sw_acc[ksort], labels = "SW")
plot!(gl_acc[ksort], labels = "GL")
ylims!((0.4,1))

