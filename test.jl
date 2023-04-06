
using 	LinearAlgebra, CovarianceEstimation,
PosDefManifold, PosDefManifoldML,
Diagonalizations, NPZ, YAML, HDF5, EzXML,
BlockDiagonals, DelimitedFiles, TimerOutputs, Base.Threads

# Include required custom packages 
push!(LOAD_PATH, joinpath(@__DIR__, "Modules"))
using 	EEGio, System, FileSystem, EEGpreprocessing, EEGprocessing, ERPs

# include("main_tools.jl")
include("databaseUtilityTools.jl")

# Selected Databases
dbList = "bi2014b";	        # give a list of datasets with their folder name
n_of_subject = 3;		    # number of subjects to be used (set to 0 for using all subjects)
selection_rule = :none;     # selection rule (set :rand for random selection)
n_splits = 5;			    # number of splits for train/test
split_ratio = 50;           # Train/Test ratio of the trials
turn = :local;              # :gricad
paradigm = :ERP;
random_state = 1;
# Preprocessing parameters
bandpass = (1,16);		    # bandpass filter cut-off frequencies
artefact_rejection = 1;		# set 1 for enable artefact rejection
# ERP Prototype estimation & Covariance Estimation
ERP_weight = :a;		    # adaptive weights 
overlapping = true;		    # overlapping of ERPs
PCA_dim = 16;			    # set nothing if no PCA is applied
det_normalization = false;	# normalize determinants of cov mats
ch_union = nothing;         # Union channel space for Dimensionality Transcending
dt = false;

selected_files, subject_list = select_subjects(dbList; n_of_subject = n_of_subject,
                                               selection_rule = selection_rule, turn = :none);


# TEST CODE for chekcing if there is a leakage between train and test splits
include("main_tools.jl")
temp_file = selected_files[1];
o=readNY(temp_file;
         bandpass=bandpass, upperLimit=artefact_rejection);
TAindex = findfirst(isequal("Target"), o.clabels);
for split_ratio in 10:10:90
    for n_splits in (1:10)
        train_splits_no, test_splits_no, y_train, y_test = get_splits(o.mark; 
                    n_splits = n_splits,
                    split_ratio = split_ratio,
                    return_type = :trial_no,
                    random_state = 10);
        for (tr,te) in zip(train_splits_no, test_splits_no)
            train = sort(vcat(tr...))
            test = sort(vcat(te...))
            parity_check = sum([tt in train for tt in test]);
            parity_check == 0 ? @info("No leakage between train and test splits!") : @error("Train and Test splits are overlapping!!!");
        end
        println("The size of the train splits " , string(size(train_splits_no)))
        println("The size of the train splits " , string(size(test_splits_no)))
        train_splits, test_splits = get_splits(o.mark; 
                    n_splits = n_splits,
                    split_ratio = split_ratio,
                    return_type = :index,
                    random_state = 10);
        for (tr,te) in zip(train_splits, test_splits)
            train = sort(vcat(tr...))
            test = sort(vcat(te...))
            parity_check = sum([tt in train for tt in test]);
            parity_check == 0 ? @info("No leakage between train and test splits!") : @error("Train and Test splits are overlapping!!!");
        end
        println("The size of the train splits " , string(size(train_splits_no)))
        println("The size of the train splits " , string(size(test_splits_no)))
    end
end

# TEST CODE for chekcing the overlap between consecutive train or test splits
include("dataTransformation.jl")
for n_splits in (1:10)
    train_splits_no, test_splits_no, y_train, y_test = get_splits(o.mark; 
                    n_splits = n_splits,
                    split_ratio = split_ratio,
                    return_type = :trial_no,
                    random_state = 10);
    parity_check = zeros(size(train_splits_no, 1), size(train_splits_no, 1))
    for i = 1:size(train_splits_no, 1)
        for j = i+1:size(train_splits_no, 1)
            tr_sp1 = vcat(train_splits_no[i]...);
            tr_sp2 = vcat(train_splits_no[j]...);
            parity_check[i,j] = sum([tt in tr_sp1 for tt in tr_sp2])/size(tr_sp1,1);
        end
    end
    display(parity_check)
end

all_clabels = [] 	# Class labels of all subjects
train_vecs = Vector{Matrix{Float64}}[]
test_vecs = Vector{Matrix{Float64}}[]
train_labels, test_labels = Vector{}[], Vector{}[]
for (n,temp_file) ∈ enumerate(selected_files)
    println("processing file $n of $(length(selected_files))...")
    o=readNY(temp_file;
             bandpass=bandpass, upperLimit=artefact_rejection);
    TAindex = findfirst(isequal("Target"), o.clabels);
    # Split the data
    train_splits, _ = get_splits(o.mark; 
            n_splits = n_splits,
            split_ratio = split_ratio,
            return_type = :index,
            random_state = random_state);
    train_splits_no, test_splits_no, y_train, y_test = get_splits(o.mark; 
            n_splits = n_splits,
            split_ratio = split_ratio,
            return_type = :trial_no,
            random_state = random_state);
    # ERP Prototype estimation
    Y = erp_prototype(o.X, o.wl, train_splits, TAindex;
                      overlapping = overlapping,
                      weights = ERP_weight,
                      PCA = PCA_dim);
    # Covariance Estimation
    train_covs, test_covs = estimate_cov_mat(o.trials, train_splits_no, test_splits_no, Y;
                            det_normalization = det_normalization, estimator = :lw);
    if dt
        # Avoid any mismatch of upper/lower case between channel names
        # This part is for correcting a small bug in header data of bi2014b data !!!
        ch_sub = map(x-> lowercase(x), o.sensors)
        if o.db == "bi2014b"
            deleteat!(ch_sub, findall(x->x=="afz",ch_sub))
        end
        train_tsvec, test_tsvec = tangent_vectors(train_covs, test_covs, 1:o.ne, paradigm; 
                transpose=false, dt = dt, ch_union = ch_union, ch_sub = ch_sub,
                PCA_dim = PCA_dim);
    else
        train_tsvec, test_tsvec = tangent_vectors(train_covs, test_covs, 1:o.ne, paradigm; 
                transpose=false)
    end
    # append data
    push!(train_vecs, train_tsvec)
    push!(test_vecs, test_tsvec)
    push!(train_labels, y_train)
    push!(test_labels, y_test)
    append!(all_clabels,[o.clabels])
end
train_vecs, test_vecs = swap_dims(train_vecs, test_vecs);

# Subject-wise learning
sw_acc, sw_std = sw_training(train_vecs, test_vecs, train_labels, test_labels; classifier = :LinearSVC);


# TEST CODE for checking select_subjects funtion on all databases
include("databaseUtilityTools.jl")
include("dbInfo.jl")
# dbList = ["bi2012", "bi2014a", "bi2014b", "bi2015a", "bnci2014001", "bnci2014002",
#           "bnci2014004", "BNCI2014008", "bnci2015001", "BNCI2015003"];
dbList = ["bi2012", "bi2014a", "bi2014b", "bi2015a", "BNCI2014008", "BNCI2015003"];
n_of_subject = 0;
random_state = nothing;
selection_rule = :thre;
turn = :local; 
for db in dbList
    (n_of_subject >= dbSubjectCount[db]) || n_of_subject==0 ? temps = dbSubjectCount[db] : temps = n_of_subject;
    selected_files, subject_list = select_subjects(db; n_of_subject = n_of_subject, threshold = 0.2,
                                                   selection_rule = selection_rule, turn = :none);
    c = size(selected_files,1)
    if c == temps * dbSessionCount[db]
        @info("Database:    $(db)   >>>     Filename list count matches")
    else
        @warn("Database:    $(db)   >>>     There is a missing file(s)!!!")
    end
    println("Number of selected files $c")
end



