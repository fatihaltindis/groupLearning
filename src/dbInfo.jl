# Create paradigm dictionary for available databases
dbListParadigm = Dict(
    "bi2012"        => :ERP,
    "bi2014a"       => :ERP,
    "bi2014b"       => :ERP,
    "bi2015a"       => :ERP,
    "bnci2014001"   => :MI,
    "bnci2014002"   => :MI,
    "bnci2014004"   => :MI,
    "BNCI2014008"   => :ERP,
    "bnci2015001"   => :MI,
    "BNCI2015003"   => :ERP);

# Assign prefix for available databases
dbListPrefixes = Dict(
    "bi2015a"       => "subject_",
    "bi2014b"       => "subject_",
    "bi2014a"       => "subject_",
    "bi2012"        => "subject_",
    "bnci2014001"   => "A",
    "bnci2015001"   => "S",
    "bnci2014002"   => "S",
    "bnci2014004"   => "B",
    "bnci2015004"   => "",
    "BNCI2014008"   => "subject_",
    "BNCI2015003"   => "subject_");

# Assign modality
dbModalities = Dict(
    "bi2012"        => "P300",
    "bi2014a"       => "P300",
    "bi2014b"       => "P300",
    "bi2015a"       => "P300",
    "bnci2014001"   => "MI",
    "bnci2014002"   => "MI",
    "bnci2014004"   => "MI",
    "BNCI2014008"   => "P300",
    "bnci2015001"   => "MI",
    "BNCI2015003"   => "P300");

# Assign session numbers
dbSessionCount = Dict(
    "bi2012"        => 1,
    "bi2014a"       => 1,
    "bi2014b"       => 3,
    "bi2015a"       => 3,
    "bnci2014001"   => "MI",
    "bnci2014002"   => "MI",
    "bnci2014004"   => "MI",
    "BNCI2014008"   => 1,
    "bnci2015001"   => "MI",
    "BNCI2015003"   => 2);

# Assign subject numbers
dbSubjectCount = Dict(
    "bi2012"        => 22,
    "bi2014a"       => 64,
    "bi2014b"       => 31,
    "bi2015a"       => 42,
    "bnci2014001"   => "MI",
    "bnci2014002"   => "MI",
    "bnci2014004"   => "MI",
    "BNCI2014008"   => 8,
    "bnci2015001"   => "MI",
    "BNCI2015003"   => 10);