# groupLearning.jl

groupLearning.jl is a [Julia](https://julialang.org/) package for performing __many-to-many__ and __many-to-one__ transfer learning on feature sets of multiple domains. 

## Description

This repository implements the **G**roup **Ali**gnment **A**lgorithm (**GALIA**) method [1]. All necessary code and some examples data are provided in order to test and replicate study [1].

## Getting Started

### Dependencies

* Make sure the code is running on [Julia release v1.9.0.](https://julialang.org/downloads/oldreleases/)
* All the required packages (with compatible versions) are provided in *Project.toml* file.

### Installing

* Download the repository.
* Run Julia REPL in the top folder of the repository.
* Instantiate the environment by running
```julia
]instantiate
```

### Notes
* Machine learning models are imported from the scikit-learn Python library. Therefore, Python must be installed on the PC. We suggest to follow the steps on [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) page.

* As an example, recordings of 8 subjects (three session from each) are provided under the folder *exampleData/bi2015a*.

* Other databases can be tested as long as they are in the **NY file** format. Please locate custom databases as subfolders under *data* folder. For other formats you will have to replace the relevant EEG reading functions with your own.

* Please make sure the current working directory is set to the filepath of the repository.

### Running the Group Learning

* Open the **test_pipeline.jl** file.

* Create required objects for pipelines
```julia
# Subject specific train/test pipeline
obj_list = initiateObjects(dbName, filepath);
```

* Create a list comprised of pipeline steps. 
```julia
# Group learning pipeline
pipeline2 = [createTSVectors, prepareGL, runGL, trainGL];

```
* Run the pipeline
```julia
runPipe!(pipeline2, obj_list)
```

* Plot and compare pipelines
```julia
# PLot and compare pipelines
plotAcc(obj_list)
```

## Authors

[Fatih Altindis]() is a Research Assistant at Abdullah Gul Univeristy, Kayseri. ***contact:*** fthaltindis *at* gmail *dot* com

[Marco Congedo](https://sites.google.com/site/marcocongedo), is a Research Director of CNRS (Centre National de la Recherche Scientifique), working at UGA (University of Grenoble Alpes). 

The research on **GALIA** has has been carried out during a visit of Fatih in Grenoble.

## License

This project is licensed under the BSD-3-Clause License

## References
[1] Altindis F., Banerjee A., Phlypo R., Yilmaz B., Congedo M. (2023) Transfer Learning for Brain-Computer Interfaces by Joint Alignment of Feature Vectors, submitted.

[2] [Congedo M., Bleuzé A., Mattout J. (2022) Group Learning by Joint Alignment in the Riemannian Tangent Space GRETSI conference, 6-9 September 2022, Nancy, France.](https://hal.science/hal-03778481v1/document)

[3] [Congedo M., Phlypo R., Chatel-Goldman J. Orthogonal and non-orthogonal joint blind source separation in the least-squares sense. (2012) The 20th European Signal Processing Conference (EUSIPCO), 27-31 August 2012, Bucharest, Romania.](https://ieeexplore.ieee.org/document/6334247)

 
