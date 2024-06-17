# On Scalable and Interpretable Autism Detection from Social Interaction Behavior

# Paper

## Abstract
Autism Spectrum Condition (ASC) is characterized
by social interaction difficulties that can be challenging to assess
objectively in the diagnostic process. In this paper, we evaluate
the capability of using videos of a standardized social interaction
to differentiate non-verbal behaviors of individuals with and
without ASC. We collected a large video dataset consisting of
164 participants with ASC (n = 83) and neurotypical individuals
(n = 81) who completed the computer-based Simulated Interaction Task (SIT) in different studies including lab and home
settings. To classify individuals with and without ASC, we trained
uni- and multimodal machine learning models based on different
modalities such as facial expressions, gaze behavior, head pose
and voice features. Our results indicate that a multimodal
late fusion approach achieved the highest accuracy (74%). In
the unimodal setting, classification based on facial expressions
(accuracy 73%) and voice features (accuracy 70%) were most
effective. An explainability analysis of the most relevant features
for the facial expression model indicated that features from all
emotional parts as well as from both the speaking and listening
part of the interaction were informative. Based on our results, we
developed a scalable online version of the SIT to collect diverse
data on a large scale for the development of machine learning
models that can differentiate between different clinical conditions.
Our study highlights the potential of machine learning on videos
of standardized social interactions in supporting clinical diagnosis
and the objective and effective measurement of differences in
social interaction behavior.

[Link to paper.](https://doi.org/10.1109/ACII59096.2023.10388157)

# Implementation and results

In the `notebooks` directory you can find all steps, 
which were included in the analysis. 

In the `1_data_preprocessing`, 
we first examine the quality of the videos, to make sure FPS is high enough.
In FPS is not too low and not high enough (15<FPS<25),
videos will be converted and upscaled to 30 FPS.

In the `2_feature_extraction`, 
we run OpenFace toolkit to extract visual information for _Facial_, _Gaze_ and 
_Head_ modalities. For _Voice_ modality OpenSmile will be utilized. 

In the `3_feature_preparation`,
data will be cleaned and combined. Functionals are calculated for each 
interation part of [SIT paradigm](https://doi.org/10.1038/s41746-020-0227-5).

In the `4_classification` notebook code for classification is presented.

# Setup

If you want to run the code and/or notebooks from this repository, 
follow the instructions to create an environment and install all requirements.

* Create an environment with Python 3.12+

```shell 
$ conda create -n sit-autism python=3.12
```

* Activate the environment

```shell 
$ conda activate sit-autism
```

* Install requirements

```shell 
$ pip install -r requirements.txt
```

* Install MBP Package (being in the project directory)

```shell 
$ pip install ./packages/mbp-....whl
```

* Run Jupyter Notebook package to run notebooks in interactive mode

```shell
$ jupyter notebook
```