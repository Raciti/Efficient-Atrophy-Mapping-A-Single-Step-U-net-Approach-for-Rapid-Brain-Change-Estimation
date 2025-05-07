# Efficient Atrophy Mapping: A Single-Step U-net Approach for Rapid Brain Change Estimation

The estimation brain atrophy can be crucial in the
evaluation of brain diseases and can help to analyze their progression. Existing methods compute the atrophy map from two
Magnetic Resonance Imaging (MRI) scans of the same subject,
which has limitations in terms of evaluation time, often due to
the multi-step process. In this work we proposed a new technique
for atrophy map calculation. It is designed to estimate the change
between two MRI scans with the aim of significantly reducing the
execution time. The consecutive subject time points are evaluated
by a simple U-net which shows the goodness of a single-step
process. We train and evaluate our system on a dataset consisting
of 2000 T1-weighted MRI scans sourced from ADNI and OASIS dataset.
Experimental results demonstrated a considerably reduction in
execution time while maintaining the performance in terms of
atrophy mapping. We believe that this pipeline could significantly
benefit clinical applications regarding the measurement of brain
atrophy. <br>
Our model manages to achieve, with our GPU and CPU, a flow given two magnetic resonances in **0.02s** using the *GPU* and in **2.67s** using the *CPU*, the first iterations suffer from a latency time due to loading the model into the cache.
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/Raciti/A-faster-algorithm-for-brain-change-estimation/blob/main/img/tg_A.png" alt="example input" width="200" height="250" />
    <img src="https://github.com/Raciti/A-faster-algorithm-for-brain-change-estimation/blob/main/img/tg_c.png" alt="example input" height="250"/>
    <img src="https://github.com/Raciti/A-faster-algorithm-for-brain-change-estimation/blob/main/img/tg_s.png" alt="example input" height="250"/>
</div>

# Getting Started
To begin with, in the Environment folder there is a file named [environment.yaml](https://github.com/Raciti/A-faster-algorithm-for-brain-change-estimation/blob/main/Environment/environment.yml), it contains all the necessary libraries to be able to use our model. <br>
To create an environment with the following .yaml file using conda run the following code:

    conda env create -f environment.yaml

# Pretrained Model
In the Model folder is the already trained model ready to be used, [Model](https://github.com/Raciti/A-faster-algorithm-for-brain-change-estimation/blob/main/Model/Unet.pth). <br>
If you do not want to use the model already trained in the Code folder there is the scrip to perform the training called *trainBasicUNet.py*, in the same folder there is a README.md file that explains how to use the script.

# Using the Model
The model can be used by providing a pair of MRIs without a skull and already recorded between them, they can be easily obtained through many tools, with a size of 121×145×113. 
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/Raciti/A-faster-algorithm-for-brain-change-estimation/blob/main/img/MRI_skullStripped.png" alt="example input" width="600"/>
</div>

The output of the model will have to be **normalized** to return it to the original range to do so just use the following formula $(result - 2000) / 1000$.

# Example of Usage
In the Code folder is prsetne a file called *results.py*, in this file the model is used to test its operation and draw conclusions, it is very simple to understand its operation just exclude the comparison with the GT.

# Readyto use model
There is the file [use.py](https://github.com/Raciti/A-faster-algorithm-for-brain-change-estimation/blob/main/Code/use.py) which allows by specifying **path datset csv** and **path directory to save flow** to use the model, with everything already implemented and ready.

# Reference
At the following link is the ([paper](https://ieeexplore.ieee.org/document/10796213)).
