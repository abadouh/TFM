## Analyzing European Deep-Learning libraries with Industry Standard Benchmark

In this project we use three different DL libraries: **Keras-TF**, **PyTorch** and **EDDL/ECVL**.  
The Aim of this work is to give a baseline comparision between the three libraries following the guidelines of [MLPerf](https://mlperf.org/) benchmark.


**EDDL** is an open source library for Distributed Deep Learning and Tensor Operations in C++ for **CPU**, **GPU** and **FPGA**,  [EDDL-Documentation](https://deephealthproject.github.io/eddl/)  
**ECVL** is an open source library for Computer Vision in C++ for **CPU**, **GPU** and **FPGA**. [ECVL-Documentation](https://deephealthproject.github.io/ecvl/)  
EDDL and ECVL are developed inside the DeepHealth project. For more information about DeepHealth project go to: Deep-Health [Official website](https://deephealth-project.eu/) or [git](https://github.com/deephealthproject)  

## Notice
Currently, the project support only two datesets, [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ISIC](https://www.isic-archive.com/).  
1. **CIFAR-10**: In order to use this dataset, you need to download it using the DL library you intend to use (i.e. Keras, PyTorch, ECVL/EDDL).
2. **ISIC**: This dataset can be found in the DeepHealth's git[Use case](https://github.com/deephealthproject/use_case_pipeline). Note: to run it with Keras, you will have to split the dataset into directories base on their class and the dataset it's belong (Train/Validation/Test). The tags available in the YML file.


## PyEDDL and PyECVL

In order to use this project you have to also install the Python wrappers of EDDL and ECVL:
1. [PyEDDL](https://github.com/deephealthproject/pyeddl)
2. [PyECVL](https://github.com/deephealthproject/pyecvl)


