# DDPTransformer
This repo is the official implementation of "DDPTransformer: Dual-Domain With Parallel
Transformer Network for Sparse View CT Image Reconstruction". 



## Updates

### 04/30/2022
Create repo



## Package dependencies

The project is built with PyTorch 1.7.1, Python3.8, CUDA10.2 on Ubuntu 20.04 . For package dependencies, you can install them by:
```
conda env create -f env.yaml
```

and you also need to install [torch-radon](https://github.com/matteo-ronchetti/torch-radon) (*Note: you should install v2 branch)



## Data preparation

For all data of this paper, you can download the  dataset [Low Dose CT Image and Projection Data (LDCT-and-Projection-data)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026).

