# Saliency-Ranking
Code release for the TPAMI 2021 paper "Instance-Level Relative Saliency Ranking with Graph Reasoning" by Nian Liu, Long Li, Wangbo Zhao, and Junwei Han.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

![avatar](image.png)

## Installation
See [INSTALL.md](INSTALL.md).

## Data Preparation
Download the datatset from [Baidu Driver](https://pan.baidu.com/s/1vzH_av0zCFhTL4WqpbTVmQ) (zsqn) or [Goggle Driver](https://drive.google.com/file/d/1R-S9yT0khNehAaA1M13N0AQGOicJS7uh/view?usp=sharing) and unzip them to './dataset'. Then the structure of the './dataset' folder will show as following:

````
-- dataset
   |-- Annotations
   |   |-- | train.pkl
   |   |-- | test.pkl
   |-- Images
   |   |-- train
   |   |-- |-- | image
   |   |-- |-- | gt
   |   |-- test
   |   |-- |-- | image
   |   |-- |-- | gt
````
