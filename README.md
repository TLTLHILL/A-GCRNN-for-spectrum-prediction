# A-GCRNN: Attention Graph Convolution Recurrent Neural Network for Multi-band Spectrum Prediction
This is a PyTorch implementation of A-GCRNN in the following paper: A-GCRNN: Attention Graph Convolution Recurrent Neural Network for Multi-band Spectrum Prediction(https://ieeexplore.ieee.org/document/10251662). 

## Structure
<div align=center><img src="https://github.com/TLTLHILL/A-GCRNN-for-spectrum-prediction/blob/main/photo/A-GCRNN.png" width="500" height="470" /></div>

## Dataset
The dataset for this project comes from the open source platform: https://electrosense.org

|Dataset parameters|Value|
|------------------|-----|
|Dataset source|https://electrosense.org|
|Sensor location|Madrid, Spain|
|Frequency band|500 MHz–800 MHz|
|Monitoring time|2021.5.28–2021.6.28|
|Frequency resolution|2 MHz|
|Time resolution|15 minutes|
|The dimensionality of samples|151 × 2880|

**！！！The opening time of sensors on this platform is uncertain, and there may be some sensors shutdown.**

## Usage
### File description
- Data: Datasets storage file
- models: Models storage file
- photo: Models and some experimental results image storage file
- tasks: Tasks storage file
- utils: Key function code storage file
- adj_create.py: Code for constructing adjacency matrix
- main.py: Training Code
- tesy_main.py: Test Code
### Requirements
- Numpy
- torch
- pytorch-lightning
- pandas
- matplotlib

### Model training
```
python main.py --model_name AGCRNN --max_epochs 3000 --learning_rate 0.0001 --batch_size 64 --hidden_dim 100  --settings supervised --gpus 1
```
### Model test
```
python test_main.py --model_name AGCRNN --max_epochs 3000 --learning_rate 0.0001 --batch_size 64 --hidden_dim 100  --settings supervised --gpus 1
```
### Parameters description
**！！！These parameters can be adjusted independently.**
|Parameters|Description|
|------------------|-----|
|--data|The name of the dataset|
|--seq_len|Required historical data length|
|--pre_len|Predicted data length|
|--split_ratio|Dataset spliting ratio|
|--hidden_dim|Number of GRU hidden layers|

Run `tensorboard --logdir lightning_logs/version_0 --samples_per_plugin scalars=999999999` in terminal to view the prediction results and experimental indicators.
# Citation
Please cite the following paper if you use the code in your work:
```
@ARTICLE{ZhangTVT2023a,
  author={Zhang, Xile and Guo, Lantu and Ben, Cui and Peng, Yang and Wang, Yu and Shi, Shengnan and Lin, Yun and Gui, Guan},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={A-GCRNN: Attention Graph Convolution Recurrent Neural Network for Multi-Band Spectrum Prediction}, 
  year={2023},
  doi={10.1109/TVT.2023.3315450}}
```
