# A-GCRNN: Attention Graph Convolution Recurrent Neural Network for Multi-band Spectrum Prediction
This is a PyTorch implementation of A-GCRNN in the following paper: A-GCRNN: Attention Graph Convolution Recurrent Neural Network for Multi-band Spectrum Prediction.

## Requirements
- Numpy
- torch
- pytorch-lightning
- pandas
- matplotlib
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

**！！！The opening time of sensors on this platform is uncertain, and there may be some sensor shutdowns.**
## Structure
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="[YOUR-DARKMODE-IMAGE](https://github.com/TLTLHILL/A-GCRNN-for-spectrum-prediction/blob/main/photo/A-GCRNN.png)">
 <source media="(prefers-color-scheme: light)" srcset="[YOUR-LIGHTMODE-IMAGE](https://github.com/TLTLHILL/A-GCRNN-for-spectrum-prediction/blob/main/photo/A-GCRNN.png)https://github.com/TLTLHILL/A-GCRNN-for-spectrum-prediction/blob/main/photo/A-GCRNN.png">
 <img alt="YOUR-ALT-TEXT" src="[YOUR-DEFAULT-IMAGE](https://github.com/TLTLHILL/A-GCRNN-for-spectrum-prediction/blob/main/photo/A-GCRNN.png)https://github.com/TLTLHILL/A-GCRNN-for-spectrum-prediction/blob/main/photo/A-GCRNN.png">
</picture>

