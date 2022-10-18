# Deep-BVQM
# Development of a Bitstream-based Video Quality Prediction Model Using Deep Learning Techniques for Gaming Streaming Services

This repository provides the code for testing the Deep-BGVQ model, a two-step model designed for predicting the quality of videos at frame level as well as video level using bitstream information. With this repo, the bitstream information of two dataset are provided.  The current model only works for 1080p resolution videos under encoding setting provided in the range of paratmers section. For ease of testing, a dataset named Kingston University Gaming Video Dataset (KUGVD) is provided, along with corresponding VMAF and MOS values. 

## Access to the paper
The full paper published by the ACM Digital Library, is available at this [link](https://dl.acm.org/doi/abs/10.1145/3503161.3548374).

## Kingston University Gaming Video Dataset (KUGVD)

KUGVD is one of the datasets available for estimating game video quality. It includes six raw game videos with 30-second durations, 30 frames per second, 24 resolution-bitrate pairings encoded with H.264/MPEG-AVC, and in total 144 distorted gaming sequences for training. In addition, VMAF and MOS values for 90 sequences are measured, which may be utilized as a validation set. In this repo only the 1080p resolution data are provided. 

## Range of Parameters

The proposed model works under the following encoding parameters:
```
Codec: H.264
Resolution: 1080p
Frame rate: 30 and 60 fps
Rate controller: CBR
Preset: veryfast, medium, llhq
```
## Input to the model

As inputs to the mode, the bitstream file in CSV format must be provided to the model. The bitstream file can be easily extracted using ffmpeg-qp-parse using mode 3 of the model. If the VMAF and MOS values are available, they can be added in a separate file for evaluation purposes. In this repo, an example of how to sort the input data is provided. 

## How to use

This model offers two options to execute the model and obtain the predicted quality, depends on the availability of VMAF values and MOS scores. In the first option, not only the model will be executed, but also the evaluation in terms of PLCC, RMSE and Scatter plot will be provided. 

**_It should be noted that due to the storage limitation in GitLab, the input files including MOS file, VMAF file, and Bitstream Dataset have been uploaded on TU Berlin cloud. After downloding the folder 'Inputs' from [here](https://tubcloud.tu-berlin.de/s/dZNAkaFQipmC8K4), please place it in the folder 'Dataset' downloaded from the repository. At the end, the final structure will consist of two folders: 'Code' and 'Dataset'; under the 'Dataset,' there will be three subfolders: 'Inputs', 'Models', and 'Results'. TubCloud must be used to download the 'Input’ subdirectory_


#### Option 1

If VMAF and MOS values are available, as in the dataset KUGVD provided.  Then, execute the following code:

```
    python validation_for_lstm_withVMAFandMOS.py  
        --MOS=KUGVD_MOS.xlsx
        --VMAF=KUGVD_VMAF_PerFrameValues.xlsx 
        --ResPath=../Dataset/Results/
        --InputPath= ../Dataset/Inputs/ 
        --ModelPath=../Dataset/Models/
```

#### Option 2
If VMAF and MOS values are unavailable, run the following code;
```
    python validation_for_lstm_withoutVMAFandMOS.py  
        --ResPath=../Dataset/Results/
        --InputPath=../Dataset/Inputs/ 
        --ModelPath=../Dataset/Models/
```
## Output of the model

Depending on the option you chose to execute the code, the model returns the predicted VMAF (model output at frame level) as well as the predicted MOS per second and per video (model output at video level) as excel files. In addition, if you choose option 1, the model provides you with two scatter plots comparing the actual and predicted values.

## Requirements

The libraries necessary to run the code are listed below;

- keras
- torch
- argparse
- matplotlib
- scipy.stats 
- sklearn
- pickle
- pandas
- numpy
- math
- glob
