# DIPP-Differentiable Integrated Prediction and Planning
This repo is the implementation of the following paper:

**Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving**
<br> [Zhiyu Huang](https://mczhi.github.io/), Haochen Liu, [Jingda Wu](https://wujingda.github.io/), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[Project Website]](https://mczhi.github.io/DIPP/)**

## Dataset
Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1; only the files in ```uncompressed/scenario/training_20s``` are needed. Place the downloaded files into training and testing folders separately.

## Installation
### Install dependency
```bash
sudo apt-get install libsuitesparse-dev
```

### Create conda env
```bash
conda env create -f environment.yml
```

### Activate env
```bash
conda activate DIPP
```

### Install Theseus
Install the [Theseus library](https://github.com/facebookresearch/theseus), follow the guidelines.

## Usage
### Processing
Run ```data_process.py``` to process the raw data for training. This will convert the original data format into a set of ```.npz``` files, each containing the data of a scene with the AV and surrounding agents. You need to specify the file path to the original data and the path to save the processed data. You can optionally use multiprocessing to speed up the processing. 
```shell
python data_process.py \
--load_path /path/to/original/data \
--save_path /output/path/to/processed/data \
--use_multiprocessing
```

### Training
Run ```train.py``` to learn the predictor and planner (if set ```--use_planning```). You need to specify the file paths to training data and validation data. Leave other arguments vacant to use the default setting.
```shell
python train.py \
--name DIPP \
--train_set /path/to/train/data \
--valid_set /path/to/valid/data \
--use_planning \
--seed 42 \
--num_workers 8 \
--pretrain_epochs 5 \
--train_epochs 20 \
--batch_size 32 \
--learning_rate 2e-4 \
--device cuda
```

### Open-loop testing
Run ```open_loop_test.py``` to test the trained planner in an open-loop manner. You need to specify the path to the original test dataset (path to the folder) and also the file path to the trained model. Set ```--render``` to visualize the results and set ```--save``` to save the rendered images.
```shell
python open_loop_test.py \
--name open_loop \
--test_set /path/to/original/test/data \
--model_path /path/to/saved/model \
--use_planning \
--render \
--save \
--device cpu
```

### Closed-loop testing
Run ```closed_loop_test.py``` to do closed-loop testing. You need to specify the file path to the original test data (a single file) and also the file path to the trained model. Set ```--render``` to visualize the results and set ```--save``` to save the videos.
```shell
python closed_loop_test.py \
--name closed_loop \
--test_file /path/to/original/test/data \
--model_path /path/to/saved/model \
--use_planning \
--render \
--save \
--device cpu
```
