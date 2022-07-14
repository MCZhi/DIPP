# DIPP-Differentiable Integrated Prediction and Planning
This repo is the implementation of the following paper:

**Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving**
<br> [Zhiyu Huang](https://mczhi.github.io/), Haochen Liu, [Jingda Wu](https://wujingda.github.io/), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[Project Website]](https://mczhi.github.io/DIPP/)**

## Dataset
Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1; only the files in ```uncompressed/scenario/training_20s``` are needed. Place the downloaded files into training and testing folders separately.

## Installation
### Conda environment
```shell
conda create -n DIPP python=3.8
conda activate DIPP
```

### Theseus
Install the [Theseus library](https://github.com/facebookresearch/theseus) and follow the official guideline.

## Usage
### Training
Run imitation_learning_uncertainty.py to learn the imitative expert policies. You need to specify the file path to the recorded expert trajectories. You can optionally specify how many samples you would like to use to train the expert policies.
```shell
python imitation_learning_uncertainty.py expert_data/left_turn --samples 40
```

### Open-loop testing
5. Run train.py to train the RL agent. You need to specify the algorithm and scenario to run, and also the file path to the pre-trained imitative models if you are using the expert prior-guided algorithms. The available algorithms are sac, value_penalty, policy_constraint, ppo, gail. If you are using GAIL, the prior should be the path to your demonstration trajectories.
```shell
python train.py value_penalty left_turn --prior expert_model/left_turn 
```

### Closed-loop testing
Run plot_train.py to visualize the training results. You need to specify the algorithm and scenario that you have trained with, as well as the metric you want to see (success or reward).
```shell
python plot_train.py value_penalty left_turn success
```

Run test.py to test the trained policy in the testing situations, along with Envision to visualize the testing process at the same time. You need to specify the algorithm and scenario, and the file path to your trained model. 
```shell
scl run --envision test.py value_penalty left_turn train_results/left_turn/value_penalty/Model/Model_X.h5
```

Run plot_test.py to plot the vehicle dynamics states. You need to specify the path to the test log file.
```shell
python plot_test.py test_results/left_turn/value_penalty/test_log.csv
```
