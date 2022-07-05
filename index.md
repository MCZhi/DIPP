# Differentiable Integrated Multi-agent Prediction and Motion Planning Framework with Learnable Cost Function for Autonomous Driving

[AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)

### Abstract

Predicting the future states of surrounding traffic participants and planning a safe, smooth, and socially compliant trajectory accordingly is crucial for autonomous vehicles. However, there are two major issues with the current autonomous driving system: the prediction module is often decoupled from the planning module and the cost function for planning is hard to specify and tune. Therefore, we propose an end-to-end differentiable framework that integrates prediction and planning modules and is able to learn the cost function from data. Specifically, we employ a differentiable nonlinear optimizer as the motion planner, which takes as input the predicted trajectories of surrounding agents given by the neural network and optimizes the trajectory for the autonomous vehicle, thus enabling all operations in the framework to be differentiable including the cost function weights. The proposed framework is trained on a large-scale real-world driving dataset with the objective to reproduce human driving trajectories in the entire driving scene and tested in both open-loop and closed-loop manners. The results reveal that the proposed method outperforms the baseline methods, especially in the closed-loop test where the baseline methods suffer from distributional shifts. We also demonstrate that our proposed method delivers planning-aware prediction results, allowing the planning module to output close-to-human trajectories.

### Prediction

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Planning

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/MCZhi/DIPP/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.
