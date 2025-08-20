# PIN-WM : Learning Physics-INformed World Models for Non-Prehensile Manipulation


### [Project Page](https://pinwm.github.io) | [Paper](https://arxiv.org/abs/2504.16693)

#### Wenxuan Li*, Hang Zhao*, Zhiyuan Yu, Yu Du, Qin Zou, Ruizhen Hu†, Kai Xu†

\*Equal Contribution, †Corresponding Authors

> While non-prehensile manipulation (e.g., controlled pushing/poking) constitutes a foundational robotic skill, its learning remains challenging due to the high sensitivity to complex physical interactions involving friction and restitution. To achieve robust policy learning and generalization, we opt to learn a world model of the 3D rigid body dynamics involved in non- prehensile manipulations and use it for model-based reinforcement learning.We propose PIN-WM, a Physics-INformed World Model that enables efficient end-to-end identification of a 3D rigid body dynamical system from visual observations. Adopting differentiable physics simulation, PIN-WM can be learned with only few-shot and task-agnostic physical interaction trajectories. Further, PIN-WM is learned with observational loss induced by Gaussian Splatting without needing state estimation. To bridge Sim2Real gaps, we turn the learned PIN-WM into a group of Digital Cousins via physics-aware perturbations which perturb physics and rendering parameters to generate diverse and meaningful variations of the PIN-WM.Extensive evaluations on both simulation and real-world tests demonstrate that PIN- WM, enhanced with physics-aware digital cousins, facilitates learning robust non-prehensile manipulation skills with Sim2Real transfer, surpassing the Real2Sim2Real state-of-the-arts.

<!-- ![image](figs/teaser.png) -->
<div align=center>
<img src=figs/overview.jpg width=95%/>
</div>


## TODO

- [x] Release the code of world model training.
- [ ] Release the code of data preparation.
- [ ] Release the code of policy training.

### Installation
```bash
git clone https://github.com/XuAdventurer/PIN-WM.git
cd PIN-WM
conda env create -f environment.yaml
conda activate pinwm
```



### Training the world model
```bash
python ./scripts/run.py
```

### Acknowledgement
Parts of the code are modified from  [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [diffsdfsim](https://github.com/EmbodiedVision/diffsdfsim). Thanks to the original authors.



## Citation
~~~
@article{li2025pin,
        title={PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation},
        author={Li, Wenxuan and Zhao, Hang and Yu, Zhiyuan and Du, Yu and Zou, Qin and Hu, Ruizhen and Xu, Kai},
        journal={arXiv preprint arXiv:2504.16693},
        year={2025}
      }
~~~