# Local-Sensing Action Masking: Learning Transferable Zero-Collision Policies for Decentralized Multi-Agent

This the repository for the conference paper of 

"Local-Sensing Action Masking: Learning Transferable Zero-Collision Policies for Decentralized Multi-Agent"

## About
This project focuses on Multi-AGV scenario, and learning transferable zero-collsion MARL policies from grid-based environments to continous settings. Including the repo, you can find:

* **Core code of the Local-Sensing Action Masking~(LSAM)**: The implementation of LSAM with MAPPO algrithom.
* **Grid-based environment**: The grid-based environment of the multi-AGV logistic sorting application.
* **Continous environment in MuJoCo**: Corresponding MuJoCo environments of continous environments.
* **Code for training**: Training the policies used in the paper.
* **Code for evalutation**: Replicating the results of performance in the paper.


## Project Structure and key files

```text
├── eval/                     # Folder of the original project 
    └── eval_model.py         # Entry file of evalutation 
├── training/                 # Folder of the original project 
    └── train.py              # Entry file of training
├── environment.yml           # environment file for conda
├── human_play.py             # entry point for human_play test
├── maps.py                   # the layout of all maps in paper (a-d)
└── README.md           
```

## Installation / Environment

### Environment Specifications:

* Python: 3.7.12

* Conda: 25.11.1

### Install requirements:

```bash
# Only support Conda now
conda env create -f environment.yml
conda activate lsam
```

## Quick Start

### Human play our grid maps: 

```bash
python human_play.py --map "a"
```
* **--map**: choose a map in the paper from a-d, can be "a", "b", "c" or "d" ,default: "a"

You can control the interaction with the following keys:
- Up Arrow keys: move current agent forward
- Left/ Right Arrow keys: rotate current agent left/ right
- P: pickup / place workpiece
- SPACE: do nothing
- TAB: change the current agent
- ESC: exit

### Repilicating our testing: 

```bash
cd eval
python eval_model.py --map "a" --eval_pattern "base" --episodes 100 --render_speed "slow"
```

Help for the command:

* **--map**: choose a map in the paper from a-d, can be "a", "b", "c" or "d" ,default: "a"
* **--eval_pattern**: choose evalution pattern in the paper, default: "base", can be "base", "cont", "lsam" where $i \in \{a, b, c, d\}$: 
    * "base" is the Benchmark tests of $\pi_i$, 
    * "cont" is the Continuous tests of $\pi_i$, 
    * "lsam" is the Continuous tests of $\pi_i'$, 
* **--episodes**: set the number of the test episode, corresponding to the $E_{total}$ in the paper, default 10
* **--render_speed**: render mode "slow" or "fast".
    * **"slow"**: insert a 0.5 seconds sleep between two step if it is  for huamn observation 
    * **"fast"**:0 second sleep for fast testing.

### Train new policies: 

```bash
cd training
python train.py --map "a" --lsam 0 --num_env_steps 100000 --n_rollout_threads 4
```
Help for the command:

* **--map**: choose a map in the paper from a-d, can be "a", "b", "c" or "d" ,default: "a"
* **--lsam**: flag of using LSAM during the training
    * 0: not using LSAM
    * 1: using LSAM
* **--num_env_steps**: set the number of the training step, default 20e5
* **--n_rollout_threads**: number of parallel envs for training, default 4.

## References and Acknowledgements

We thank the authors of MAPPO/Light-MAPPO and RWARE for their open-source implementations.

* **MAPPO**: Yu, C., et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS 2022. [[Paper]](https://arxiv.org/abs/2103.01955) [[Code]](https://github.com/marlbenchmark/on-policy)

* **Light-MAPPO**: Zhiqiang H., light_mappo: Lightweight MAPPO implementation.[[Code]](https://github.com/tinyzqh/light_mappo)

* **RWARE**: Papoudakis, G., et al. "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks." NeurIPS 2021. [[Paper]](https://arxiv.org/abs/2006.07869) [[Code]](https://github.com/semitable/robotic-warehouse)