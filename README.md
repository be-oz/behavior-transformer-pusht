# Behavior Transformer on PushT (LeRobot)

This repository contains my submission for the Sereact AI Imitation Learning Challenge. It implements a simplified version of the Behavior Transformer algorithm on the `PushT` environment using the [LeRobot](https://github.com/huggingface/lerobot) framework.

Author: **Berke Özmen**

---

## Overview

This project implements a two-stage behavior cloning pipeline:
- **Stage 1**: Coarse classification via KMeans discretization of actions.
- **Stage 2**: Fine-grained residual regression with a Transformer model.

The model is trained using only **2D agent position** (no vision or object state) in alignment with challenge constraints. A full ablation study is also included, using an MLP baseline for comparison.

---

## Project Structure

```
.
├── train_transformer.py         # Transformer training
├── run_my_policy_eval.py        # Policy rollout/evaluation
├── my_policy.py                 # Transformer architecture
├── my_policy.pt                 # Trained model weights
├── transformer_dataset.npy      # Expert training dataset
├── action_codebook.pt           # KMeans bin centroids
├── training_curves.png          # Training loss visualization
├── report.pdf                   # Final 2-page LaTeX report
│
├── ablation/                    # MLP baseline
│   ├── mlp_bc_train.py
│   ├── mlp_policy_eval.py
│
├── scripts/                     # Helper utilities
│   ├── collect_expert_trajectories.py
│   ├── process_expert_data.py
│   ├── visualize.py, etc.
```

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- LeRobot (install from [GitHub](https://github.com/huggingface/lerobot))
- NumPy, scikit-learn, matplotlib
- Gymnasium

```bash
conda create -n lerobot python=3.9  // 3.10 also viable
conda activate lerobot
pip install torch torchvision
git clone https://github.com/huggingface/lerobot
cd lerobot && pip install -e .
```

---

## Training (Transformer)

```bash
python train_transformer.py
```

This will train the Behavior Transformer on the preprocessed expert dataset (`transformer_dataset.npy`) using 2D state observations. Training curves will be saved as `training_curves.png`.

---

## Evaluation

```bash
python run_my_policy_eval.py
```

This loads `my_policy.pt`, runs a rollout in the PushT-v0 environment, and logs final reward and behavior. The policy only uses 2D positional input.

---

## Ablation Study (MLP Baseline)

A simpler multi-layer perceptron (MLP) policy was trained using the same dataset to evaluate architectural impact.

```bash
cd ablation
python mlp_bc_train.py       # Train baseline MLP
python mlp_policy_eval.py    # Evaluate baseline MLP
```

Results show that while the MLP learns local movement dynamics, it performs worse in long-horizon tasks than the Transformer.

---

## Results

| Model         | Classification Loss ↓ | Regression Loss ↓ | Task Success |
|---------------|------------------------|--------------------|--------------|
| Transformer   | 0.184                  | 0.0055             | ✘ (Qualitative rollout only) |
| MLP Baseline  | 0.293                  | 0.0082             | ✘ |

**Note:** Full training curves and rollouts are available in the repo.

---

## Report

The full 2-page report is provided as `report.pdf` and includes:
- Data preprocessing
- Architecture and rationale
- Design decisions and challenges
- Future improvements
- Ablation study
- Acknowledgments

---

## Acknowledgments

Special thanks to **sereact.ai** team and **sereact GmbH** for this challenge. It provided an excellent opportunity to explore imitation learning in robotic control using Transformer architectures.

---

## License

MIT License.
