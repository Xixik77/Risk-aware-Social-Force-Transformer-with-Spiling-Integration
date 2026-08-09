# Risk-aware Social Force Transformer with Spiking Integration

This repository contains the code for the paper:

**Risk-aware Social Force Transformer with Spiking Integration: A Hybrid ANN-SNN Architecture for Pedestrian Trajectory Prediction**

---

## 📋 Introduction

This repository provides the source code for pedestrian trajectory prediction based on a hybrid architecture that integrates **Transformer**, **Spiking Neural Networks (SNNs)**, and **social force priors**. The model employs a dual-branch temporal encoder combining standard Transformer layers with LIF-based SNN, fused via a learnable weight. A social force module explicitly computes pairwise repulsive forces between the target pedestrian and its neighbors, injecting physics-based interaction knowledge. The decoder with attention over neighbor features generates final multi-modal trajectory predictions. The approach is validated on **CITR** and **DUT** pedestrian-vehicle interaction datasets.

The repository supports:

- Dual-branch temporal encoding (ANN + SNN) with learnable fusion
- Explicit social force computation for interaction modeling
- Mode-aware training with clustered motion patterns
- Top-K multi-modal trajectory inference
- Modular ablation studies (SNN / social force can be independently enabled/disabled)

---

## ⚙️ Dependencies

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib

Required Python packages:

```bash
torch>=1.10.0
numpy>=1.19.0
matplotlib>=3.3.0
🗂 Project Structure
text
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore                    # 新增：忽略临时文件
│
├── config/
│   ├── __init__.py               
│   └── default_config.py         
│
├── models/
│   ├── __init__.py
│   ├── TrajectoryModel.py
│   ├── Encoder.py
│   ├── Decoder.py
│   ├── snn_layers.py
│   └── model_utils.py            
│
├── data_utils/                  
│   ├── __init__.py
│   ├── dataset.py                
│   └── dataloader.py             
│
├── scripts/
│   ├── train.py
│   └── test.py
└── checkpoint/                  
    └── (dataset_name)/
        ├── best.pth
        ├── last.pth
        ├── training_log.csv
        ├── loss_curve.png
        ├── ade_fde_curve.png
        └── final_test_outputs/
            ├── metrics.txt
            ├── best_pred_trajs.npy
            └── gt_trajs.npy
🚀 Quick Start
Installation
bash
git clone https://github.com/Xixik77/Risk-aware-Social-Force-Transformer-with-Spiling-Integration.git
cd Risk-aware-Social-Force-Transformer-with-Spiling-Integration
pip install -r requirements.txt
Data Preparation
Place your dataset .pkl files under the data/ directory. The expected format:

Training file: {dataset_name}_train.pkl

Test file: {dataset_name}_test.pkl

Each .pkl contains a list of samples, where each sample is a tuple:

ped_obs: [obs_len, 2] — observed trajectory

ped_pred: [pred_len, 2] — ground truth future trajectory

neighbors: [N, obs_len+pred_len, 2] — neighboring trajectories

Supported datasets: CITR, DUT

Note: The current get_motion_modes in utils.py generates random motion modes for demonstration. Replace it with your clustering implementation (e.g., K-means on training trajectories) for real experiments.

Training
bash
python scripts/train.py \
    --dataset_path ./data \
    --dataset_name citr \
    --hp_config config/ETH_config.py \
    --use_snn 1 \
    --use_social_force 1 \
    --obs_len 8 \
    --pred_len 12 \
    --gpu 0
Argument	Description	Default
--use_snn	Enable/disable SNN branch	1 (enabled)
--use_social_force	Enable/disable social force module	1 (enabled)
--obs_len	Observation horizon (frames)	8
--pred_len	Prediction horizon (frames)	12
Evaluation
bash
python scripts/test.py \
    --dataset_path ./data \
    --dataset_name citr \
    --hp_config config/ETH_config.py \
    --checkpoint ./checkpoint/citr/best.pth \
    --use_snn 1 \
    --use_social_force 1 \
    --num_k 20 \
    --save_traj
Results are saved to:

text
./checkpoint/{dataset_name}/final_test_outputs/
├── metrics.txt          # ADE, FDE, num_traj
├── best_pred_trajs.npy  # Best predicted trajectories
└── gt_trajs.npy         # Ground truth trajectories
Visualization
python
import numpy as np
import matplotlib.pyplot as plt

pred = np.load("final_test_outputs/best_pred_trajs.npy", allow_pickle=True)
gt = np.load("final_test_outputs/gt_trajs.npy", allow_pickle=True)

for i in range(min(5, len(pred))):
    plt.plot(pred[i][:, 0], pred[i][:, 1], 'r-', label='Predicted')
    plt.plot(gt[i][:, 0], gt[i][:, 1], 'b-', label='Ground Truth')
    plt.legend()
    plt.title(f'Sample {i}')
    plt.show()

Key Components
Component	Description
Dual-branch Encoder	Transformer (ANN) + LIF-based SNN with learnable fusion weight α
Social Force Module	Computes pairwise repulsive forces between target and neighbors (Helbing & Molnár, 1995)
Social Decoder	Attention-based decoder conditioning on neighbor features
Mode-aware Training	Uses clustered motion patterns as supervision
Top-K Inference	Supports multi-modal trajectory sampling (best-of-K)


📝 Citation
If you find this work useful, please consider citing:

bibtex
@article{yourpaper2026,
  title={Risk-aware Social Force Transformer with Spiking Integration: A Hybrid ANN-SNN Architecture for Pedestrian Trajectory Prediction},
  author={Your Name},
  journal={},
  year={2026}
}
🤝 Contributing
We welcome and encourage contributions to this project! Whether it's reporting bugs, suggesting new features, or improving existing functionality, your input is highly valuable. You can open an issue on our GitHub Issues page to share feedback or raise problems.

If you'd like to contribute directly, feel free to fork this repository and submit a pull request. Be sure to include necessary tests, comments, and documentation with your code changes.

We also recommend following best practices for version control and keeping changes consistent with the project's structure.

📄 License
Distributed under the MIT License. See the LICENSE file for more information.

🙏 Acknowledgements
This project is developed based on our previous work on social force modeling and pedestrian trajectory prediction.

The SNN integration design was inspired by spiking neural network practices in PyTorch.

The social force model is inspired by Helbing & Molnár (1995).

CITR and DUT datasets are from the pedestrian-vehicle interaction research community
