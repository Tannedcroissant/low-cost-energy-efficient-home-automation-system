Dual-Brain Energy-Efficient Smart Home Architecture for Low Cost Edge AI

This is the complete source for my thesis covering localized, energy-efficient smart home automation It brings in a new architecture called the "Dual-Brain" that layers deep learning forecasting with reinforcement learning control over agnostic edge hardware.

Project Abstract

Commercialsmarte homes have a lot of cloud in them — which incurs latencies, has privacy risks and is only as good as the rigid rules definied are re-actively. A fully random, pro-active microservices architecture according to this project.

A 1-Dimensional Deep Convolutional Neural Network (1D-DCNN) and a Bidirectional LSTM (BLSTM) are used for day-ahead load forecasting, along with a DQN agent as real-time HVAC and Matter-over-Thread solar shade actuation, carefully balancing peak-load energy consumption shaving and all users' thermal discomfort minimization.

Repository Structure

Your architecture is separated into 2 separate environments to keep model training and edge deployment concerns completely isolated from each other.

1. offline_training_hub/

Holds the Python environment utilized to process and train neural networks for the thesis dataset (on the main workstation).

src/ — The PyTorch implementations of the DQN and Hybrid Forecaster.

models/: The compiled . pth brain weights.

dataset_generator. data_processing.py`: scripts to process UK-DALE and EnergyPlus data.

2. edge_deployment_hub/

Holds the localized deployment files that are worked on the Intel NUC / Edge Compute Hub.

ha_core_config/: The YAML files of your Home Assistant Core. Manages the physical layer, MQTT message brokering, virtual hardware mapping, and safety overrides that need to be deterministic.

With the use of a rolling memory buffer to populate an 8-Dimensional State Tensor every 5 seconds, this stateless Python container makes heavy machine learning dependencies (PyTorch) isolated from the Home Assistant core.

Technology Stack

Machine Learning : PyTorch, Pandas, NumPy

Edge Orchestration: Home Assistant OS, Docker

Networking Protocol - MQTT (Mosquitto), REST API

Algorithms: 1D-DCNN, BLSTM, DQN (RL)
