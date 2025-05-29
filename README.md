# Process-and-Forward: Deep Joint Source-Channel Coding Over Cooperative Relay Networks

This repository contains the source code for the paper titled *"Process-and-Forward: Deep Joint Source-Channel Coding Over Cooperative Relay Networks"*, published in the IEEE Journal on Selected Areas in Communications (JSAC) in 2025.

**Paper Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/10845815)

## Overview
We implement both amplify-and-forward (AF) and process-and-forward (PF) for the (three-node) cooperative relay networks working under both half-duplex and full-duplex mode.

Key features:

- **Transformer-based Models**: Utilizes vision transformer architectures for effective feature extraction and transmission.
- **Half- and Full-Duplex Relays**: Supports relay nodes working on different modes.
- **Adaptive Transmission**: In full-duplex scenarios, implements a block-based transmission strategy allowing all the three nodes to adapt to the transmit powers and channel SNR values.


## Repository Structure

- `get_args.py`: Parses command-line arguments for various scripts.
- `modules.py`: Defines transformer-based neural network modules.
- `modules_cnn.py`: Also supports convolutional neural network (CNN) modules.
- `relay_network_fd.py`: Implements the full-duplex relay network model.
- `relay_network_hd.py`: Implements the half-duplex relay network model.
- `run_fd_transformer.py`: Script to train and evaluate the full-duplex transformer model.
- `run_fd_transformer_cnn.py`: Script to train and evaluate the full-duplex CNN model.
- `run_transformer_hd_time.py`: Script to train and evaluate the half-duplex transformer model.



## Running Experiments 

**[Example]** To train the full-duplex transformer model (use the parameters in `get_args.py'):

```bash
python run_fd_transformer.py
```

## Citation

If you find this work useful in your research, please cite the following paper:

```bibtex
@article{bian2025process,
  title={Process-and-Forward: Deep Joint Source-Channel Coding Over Cooperative Relay Networks},
  author={Bian, Chenghong and Shao, Yulin and Wu, Haotian and Ozfatura, Emre and Gunduz, Deniz},
  journal={IEEE Journal on Selected Areas in Communications},
  year={2025},
  volume={43},
  number={5},
  pages={1234--1245},
  doi={10.1109/JSAC.2025.10845815}
}
```
