# Code for paper "Multi-agent Reinforcement Learning-based Joint Precoding and Phase Shift Optimization for RIS-aided Cell-Free Massive MIMO Systems"

This is a code package related to the following scientific article:

Y. Zhu, E. Shi, Z. Liu, J. Zhang and B. Ai, "Multi-agent Reinforcement Learning-based Joint Precoding and Phase Shift Optimization for RIS-aided Cell-Free Massive MIMO Systems," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2024.3392883

The package contains a demo environment, based on Python, and core part of our algorithm. We encourage you to also perform reproducible research! And feel free to email me for more information!

## Abstract of the article

Cell-free (CF) massive multiple-input multiple-output (mMIMO) is a promising technique for achieving high spectral efficiency (SE) using multiple distributed access points (APs). However, harsh propagation environments often lead to significant communication performance degradation due to high penetration loss. To overcome this issue, we introduce the reconfigurable intelligent surface (RIS) into the CF mMIMO system as a low-cost and power-efficient solution. In this paper, we focus on optimizing the joint precoding design of the RIS-aided CF mMIMO system to maximize the sum SE. This involves optimizing the precoding matrix at the APs and the reflection coefficients at the RIS. To tackle this problem, we propose a fully distributed multi-agent reinforcement learning (MARL) algorithm that incorporates fuzzy logic (FL). Unlike conventional approaches that rely on alternating optimization techniques, our FL-based MARL algorithm only requires local channel state information, which reduces the need for high backhaul capacity. Simulation results demonstrate that our proposed FL-MARL algorithm effectively reduces computational complexity while achieving similar performance as conventional MARL methods.

## Installation

**Note**: This repository exclusively encompasses the core algorithm and `main.py`. The environment provided is merely a framework and is **SOLELY UTILIZED FOR DEMONSTRATION PURPOSES**. Therefore, it is imperative to set up the environment before executing the code.

```bash
pip install -r requirements.txt
```

## Running the Code

Upon completing the environment setup, execute the following command:

```bash
python main.py --<args>
```

to initiate the code. For comprehensive details, kindly execute `python main.py -h` or refer to the code within `main.py`.

## License and Referencing

This code package is licensed under the MIT license. If you in any way use this code for research that results in publications, please cite our original article.

```
@ARTICLE{10508095,
  author={Zhu, Yiyang and Shi, Enyu and Liu, Ziheng and Zhang, Jiayi and Ai, Bo},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Multi-agent Reinforcement Learning-based Joint Precoding and Phase Shift Optimization for RIS-aided Cell-Free Massive MIMO Systems}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Precoding;Optimization;Fuzzy logic;Reconfigurable intelligent surfaces;Symbols;Wireless communication;Training;Reconfigurable intelligent surface;cell-free massive MIMO;precoding;spectral efficiency;multi-agent reinforcement learning},
  doi={10.1109/TVT.2024.3392883}}
```