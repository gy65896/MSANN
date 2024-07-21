 # <p align=center> [TAI 2023] Blind Image Despeckling Using a Multiscale Attention-Guided Neural Network</p>

<div align="center">
 
[![Paper](https://img.shields.io/badge/MSANN-Paper-red.svg)](https://ieeexplore.ieee.org/abstract/document/10012299)

</div>
<div align=center>
<img src="https://github.com/user-attachments/assets/773d79c4-f33a-437c-8a2a-7dd67f58597f" width="720">
</div>

---
>**Blind Image Despeckling Using a Multiscale Attention-Guided Neural Network**<br>  [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN), [Yuxu Lu](https://scholar.google.com.hk/citations?user=XXge2_0AAAAJ&hl=zh-CN), [Ryan Wen Liu](http://mipc.whut.edu.cn/index.html)<sup>* </sup>, Fenghua Zhu <br>
(* Corresponding Author)<br>
>IEEE Transactions on Artificial Intelligence

> **Abstract:** *Coherent imaging systems have been applied in the detection of target of interest, natural resource exploration, ailment diagnosis, etc. However, it is easy to generate speckle-degraded images due to the coherent interference of reflected echoes, restricting these practical applications. Speckle noise is a granular interference that affects the observed reflectivity. It is often modeled as multiplicative noise with a negative exponential distribution. This nonlinear property makes despeckling of imaging data an intractable problem. To enhance the despeckling performance, we propose to blindly remove speckle noise using an intelligent computing-enabled multiscale attention-guided neural network (termed MSANN). In particular, we first introduce the logarithmic transformation to convert the multiplicative speckle noise model to an additive version. Our MSANN, essentially a feature pyramid network, is then exploited to restore degraded images in the logarithmic domain. To enhance the generalization ability of the MSANN, a multiscale feature enhancement attention module is incorporated into the MSANN to extract multiscale features for improving the imaging quality. A multiscale mixed loss function is further presented to increase the network robustness during training. The final despeckled images are naturally equivalent to the exponential versions of the output of the MSANN. Experimental results have shown that the MSANN has the capacity of effectively removing the speckle noise while preserving essential structures. It can achieve superior despeckling results in terms of visual quality and quantitative measures.*
---

## Network Architecture

![2](https://github.com/user-attachments/assets/691eb6eb-d3e8-4e59-9c81-029fb8e3f216)

## Test
* Put the noisy image in the `./input` folder
* Run `main.py`. 
* The enhancement result will be saved in the `./output` folder.

## Citation

```
@article{guo2023blind,
  title={Blind Image Despeckling Using a Multiscale Attention-Guided Neural Network},
  author={Guo, Yu and Lu, Yuxu and Liu, Ryan Wen and Zhu, Fenghua},
  journal={IEEE Transactions on Artificial Intelligence},
  volume={5},
  number={1},
  pages={205--216},
  year={2023}
}
```

#### If you have any questions, please get in touch with me (guoyu65896@gmail.com).
