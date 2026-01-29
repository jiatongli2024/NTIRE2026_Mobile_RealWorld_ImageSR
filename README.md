# [NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

[![ntire](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fzhengchen1999%2FNTIRE2025_ImageSR_x4%2Fmain%2Ffigs%2Fdiamond_badge.json)](https://www.cvlai.net/ntire/2026/)


## About the Challenge

The challenge is part of the 11th NTIRE Workshop at CVPR 2026, which targets the real-world image super-resolution on mobile devices. Participants should recover a high‑resolution image from a single low‑resolution input that is 4 × smaller and with unknown degradations.

The evaluation consists of comparing the restored high-resolution images with the ground truth high-resolution images. To comprehensively assess the results, we employ evaluation metrics as follows:  

- **Inference Speed:** We will benchmark the inference speed on the **MediaTek Dimensity 8400** platform, using the inference speed of **OSEDiff** on this platform as the baseline. The input image size is $128\times 128$, and the ouput size is $512\times 512$. We define $t_{osediff}$ and $t_{curmodel}$ as the average inference time on single image using OSEDiff and current model, and the definition of speedup ratio is:

$$
Speedup=\frac{t_{osediff}}{t_{curmodel}}
$$


- **Perceptual Metrics:** **LPIPS**, **DISTS**, **NIQE**, **ManIQA**, **MUSIQ**, and **CLIP-IQA**. To measure the super-resolution performance, we calculate the average weighted value of the six perceptual metrics. The input image size is arbitrary. The Score is defined as follows:

$$
\text{Score} = \left(1 - \text{LPIPS}\right) + \left(1 - \text{DISTS}\right) + \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right)
$$

The final score of each participant is defined as follows:

$$
FinalScore=2^{Score}\cdot {Speedup}^{0.2}
$$

## Develop Environments
We provide a reference **pip** installation list in [requirements.txt](./requirements.txt), and participants should check whether the adopted environment meets the requirements.

In addition, we provide a list of operators supported on the **MediaTek Dimensity 8400** platform. When designing models, participants should avoid using unsupported operators; otherwise, the code may fail to run, resulting in an unqualified submission.

As a reference, both the [Stable-Diffusion-2.1-base](https://huggingface.co/Manojb/stable-diffusion-2-1-base) and the [Stable-Diffusion-3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) can run successfully on this platform.

![alt text](figs/opts.PNG)

## Reference Code
We provide a [reference implementation for checkpoint saving](./uitls/ref_ckpt_save.py), which we will use to reproduce participants’ experimental results.
Participants may use our implementation as-is or modify it based on our reference code.

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
