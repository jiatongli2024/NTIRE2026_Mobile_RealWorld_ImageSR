# [NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

[![ntire](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fzhengchen1999%2FNTIRE2025_ImageSR_x4%2Fmain%2Ffigs%2Fdiamond_badge.json)](https://www.cvlai.net/ntire/2026/)
[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://gobunu.github.io/ntire_mobile_sr)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=jiatongli2024.NTIRE2026_Mobile_RealWorld_ImageSR&right_color=violet)](https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR)
[![GitHub Stars](https://img.shields.io/github/stars/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR?style=social)](https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR)

## Notice

All submitted code must follow the format defined in this repository. Submissions that do not follow the required format may be rejected during the final evaluation stage.

After the challenge ends, we will release all submitted code as open-source for reproducibility. If you would like your model to remain confidential, please contact the organizers in advance.

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

## How to test the baseline model?

1. `git clone https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR.git`

2. Select the model you would like to test:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
   ```

   - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
   - We provide a baseline (team00): DAT (default). Switch models (default is DAT) through commenting the code in [test.py](./test.py#L19).

## How to add your model to this baseline?

> [!IMPORTANT]
>
> **🚨 Submissions that do not follow the official format will be rejected.**

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/10eMuQwZ36DhdC-RCXJvWJHRAs2Lk0dMXLwm8JzFh758/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in folder:  `./models/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
3. Put the pretrained model in folder: `./model_zoo/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
   - Note: Please provide a download link for the pretrained model, if the file size exceeds **100 MB**. Put the link in `./model_zoo/[Your_Team_ID]_[Your_Model_Name]/[Your_Team_ID]_[Your_Model_Name].txt`: e.g. [team00_dat.txt](./model_zoo/team00_dat/team00_dat.txt)
4. Add your model to the model loader `test.py` as follows:

   - Edit the `else` to `elif` in [test.py](./test.py#L24), and then you can add your own model with model id.

   - `model_func` **must** be a function, which accept **4 params**. 

     - `model_dir`: the pretrained model. Participants are expected to save their pretrained model in `./model_zoo/` with in a folder named `[Your_Team_ID]_[Your_Model_Name]` (e.g., team00_dat). 

     - `input_path`: a folder contains several images in PNG format. 

     - `output_path`: a folder contains restored images in PNG format. Please follow the section Folder Structure. 

     - `device`: computation device.
5. Send us the command to download your code, e.g,

   - `git clone [Your repository link]`
   - We will add your code and model checkpoint to the repository after the challenge.

> [!TIP]
>
> Your model code does not need to be fully refactored to fit this repository. 
> Instead, you may add a lightweight external interface (e.g., `models/team00_DAT/io.py`) that wraps your existing code, while keeping the original implementation unchanged.
>
> Refer to previous NTIRE challenge implementations for examples: 
> https://github.com/zhengchen1999/NTIRE2025_ImageSR_x4/tree/main/models

## How to eval images using IQA metrics?

### Environments

```sh
conda create -n NTIRE-SR python=3.8
conda activate NTIRE-SR
pip install -r requirements.txt
```


### Folder Structure

```
test_dir
├── HR
│   ├── 0901.png
│   ├── 0902.png
│   ├── ...
├── LQ
│   ├── 0901x4.png
│   ├── 0902x4.png
│   ├── ...
    
output_dir
├── 0901x4.png
├── 0902x4.png
├──...

```

### Command to calculate metrics

```sh
python eval.py \
--output_folder "/path/to/your/output_dir" \
--target_folder "/path/to/test_dir/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```

The `eval.py` file accepts the following 4 parameters:

- `output_folder`: Path where the restored images are saved.
- `target_folder`: Path to the HR images in the `test` dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

## Reference Code
We provide a [reference implementation for checkpoint saving](./uitls/ref_ckpt_save.py), which we will use to reproduce participants’ experimental results.
Participants may use our implementation as-is or modify it based on our reference code.

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
