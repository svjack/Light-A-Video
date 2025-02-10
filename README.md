<div align="center">
    <img src='__assets__/title.png'/>
</div>
⭐️ **Our team's works:** [[**MotionClone**](https://bujiazi.github.io/motionclone.github.io/)]  [[**BroadWay**](https://bujiazi.github.io/BroadWay.github.io/)]  

# Light-A-Video: Training-free Video Relighting via Progressive Light Fusion
This repository is the official implementation of Light-A-Video. It is a **training-free framework** that enables 
zero-shot illumination control of any given video sequences or foreground sequences.

<details><summary>Click for the full abstract of Light-A-Video</summary>
> Recent advancements in image relighting models, driven by large-scale datasets and pre-trained diffusion models, 
have enabled the imposition of consistent lighting. 
However, video relighting still lags, primarily due to the excessive training costs and the scarcity of diverse, high-quality video relighting datasets.
A simple application of image relighting models on a frame-by-frame basis leads to several issues: 
lighting source inconsistency and relighted appearance inconsistency, resulting in flickers in the generated videos.
In this work, we propose Light-A-Video, a training-free approach to achieve temporally smooth video relighting.
Adapted from image relighting models, Light-A-Video introduces two key techniques to enhance lighting consistency.
First, we design a Consistent Light Attention (CLA) module, which enhances cross-frame interactions within the self-attention layers 
to stabilize the generation of the background lighting source. Second, leveraging the physical principle of light transport independence, 
we apply linear blending between the source video’s appearance and the relighted appearance, using a Progressive Light Fusion \textbf{(PLF)} strategy to ensure smooth temporal transitions in illumination. 
Experiments show that Light-A-Video improves the temporal consistency of relighted video
while maintaining the image quality,  ensuring coherent lighting transitions across frames.
</details>

**[Light-A-Video: Training-free Video Relighting via Progressive Light Fusion]()** 
</br>
[Yujie Zhou*](https://github.com/YujieOuO/),
[Jiazi Bu*](https://github.com/Bujiazi/),
[Pengyang Ling*](https://github.com/LPengYang/),
[Pan Zhang<sup>†</sup>](https://panzhang0212.github.io/),
[Tong Wu](https://wutong16.github.io/),
[Qidong Huang](https://shikiw.github.io/)
[Jinsong Li](https://li-jinsong.github.io/)
[Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en/),
[Yuhang Zang](https://yuhangzang.github.io/),
[Yuhang Cao](https://scholar.google.com/citations?hl=zh-CN&user=sJkqsqkAAAAJ)
[Anyi Rao](https://anyirao.com/)
[Jiaqi Wang](https://myownskyw7.github.io/),
[Li Niu<sup>†</sup>](https://www.ustcnewly.com/)  
(*Equal Contribution)(<sup>†</sup>Corresponding Author)

[![arXiv](https://img.shields.io/badge/arXiv-2406.05338-b31b1b.svg)](https://arxiv.org/abs/2406.05338)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://bujiazi.github.io/light-a-video.github.io/)

## 📜 News

**[2025/2/11]** Code is available now!

**[2025/2/10]** The [paper]([ShareGPT4V.pdf](https://arxiv.org/pdf/2311.12793.pdf)) and
 [project page](https://bujiazi.github.io/light-a-video.github.io/) are released!

## 🏗️ Todo
[ ] Release Light-A-Video code with CogVideo-2B pipeline.

## 🚀 Method Overview

<div align="center">
    <img src='__assets__/pipeline.png'/>
</div>

Light-A-Video leverages the the capabilities of image relighting models and VDM motion priors to achieve temporally consistent video relighting. 

## 🔧 Installations (python==3.10 recommended)

### Setup repository and conda environment

```
git clone https://github.com/bcmi/Light-A-Video.git
cd Light-A-Video

conda create -n lav python=3.10
conda activate lav

pip install -r requirements.txt
```

## 🔑 Pretrained Model Preparations

### Download IC-Light Model

### Prepare Community Models

### Prepare AnimateDiff Motion Modules

## 🎈 Quick Start

### Perform video relighting with customized illumination control
```
python lav_relight.py --config "configs/relight/car.yaml"
```
### Perform foreground sequences relighting with background generation
```
python lav_paint.py --config "configs/relight_inpaint/car.yaml"
```

## 📎 Citation 

If you find this work helpful, please cite the following paper:

```
@article{ling2024motionclone,
  title={MotionClone: Training-Free Motion Cloning for Controllable Video Generation},
  author={Ling, Pengyang and Bu, Jiazi and Zhang, Pan and Dong, Xiaoyi and Zang, Yuhang and Wu, Tong and Chen, Huaian and Wang, Jiaqi and Jin, Yi},
  journal={arXiv preprint arXiv:2406.05338},
  year={2024}
}
```

## 📣 Disclaimer

This is official code of Light-A-Video.
All the copyrights of the demo images and audio are from community users. 
Feel free to contact us if you would like remove them.

## 💞 Acknowledgements
The code is built upon the below repositories, we thank all the contributors for open-sourcing.
* [IC-Light](https://github.com/lllyasviel/IC-Light)
* [AnimateDiff](https://github.com/guoyww/AnimateDiff)