<div align="center">
    <img src='__assets__/title.png'/>
</div>

# Light-A-Video
This repository is the official implementation of Light-A-Video. It is a **training-free framework** that enables 
zero-shot illumination control of any given video sequences or foreground sequences.
<details><summary>Click for the full abstract of Light-A-Video</summary>
> Recent advancements in image relighting models, driven by large-scale datasets and pre-trained diffusion models, 
have enabled the imposition of consistent lighting. 
However, video relighting still lags, primarily due to the excessive training costs and the scarcity of diverse, high-quality video relighting datasets.
A simple application of image relighting models on a frame-by-frame basis leads to several issues: 
lighting source inconsistency and relighted appearance inconsistency, resulting in flickers in the generated videos.
In this work, we propose \textbf{Light-A-Video}, a training-free approach to achieve temporally smooth video relighting.
Adapted from image relighting models, Light-A-Video introduces two key techniques to enhance lighting consistency.
First, we design a Consistent Light Attention \textbf{(CLA)} module, which enhances cross-frame interactions within the self-attention layers 
to stabilize the generation of the background lighting source. Second, leveraging the physical principle of light transport independence, 
we apply linear blending between the source video’s appearance and the relighted appearance, using a Progressive Light Fusion \textbf{(PLF)} strategy to ensure smooth temporal transitions in illumination. 
Experiments show that Light-A-Video improves the temporal consistency of relighted video
while maintaining the image quality,  ensuring coherent lighting transitions across frames.
</details>

