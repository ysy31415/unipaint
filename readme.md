# Uni-paint: A Unified Framework for Multimodal Image Inpainting with Pretrained Diffusion Model

#### Shiyuan Yang<sup>1</sup>, Xiaodong Chen<sup>2</sup>, Jing Liao<sup>1</sup>*.

<sup>1</sup> City University of Hong Kong, Hong Kong SAR.
<sup>2</sup> Tianjin University, China.

[[Paper Link]](https://dl.acm.org/doi/10.1145/3581783.3612200) . Supplementary materials can be found in [Arxiv](https://arxiv.org/abs/2310.07222) version.




## Abstract
Recently, text-to-image denoising diffusion probabilistic models (DDPMs) have demonstrated impressive image generation capabilities and have also been successfully applied to image inpainting. However, in practice, users often require more control over the inpainting process beyond textual guidance, especially when they want to composite objects with customized appearance, color, shape, and layout. Unfortunately, existing diffusion-based inpainting methods are limited to single-modal guidance and require task-specific training, hindering their cross-modal scalability. To address these limitations, we propose Uni-paint, a unified framework for multimodal inpainting that offers various modes of guidance, including unconditional, text-driven, stroke-driven, exemplar-driven inpainting, as well as a combination of these modes.
Furthermore, our Uni-paint is based on pretrained Stable Diffusion and does not require task-specific training on specific datasets, enabling few-shot generalizability to customized images.
We have conducted extensive qualitative and quantitative evaluations that show our approach achieves comparable results to existing single-modal methods while offering multimodal inpainting capabilities not available in other methods.




## Setup
### Conda enviromnet
```
conda env create -f environment.yaml
conda activate ldm
```

### Model
Download pretrained Stable Diffusion v1.4 from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and place it at `ckpt/sd-v1-4-full-ema.ckpt`.

 Please refer to [official SD repo](https://github.com/CompVis/stable-diffusion#requirements) for more details.

### CLIP text embedding

Download pre-computed CLIP text embedding (see paper Eq.6 for explanation) from [onedrive](https://portland-my.sharepoint.com/:u:/g/personal/shiyyang8-c_my_cityu_edu_hk/EbfbmS30q2hEqpAr336W2_IBfublGOQXHf-NHG4lhX5_Ew?e=QbEmBN) and place it at `ckpt/clip_emb_normalized(49407x768).pth`. Or you can skip downloading now, the code also will generate this file if it's not found, this process may take several mintues.


## Usage

### Python notebook

* For unconditional/text-driven/stroke-driven inpainting, see `inpaint.ipynb`.
* For exemplar-driven inpainting, see `inpaint_with_exemplar.ipynb`.


### Gradio demo
We also made an interactive gradio demo for convenient use. Here are the step-by-step guidelines:

1. Launch the demo script `gradio_demo/demo.py`.

2. By default, go to http://127.0.0.1:7860/ in your browser, the demo should be displayed there. If you are runing the model on a server, you may forward the demo to your local pc browser by using the command `ssh username@xxx.xxx.xxx.xxx -p 22 -L 7860:localhost:7860`.
    
3. Input image: at the left-top section, provide the input image and draw the mask area.

4. [Optional] Exemplar image: In second column, provide an exemplar image and check the box `Enable exemplar`.
    
5. Initialize: Click  `Initialize` button (this will setup the model and prepare your inputs).

6. Finetune: Click `Finetune` button to launch the finetuning on your inputs. Please wait until finetuning is finished (which takes ~1 minute, you will see button changes from `Finetuning...` back to `Finetune` when it's done).

7. Inference:

    * Unconditional inpainting: make sure to uncheck all the boxes in the top row (i.e., `Enable text`, `Enable exemplar`, `Enable stroke` ), then click `Inference` button.
    * Text inpainting: In Text condition section (3rd column), first check the box `Enable text`, and input your text prompt, then click `Inference` button.
    * Exemplar inpainting: In Exemplar condition section (2nd column), check the box `Enable exemplar`, then click `Inference` button.
    * Stroke inpainting: In Stroke condition section (last column), first check the box `Enable stroke`, then you will see the masked input being displayed below, use the color brush tool to draw the color stroke within the black masked area. Or you can upload your own stroke image (the background needs to be black). Finally click `Inference` button. 
    * Mixed inpainting: for example, to perform text + stroke inpainting, check both `Enable text` and `Enable stroke` boxe, uncheck `Enable exemplar` box, input your text prompt and draw the color stroke, then click `Inference` button.

        Note: you can adjust the stroke blending timestep slide bar to adjust the realism-faithfulness trade-off (larger value leads to more realistic but less aligned result).
  
  8. Outputs: The generated results will be shown at bottom row.

**Other notes:**
* If you change the input image and/or exemplar image, you need to redo the Initialization and Finetuning process (repeat step 3-7).
* If you want to change text and/or stroke image, you do NOT need to repeat Initialization and Finetuning (just re-do step 7).
* Sometimes the stroke image may not be fully or successfully displayed after clicking `Enable stroke` box, this might be caused by the unknown bug of the gradio, check and uncheck the `Enable stroke` box several times can solve this issue.


## Citation
```
@inproceedings{unipaint,
author = {Yang, Shiyuan and Chen, Xiaodong and Liao, Jing},
title = {Uni-Paint: A Unified Framework for Multimodal Image Inpainting with Pretrained Diffusion Model},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3581783.3612200},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {3190–3199},
location = {Ottawa ON, Canada},
series = {MM '23}
}
```

## Acknowledgment
The code is built based on [LDM](https://github.com/CompVis/stable-diffusion) and [Textual Inversion](https://github.com/rinongal/textual_inversion).

