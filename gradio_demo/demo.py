import gradio as gr
from omegaconf import OmegaConf
import torch, torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pytorch_lightning import seed_everything
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os, sys
sys.path.append(os.getcwd())
sys.path.append('src/clip')
sys.path.append('src/taming-transformers')
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, load_model_from_config, ExemplarAugmentor
import kornia
from skimage import io

################### setup #####################################
n_samples = 4
os.environ['CUDA_VISIBLE_DEVICES']='0'

it = 100  # number of fine-tune iters
lr = 1e-5  # learning rate
h = w = 512
ddim_steps = 50
ddim_eta = 0.0
device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
ckpt = torch.load("ckpt/sd-v1-4-full-ema.ckpt", map_location='cpu')

img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.unsqueeze(0) * 2. - 1)])
mask_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: (x.unsqueeze(0) > 0.5).float())])

# Pre-define global var/func with dummy values for now, so that they can be accessed across the functions ###
# These dummy values will be overwritten later

x = torch.rand(1, 3, 512, 512)  # input gt image
m = torch.rand(1, 3, 512, 512)  # input mask
z_m = torch.rand(1, 4, 64, 64) # latent mask
z_xm = torch.rand(1, 4, 64, 64)  # masked latent
attn_mask = {}  # attention mask
x_in = torch.rand(1, 3, 512, 512)  # input masked image
x_ref = torch.rand(1, 3, 512, 512)  # exemplar image
z_ref = torch.rand(1, 4, 64, 64)  # exemplar latent
uc = torch.rand(1, 77, 768) # null-text emb

model = torch.nn.Module()
sampler = None
optimizer = None

C = lambda x, y: None  # text encode function
E = lambda x: None  # vae encode
D = lambda x: None  # vae decode
exemplar_augmentor = lambda x: None

def ts2np(x):
    return (255 * (x[0] + 1) / 2).cpu().permute(1, 2, 0).numpy().astype(np.uint8)

def tsave(tensor, save_path, **kwargs):
    save_image(tensor, save_path, normalize=True, scale_each=True, value_range=(-1, 1), **kwargs)
#####


# prepare model and input,mask or []
def init(input_image, stroke_image, exemplar_image, txt_textbox,
         exemplar_checkbox, stroke_checkbox, txt_checkbox,
         auto_sub_checkbox, sub_word_textbox,
         iteration, tau, ddim_steps, n_samples, cfg, seed):
    global x, m, z_xm, x_in, x_ref, z_ref, c_ref, z_m, attn_mask, uc
    global model, sampler, D, E, C, exemplar_augmentor, optimizer
    global it, lr

    # lr = lr
    it = iteration

    input_masked_np = input_image["image"] * (1 - (input_image["mask"][:, :, :3] > 128).astype(np.uint8))
    io.imsave("gradio_demo/__masked__input__.png", input_masked_np)

    seed_everything(seed)

    # x = img_transforms(input_image["image"]).repeat(n_samples, 1, 1, 1).to(device)
    x = img_transforms(input_image["image"]).to(device)
    # m = mask_transforms(input_image["mask"][:, :, :1]).repeat(n_samples, 1, 1, 1).to(device)
    m = mask_transforms(input_image["mask"][:, :, :1]).to(device)
    x_in = x * (1 - m)

    print(f"x shape:{x.shape}")
    print(f"m shape:{m.shape}")
    print(f"x_in shape:{x_in.shape}")

    for attn_size in [64, 32, 16, 8]:  # create attention mask dict for multi-scale layers in unet
        attn_mask[str(attn_size ** 2)] = (F.interpolate(m, (attn_size, attn_size), mode='bilinear'))[0, 0, ...]

    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")

    if exemplar_checkbox:
        # x_ref = img_transforms(exemplar_image).repeat(n_samples, 1, 1, 1).to(device)
        x_ref = img_transforms(exemplar_image).to(device)
        # overwrite config file
        config.model.params.personalization_config.params.initializer_words = ['__clip__'] if auto_sub_checkbox else [
            sub_word_textbox]
        config.model.params.personalization_config.params.placeholder_strings = ['#']
        # either UGLY CODE!!!
        io.imsave("gradio_demo/__exemplar_image__.png", exemplar_image)
        config.model.params.personalization_config.params.initializer_images = ["gradio_demo/__exemplar_image__.png"]
        # or
        # model.embedding_manager.initializer_images=[Image.fromarray(exemplar_image)]
        exemplar_augmentor = ExemplarAugmentor(mask=input_image["mask"])

    model = instantiate_from_config(config.model)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    sampler = DDIMSampler(model)
    params_to_be_optimized = list(model.model.parameters())
    optimizer = torch.optim.Adam(params_to_be_optimized, lr=lr)

    D = lambda _x: torch.clamp(model.decode_first_stage(_x), min=-1, max=1).detach()
    E = lambda _x: model.get_first_stage_encoding(model.encode_first_stage(_x))

    def C_(txt, emb_mger=False):
        with torch.no_grad():
            # return model.get_learned_conditioning(n_samples * [txt], emb_mger)
            return model.get_learned_conditioning([txt], emb_mger)

    C = C_

    z_xm = E(x_in)
    z_m = F.interpolate(m, size=(h // 8, w // 8))
    z_m = kornia.morphology.dilation(z_m, torch.ones((3, 3), device=device))  # dilate mask a little bit

    uc = C("", False).to(device)
    c_ref = C('#', True) if exemplar_checkbox else None
    z_ref = E(x_ref) if exemplar_checkbox else None

    print("done!")
    # return [ts2np(tmp) for tmp in [x, x_in, x_ref]]
    return True


# Step 3.2 - fine-tune model ####################################################################################
def finetune(input_image, stroke_image, exemplar_image, txt_textbox,
             exemplar_checkbox, stroke_checkbox, txt_checkbox,
             auto_sub_checkbox, sub_word_textbox,
             iteration, tau, ddim_steps, n_samples, cfg, seed):
    # flag_finetune = True
    model.train()
    pbar = tqdm(range(iteration), desc='Fine-tune the model')
    for i in pbar:
        optimizer.zero_grad()

        if exemplar_checkbox:
            t_emb = torch.randint(model.num_timesteps, (1,), device=device)
            x_reff, x_reff_mask = exemplar_augmentor(x_ref)
            z_reff = E(x_reff)
            z_reff_mask = F.interpolate(x_reff_mask, size=(64, 64), mode='bilinear')
            noise1 = torch.randn_like(z_xm)
            z_ref_t = model.q_sample(z_reff, t_emb, noise=noise1)
            pred_noise_ref = model.apply_model(z_ref_t, t_emb, c_ref)
            loss_ref = F.mse_loss(pred_noise_ref * z_reff_mask, noise1 * z_reff_mask)  # noise loss
            # loss_ref = F.mse_loss(pred_z0_ref, z_ref)        # z0 loss, this is better

        t_emb2 = torch.randint(model.num_timesteps, (1,), device=device)
        noise2 = torch.randn_like(z_xm)
        z_bg_t = model.q_sample(z_xm, t_emb2, noise=noise2)
        pred_noise_bg = model.apply_model(z_bg_t, t_emb2, uc)
        loss_bg = F.mse_loss(pred_noise_bg * (1 - z_m), noise2 * (1 - z_m))

        if exemplar_checkbox:
            loss = loss_bg + loss_ref
        else:
            loss = loss_bg
        loss.backward()
        optimizer.step()

    return True


@torch.no_grad()
def infernece(input_image, stroke_image, exemplar_image, txt_textbox,
              exemplar_checkbox, stroke_checkbox, txt_checkbox,
              auto_sub_checkbox, sub_word_textbox,
              iteration, tau, ddim_steps, n_samples, cfg, seed, bg_blend
              ):
    model.eval()
    # seed_everything(seed)

    if exemplar_checkbox == True and txt_checkbox == True:
        c = C(txt_textbox, True)
        s = cfg
    elif exemplar_checkbox == True and txt_checkbox == False:
        c = C('#', True)
        s = cfg
    elif exemplar_checkbox == False and txt_checkbox == True:
        c = C(txt_textbox, False)
        s = cfg
    else:
        c = uc
        s = 1
    # print(f"txt_textbox: {txt_textbox}")
    # print(f"s = {s}")
    # print(f"bg_blend={bg_blend}")

    with torch.autocast(device.type):
        if not stroke_checkbox:
            tmp, _ = sampler.sample(S=ddim_steps, conditioning=c.repeat(n_samples,1,1), batch_size=n_samples, shape=[4, h // 8, w // 8],
                                    verbose=False, unconditional_guidance_scale=s, unconditional_conditioning=uc.repeat(n_samples,1,1), eta=0,
                                    blend_interval=[0, bg_blend], x0=z_xm.repeat(n_samples,1,1,1), mask=z_m.repeat(n_samples,1,1,1), attn_mask=attn_mask)
        else:
            m_big = (input_image["mask"][:, :, :3] > 0).astype(np.uint8)  # for inpainting
            m_small = (input_image["mask"][:, :, :3] > 254).astype(np.uint8)  # for clearing stroke bg
            x_stroke_np = (stroke_image - input_image["image"] * (
                    1 - m_big)) * m_small if stroke_image is not None else np.zeros_like(m_big)

            # x_stroke = img_transforms(x_stroke_np).repeat(n_samples, 1, 1, 1).to(device)
            x_stroke = img_transforms(x_stroke_np).to(device)

            x_stroke_mask = (torch.mean(x_stroke, dim=1, keepdim=True) > -1).float()
            z_stroke = E(x_stroke)
            z_stroke_mask = F.interpolate(x_stroke_mask, size=(h // 8, w // 8))

            tmp, _ = sampler.sample(S=ddim_steps, conditioning={'t': [[0, tau], [tau, 1]], 'c': [c.repeat(n_samples,1,1), uc.repeat(n_samples,1,1)]},
                                    batch_size=n_samples, shape=[4, h // 8, w // 8],
                                    verbose=False, unconditional_guidance_scale=s, unconditional_conditioning=uc.repeat(n_samples,1,1), eta=0,
                                    blend_interval=[[0, bg_blend], [tau, tau + 0.02]], x0=[z_xm.repeat(n_samples,1,1,1), z_stroke.repeat(n_samples,1,1,1)],
                                    mask=[z_m.repeat(n_samples,1,1,1), 1 - z_stroke_mask.repeat(n_samples,1,1,1)],
                                    attn_mask=attn_mask)

    out = 255 * (D(tmp) + 1) / 2
    out_np = [item.cpu().permute(1, 2, 0).numpy().astype(np.uint8) for item in out]
    return out_np, True


def dummpy_run(input_image, stroke_image, exemplar_image, txt_textbox, tau, ddim_steps, n_samples, cfg, seed):
    # [input_image, stroke_image, exemplar_image, txt_textbox, tau, ddim_steps, n_samples, cfg, seed]
    x = input_image["image"]
    m_small = (input_image["mask"][:, :, :3] > 254).astype(np.uint8)  # for clearing stroke bg
    m_big = (input_image["mask"][:, :, :3] > 0).astype(np.uint8)  # for inpainting
    m = m_big
    x_m = x * (1 - m_big)
    x_stroke = (stroke_image - x_m) * m_small if stroke_image is not None else np.zeros_like(x)
    x_ref = exemplar_image if exemplar_image is not None else np.zeros_like(x)
    return [x, 255 * m, x_m, x_ref, x_stroke]


# def sync_input_to_stroke(input_image):
#     image_np = input_image["image"]
#     mask_np = (input_image["mask"][:,:,:3] > 128).astype(np.uint8)
#     masked_np = image_np * (1-mask_np)
#     # io.imsave("gradio_demo/examples/_.png", masked_np)
#     # masked_np = io.imread("gradio_demo/examples/_.png")
#     return masked_np

##########################################################


with gr.Blocks() as demo:
    # gr.Markdown("# Uni-paint Interactive Demo")
    # INPUTS
    with gr.Column():
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input image (required)")
                gr.Markdown("Upload or select from examples, then draw the mask area.")
                input_image = gr.Image(label="Input", source='upload', tool='sketch', type="numpy", shape=(512, 512))
                input_examples = gr.Examples(label="Input examples",
                                             examples=["gradio_demo/examples/" + tmp for tmp in
                                                       ["sofa.png", "street.png", "table.png", "bench.png", "beach.png",
                                                        "mount.png", "dish.png", "mellow.png", "table2.png"]],
                                             inputs=input_image, examples_per_page=8)
                gr.Markdown("Note: you cannot change input and exemplar image once fine-tuning is launched")

            with gr.Column():
                gr.Markdown("## Exemplar condition")
                exemplar_checkbox = gr.Checkbox(label="Enable exemplar", value=False, interactive=True)
                gr.Markdown("Upload or select from examples")
                exemplar_image = gr.Image(label="Exemplar", tool='select', type="numpy", shape=(512, 512),
                                          interactive=True)
                exemplar_examples = gr.Examples(label="Exemplar examples",
                                                examples=["gradio_demo/examples/" + tmp for tmp in
                                                          ["cat.png", "frog.png", "duck.png", "wn.png", "do.png",
                                                           "liberty.png"]],
                                                inputs=exemplar_image, examples_per_page=4)

                gr.Markdown("Subject word")
                with gr.Row():
                    auto_sub_checkbox = gr.Checkbox(label="Auto select", value=True,
                                                    interactive=True)  # when this is off, show sub_word_textbox, see https://gradio.app/docs/#update
                    sub_word_textbox = gr.Textbox(label="", max_lines=1, placeholder="Specify a single word",
                                                  interactive=True, visible=False)

            with gr.Column():
                gr.Markdown("## Text condition")
                txt_checkbox = gr.Checkbox(label="Enable text", value=False, interactive=True)
                gr.Markdown("Type text or select from examples")
                txt_textbox = gr.Textbox(label="", placeholder="Enter your text prompt here")
                gr.Markdown("When Exemplar is enabled, use # as the pseudo word.")
                text_examples = gr.Examples(label="Text examples", examples=["A lovely cat wearing a red hat", "Apples",
                                                                             "A puppy dog wearing sunglasses.",
                                                                             "A toy bear.", ], inputs=txt_textbox)
                cfg = gr.Slider(label="Guidance Scale", minimum=1, maximum=32.0, value=8, step=1)

            with gr.Column():
                # with gr.Box():
                gr.Markdown("## Stroke condition")
                gr.Markdown("-------------", visible=False)
                stroke_checkbox = gr.Checkbox(label="Enable stroke", value=False, interactive=True)
                gr.Markdown("When enabled, draw stroke in the masked area below.")
                stroke_image = gr.Image(label="Stroke", source='upload', type='numpy', interactive=True,
                                        tool='color-sketch', shape=(512, 512))
                gr.Markdown("Adjust to control realism-faithfulness trade off", visible=True)
                tau = gr.Slider(label="Blending timestep", minimum=0.0, maximum=1.0, value=0.55, step=0.01,
                                interactive=True)

        with gr.Row():
            # dummy_run_button = gr.Button(value="dummpy_run")
            init_button = gr.Button(value="Initialize")
            with gr.Column():
                finetune_button = gr.Button(value="Finetune",
                                            interactive=False)  # "Inputs Ready! Click to launch fine-tuning.")
            with gr.Column():
                inference_button = gr.Button(value="Inference", interactive=False)

            with gr.Accordion("Advanced options", open=False):
                n_samples = gr.Slider(
                    label="Num of images", minimum=1, maximum=8, value=4, step=1)
                ddim_steps = gr.Slider(label="Sampling steps", minimum=10,
                                       maximum=100, value=50, step=10)
                seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=1024, step=1, randomize=False)
                iteration = gr.Slider(label="Finetuning iter", minimum=0, maximum=150, step=25, value=100)
                bg_blend = gr.Slider(label="Bg blend", minimum=-0.1, maximum=1, step=0.1, value=1)

        # OUTPUTS
        with gr.Row():
            # with gr.Column():
            #     gr.Markdown("## Inputs")
            #     txt_textbox2 = gr.Textbox(label="Input text", interactive=False)
            #     gallery_in = gr.Gallery(label="Input, mask, masked input, exemplar, stroke", type='numpy', show_label=True).style(columns=5, height="auto")
            with gr.Column():
                gr.Markdown("## Outputs")
                gallery_out = gr.Gallery(label="Outputs", type='numpy', show_label=True).style(columns=8, height="auto")

    ######### Hidden states  ##############
    flag_init = gr.Checkbox(value=False, visible=False, interactive=False)
    flag_finetune = gr.Checkbox(value=False, visible=False, interactive=False)
    flag_inference = gr.Checkbox(value=False, visible=False, interactive=False)

    ######### Event listeners  ##############
    stroke_checkbox.change(fn=lambda x: gr.update(value=x["image"]*(1-(x["mask"][:,:,:3] > 128).astype(np.uint8))), inputs=input_image, outputs=stroke_image)

    auto_sub_checkbox.change(fn=lambda flag: gr.update(visible=not flag), inputs=auto_sub_checkbox,
                             outputs=sub_word_textbox)  # https://gradio.app/docs/#update
    # stroke_checkbox.change(fn=lambda flag: gr.update(interactive=flag), inputs=stroke_checkbox,outputs=stroke_image)
    # stroke_checkbox.change(fn=lambda flag: gr.update(interactive=flag), inputs=stroke_checkbox,outputs=tau)
    # exemplar_checkbox.change(fn=lambda flag: gr.update(interactive=flag), inputs=exemplar_checkbox,outputs=exemplar_image)
    # exemplar_checkbox.change(fn=lambda flag: gr.update(interactive=flag), inputs=exemplar_checkbox,outputs=[auto_sub_checkbox])

    # dummy_run_button.click(fn=dummpy_run,
    #                  inputs=[input_image, stroke_image, exemplar_image, txt_textbox, tau, ddim_steps, n_samples, cfg, seed],
    #                  outputs=[gallery_out])

    inference_button.click(fn=infernece,
                           inputs=[input_image, stroke_image, exemplar_image, txt_textbox,
                                   exemplar_checkbox, stroke_checkbox, txt_checkbox,
                                   auto_sub_checkbox, sub_word_textbox,
                                   iteration, tau, ddim_steps, n_samples, cfg, seed, bg_blend,
                                   ],
                           outputs=[gallery_out, flag_inference]
                           )

    finetune_button.click(fn=finetune,
                          inputs=[input_image, stroke_image, exemplar_image, txt_textbox,
                                  exemplar_checkbox, stroke_checkbox, txt_checkbox,
                                  auto_sub_checkbox, sub_word_textbox,
                                  iteration, tau, ddim_steps, n_samples, cfg, seed,
                                  ],
                          outputs=flag_finetune
                          )
    init_button.click(fn=init,
                      inputs=[input_image, stroke_image, exemplar_image, txt_textbox,
                              exemplar_checkbox, stroke_checkbox, txt_checkbox,
                              auto_sub_checkbox, sub_word_textbox,
                              iteration, tau, ddim_steps, n_samples, cfg, seed,
                              ],
                      outputs=flag_init
                      )
    ############################### init_button logic ####################
    # when click init_button, set flag_init to False, disable finetune_button, init_button, inference_button for while
    init_button.click(fn=lambda: gr.update(value=False), outputs=flag_init)
    init_button.click(fn=lambda: gr.update(value="Initializing...", interactive=False), outputs=init_button)
    init_button.click(fn=lambda: gr.update(interactive=False), outputs=inference_button)
    init_button.click(fn=lambda: gr.update(interactive=False), outputs=finetune_button)
    # after init() is finished and returns True to flag_init, enable init_button, finetune_button and inference_button then,
    # i.e., finetune and inference are not allowed while Initializing
    flag_init.change(fn=lambda flag: gr.update(interactive=flag), inputs=flag_init, outputs=finetune_button)
    flag_init.change(fn=lambda flag: gr.update(interactive=flag), inputs=flag_init, outputs=inference_button)
    flag_init.change(fn=lambda flag: gr.update(value="Initialize" if flag else "Initializing...", interactive=flag),
                     inputs=flag_init, outputs=init_button)

    ############################### finetune_button logic ###################
    # When click finetune button, set flag_finetune to 0,  disable init_button, finetune_button, inference_button for while
    finetune_button.click(fn=lambda: gr.update(value=False), outputs=flag_finetune)
    finetune_button.click(fn=lambda: gr.update(value="Finetuning...", interactive=False), outputs=finetune_button)
    finetune_button.click(fn=lambda: gr.update(interactive=False), outputs=inference_button)
    finetune_button.click(fn=lambda: gr.update(interactive=False), outputs=init_button)
    # When finetune() is finished, it returns True to flag_finetune, enable finetune_button init_button and inference_button
    flag_finetune.change(fn=lambda flag: gr.update(interactive=flag), inputs=flag_finetune, outputs=inference_button)
    flag_finetune.change(fn=lambda flag: gr.update(interactive=flag), inputs=flag_finetune, outputs=init_button)
    flag_finetune.change(fn=lambda flag: gr.update(value="Finetune" if flag else "Finetuning...", interactive=flag),
                         inputs=flag_finetune, outputs=finetune_button)

    ############################### inference_button logic #################
    # When click inference_button, disable init_button, finetune_button, inference_button for while
    inference_button.click(fn=lambda: gr.update(value=False), outputs=flag_inference)
    inference_button.click(fn=lambda: gr.update(value="Inferring...", interactive=False), outputs=inference_button)
    inference_button.click(fn=lambda: gr.update(interactive=False), outputs=finetune_button)
    inference_button.click(fn=lambda: gr.update(interactive=False), outputs=init_button)
    # When inference is ended, it returns True to flag_inference, enable finetune_button init_button and inference_button
    flag_inference.change(fn=lambda flag: gr.update(interactive=flag), inputs=flag_inference, outputs=finetune_button)
    flag_inference.change(fn=lambda flag: gr.update(interactive=flag), inputs=flag_inference, outputs=init_button)
    flag_inference.change(fn=lambda flag: gr.update(value="Inference" if flag else "Inferring...", interactive=flag),
                          inputs=flag_inference, outputs=inference_button)

demo.launch()
