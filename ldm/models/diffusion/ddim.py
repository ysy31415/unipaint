"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))  #debug
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                    1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               blend_interval=None,  #: None to disable, set (0,1) for always blend, (0.2,0.8) for blending during 80%->20%
               skip_interval=None, #: None to disable. [0.8,1] means skip sampling during 0.8T~T
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):

        if (not isinstance(conditioning,dict)) and (not isinstance(unconditional_conditioning,dict)):
            batch_size, conditioning, x0, x_T, unconditional_conditioning, mask = \
            align_batch_size_dim_DEBUG_USE_ONLY(batch_size, conditioning, x0, x_T, unconditional_conditioning, mask)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0, blend_interval=blend_interval,
                                                    skip_interval=skip_interval,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False, skip_interval=None,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, blend_interval=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        intermediates = {'x_inter': [img], 'pred_x0': [img], 'index': [total_steps]}

        if blend_interval is not None:
            assert x0 is not None, "you must provide x0 to perform blending (for multi-stage blending, you should provide a x0 list)"
            assert mask is not None, "you must provide mask to perform blending (for multi-stage blending, you should provide a mask list)"
            # unsqueeze to 2D list if it's 1D list, e.g. [0,1]->[[0,1]]
            blend_interval_list = [blend_interval] if (not isinstance(blend_interval[0], list)) else blend_interval
            # if mask is already a list, so be it; otherwise (mask is a tensor), repeat it to list with the same length as blend_interval_list
            mask_list = mask if isinstance(mask, list) else [mask]*len(blend_interval_list)
            x0_list = x0 if isinstance(x0, list) else [x0]*len(blend_interval_list)
            assert len(blend_interval_list)==len(mask_list)==len(x0_list), "You must provide equal number of (blend_interval, x0, mask) when performing multi-stage blending!"  #
        '''
        Explanation: for example, if input args are: blend_interval_list = [[a,b],[c,d],[e,f]]; mask_list=[m1,m2,m3]; x0_list=x0
        first, x0_list will be expanded to [x0,x0,x0] to match the length, then, during [a,b], m1 and x0 will be used for blending,
        during [c,d], m2 and x0 will be used for blending.
        '''
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, miniters=10, mininterval=5)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1

            # skip some steps if skip_interval is specified
            if (skip_interval is not None) and (skip_interval[0] <= index/total_steps <= skip_interval[1]):
                continue

            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if blend_interval is not None:
                # blend_flags is list of booleans, indicates which blend should be enabled in current step.
                blend_flags = [sub_list[0] <= index / total_steps <= sub_list[1] for sub_list in blend_interval_list]
                for blend_idx, blend_flag in enumerate(blend_flags):
                    if blend_flag:
                        img_orig = self.model.q_sample(x0_list[blend_idx], ts)  # DDPM blending
                        img = img_orig * (1. - mask_list[blend_idx]) + img * mask_list[blend_idx]
                        # print(f"debug: step{index}, blend interval:{blend_interval_list[blend_idx]}")

            if isinstance(cond, dict): # multi-stage cond, e.g., {'t':[[0,0.4],[0.4,1]], 'c':[c1,c2]} means c1 is used during [0,0.4] and c2 is used during [0.4,1]
                assert len(cond['t'])==len(cond['c']), f"The length of time_range list and c list must be equal, but got {len(cond['t'])} and {len(cond['c'])}!"
                cond_flags = [sub_list[0] <= index / total_steps < sub_list[1] for sub_list in cond['t']]
                c = cond['c'][cond_flags.index(True)]
            else:  # cond is already a tensor, then use it for all timesteps (default impl.)
                c = cond

            if isinstance(unconditional_conditioning, dict): # multi-stage uncond, e.g. {'t':[[0,0.4],[0.4,1]], 'uc':[uc1,uc2]}
                assert len(unconditional_conditioning['t'])==len(unconditional_conditioning['uc']), f"The length of time_range list and uc list must be equal, but got {len(unconditional_conditioning['t'])} and {len(unconditional_conditioning['uc'])}!"
                uncond_flags = [sub_list[0] <= index / total_steps < sub_list[1] for sub_list in unconditional_conditioning['t']]
                uc = unconditional_conditioning['uc'][uncond_flags.index(True)]
            else:  # default impl.
                uc = unconditional_conditioning

            outs = self.p_sample_ddim(img, c, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=uc, x0=x0, mask=mask,**kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                intermediates['index'].append(index)

        print(f" blend_interval = {blend_interval if blend_interval is not None else 'OFF'} \n")
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, x0=None, mask=None,**kwargs):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):  #: hybrid case, c is a dict
                c_in = {}
                for k in c.keys():  # concat corresponding values if c and uc along batch dim
                    c_in[k] = [torch.cat(unconditional_conditioning[k] + c[k], dim=0)]
            else:
                c_in = torch.cat([unconditional_conditioning, c])

            attn_mask = kwargs.get('attn_mask')
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, attn_mask=attn_mask).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # classfier guidance here
        if kwargs.get('x0_guidance') is not None:
            # raise NotImplementedError
            assert None not in (x0, mask), 'you must provide x0 and mask to perform x0-guidance!'
            gui_scale = 0.04 if not isinstance(kwargs.get('x0_guidance'), float) else kwargs.get('x0_guidance')
            pred_x0 = pred_x0 + gui_scale *(x0 - pred_x0) * (1. - mask)  # guide pred_x0 towards x0 in unmasked region

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


def align_batch_size_dim_DEBUG_USE_ONLY(batch_size, conditioning, x0, x_T, unconditional_conditioning, mask):
    # align batch_size dim for inputs in case they have different batch size (ugly code)
    tensors = filter(lambda x: isinstance(x,torch.Tensor), [conditioning, x0, x_T, unconditional_conditioning, mask])
    max_bz = max([tensor.shape[0] for tensor in tensors]+[batch_size])
    batch_size = max_bz
    if isinstance(conditioning, torch.Tensor) and conditioning.shape[0] < batch_size:
        conditioning = conditioning.repeat(int(max_bz/conditioning.shape[0]),1,1)
        # print(f"conditioning dim0 expanded!")
    if isinstance(unconditional_conditioning, torch.Tensor) and unconditional_conditioning.shape[0] < batch_size:
        unconditional_conditioning = unconditional_conditioning.repeat(int(max_bz/unconditional_conditioning.shape[0]),1,1)
        # print(f"unconditional_conditioning dim0 expanded!")
    if isinstance(x0, torch.Tensor) and x0.shape[0] < batch_size:
        x0 = x0.repeat(int(max_bz/x0.shape[0]),1,1,1)
        # print(f"x0 dim0 expanded!")
    if isinstance(x_T, torch.Tensor) and x_T.shape[0] < batch_size:
        x_T = x_T.repeat(int(max_bz/x_T.shape[0]),1,1,1)
        # print(f"x_T dim0 expanded!")
    if isinstance(mask, torch.Tensor) and mask.shape[0] < batch_size:
        mask = mask.repeat(int(max_bz/mask.shape[0]),1,1,1)
        # print(f"mask dim0 expanded!")
    return batch_size, conditioning, x0, x_T, unconditional_conditioning, mask