import sys
import torch
import clip
import torch.nn.functional as F


from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from typing import Any, Dict, List, Optional, Union
# from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "src/utils")
from base_pipeline import BasePipeline
from cross_attention import prep_unet

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EditingPipeline(BasePipeline):

    def get_lr(self, i, total):
        if i + 1 < total // 10:
            return 2 * ((i + 1) / (total // 10))
        
        else:
            return 2 * (1/1.05) ** (i + 1)
        

    def decode_latents_for_grad(self, latents, use_trans=True):
        _trans = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),        
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5)
        if use_trans: return _trans(image)
        else: return image
    
    def compute_noise(self, 
                       do_classifier_free_guidance, 
                       latent_model_input,
                       t,
                       encoder_hidden_states,
                       cross_attention_kwargs,
                       classifier_free_guidance_scale
                       ):
        
        # predict the noise residual
        noise_pred = self.prep_unet(latent_model_input,t,encoder_hidden_states=encoder_hidden_states,cross_attention_kwargs=cross_attention_kwargs,).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + classifier_free_guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred

    def get_x_0_hat(self, 
                    x_t, 
                    alpha_t, 
                    noise_pred):
        x_0_hat = (x_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        return x_0_hat

    def get_x_T_hat(self, 
                    x_Tmt, 
                    alpha_T,
                    alpha_Tmt, 
                    noise_pred):
        x_T_hat = (alpha_T / alpha_Tmt).sqrt() * x_Tmt + ((1 - alpha_T).sqrt() - (alpha_T * (1 - alpha_Tmt) / alpha_Tmt).sqrt()) * noise_pred
        return x_T_hat

    def get_cycle_loss_tgt_1(self, x_tgt_0, x_tgt_T, prompt_embeds_edit, cross_attention_kwargs, guidance_scale, extra_step_kwargs):
        cycle_latents = x_tgt_T
        with torch.no_grad():
            for cycle_t in self.cycle_scheduler.timesteps:
                # expand the latents if we are doing classifier free guidance
                cycle_latent_model_input = torch.cat([cycle_latents] * 2) if self.do_classifier_free_guidance else cycle_latents
                cycle_latent_model_input = self.cycle_scheduler.scale_model_input(cycle_latent_model_input, cycle_t)

                cycle_noise_pred = self.unet(cycle_latent_model_input,cycle_t,encoder_hidden_states=prompt_embeds_edit,cross_attention_kwargs=cross_attention_kwargs,).sample

                # perform guidance
                if self.do_classifier_free_guidance:
                    cycle_noise_pred_uncond, cycle_noise_pred_text = cycle_noise_pred.chunk(2)
                    cycle_noise_pred = cycle_noise_pred_uncond + guidance_scale * (cycle_noise_pred_text - cycle_noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                cycle_latents = self.cycle_scheduler.step(cycle_noise_pred, cycle_t, cycle_latents, **extra_step_kwargs).prev_sample

            pseudo_x_tgt_0 = cycle_latents

        l2_loss = torch.norm(x_tgt_0 - pseudo_x_tgt_0)
        return l2_loss
    
    
    def get_cycle_loss_src(self, x_tgt_0, x_tgt_T, prompt_embeds, cross_attention_kwargs, guidance_scale, extra_step_kwargs):
        cycle_latents = x_tgt_0.detach().clone()
        
        with torch.no_grad():
            for cycle_t in self.cycle_inv_scheduler.timesteps.flip(0):
                # expand the latents if we are doing classifier free guidance
                cycle_latent_model_input = torch.cat([cycle_latents] * 2) if self.do_classifier_free_guidance else cycle_latents
                cycle_latent_model_input = self.cycle_inv_scheduler.scale_model_input(cycle_latent_model_input, cycle_t)

                cycle_noise_pred = self.unet(cycle_latent_model_input,cycle_t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                # perform guidance
                if self.do_classifier_free_guidance:
                    cycle_noise_pred_uncond, cycle_noise_pred_text = cycle_noise_pred.chunk(2)
                    cycle_noise_pred = cycle_noise_pred_uncond + guidance_scale * (cycle_noise_pred_text - cycle_noise_pred_uncond)

                break
            pseudo_x_tgt_T = self.get_x_T_hat(cycle_latents, self.alpha_T, 1.0, cycle_noise_pred)

        
        l2_loss = torch.norm(pseudo_x_tgt_T - x_tgt_T) 
        
        return l2_loss 


    def get_clip_loss(self, image_feature):
        clip_loss = torch.max(torch.tensor(0), self.cos_sim(image_feature, self.y_src.detach()) - self.cos_sim(image_feature, self.y_tgt.detach()) + self.clip_margin)
        return clip_loss[0]

    def get_structure_loss(self, Fxtgt_t, Fxtgt_naive, Fxsrc_t):
        st_loss = self.structure_margin * F.mse_loss(Fxtgt_t, Fxsrc_t)
        if Fxtgt_naive is not None:
            st_loss = torch.max(torch.tensor(0), self.structure_margin * F.mse_loss(Fxtgt_t, Fxsrc_t) - F.mse_loss(Fxtgt_t, Fxtgt_naive))
        return st_loss

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        target_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        x_in: Optional[torch.tensor] = None,
        cycle_guidance: Optional[float] = None,
        clip_guidance: Optional[float] = None,
        structure_guidance: Optional[float] = None,
        clip_margin: Optional[float] = None,
        structure_margin: Optional[float] = None,
    ):
        x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 0. modify the unet to be useful 
        self.prep_unet = prep_unet(self.unet)
        
        # 1. setup all caching objects
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self._execution_device)
        with torch.no_grad():
            if isinstance(prompt, str):
                self.y_src = clip_model.encode_text(clip.tokenize([prompt]).to(self._execution_device))
            else:
                self.y_src = clip_model.encode_text(clip.tokenize(prompt).to(self._execution_device))
            
            if isinstance(target_prompt, str):
                self.y_tgt = clip_model.encode_text(clip.tokenize([target_prompt]).to(self._execution_device))
            else:
                self.y_tgt = clip_model.encode_text(clip.tokenize(target_prompt).to(self._execution_device))
            self.clip_margin = clip_margin
            self.structure_margin = structure_margin
        self.cos_sim = torch.nn.CosineSimilarity(dim = -1)

        feature_src = {}
        feature_tgt = {}

        
        # 2. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        self.do_classifier_free_guidance = do_classifier_free_guidance
        x_in = x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 3. Encode input prompt = 2x77x1024
        with torch.no_grad():
            prompt_embeds = self._encode_prompt( prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.cycle_scheduler.set_timesteps(num_inference_steps, device=device)
        self.inv_scheduler.set_timesteps(num_inference_steps, device=device)
        self.cycle_inv_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels

        # randomly sample a latent code if not provided
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)
        
        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with torch.no_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.compute_noise(
                            do_classifier_free_guidance,
                            latent_model_input,
                            t,
                            prompt_embeds,
                            cross_attention_kwargs,
                            guidance_scale
                    )

                    feature_src[t.item()] = {}
                    for name, module in self.prep_unet.named_modules():
                        module_name = type(module).__name__
                        if ('down_blocks' in name or 'up_blocks' in name) and module_name == "ResnetBlock2D":
                            _f_map_src = module.f_map_value 
                            # print(_f_map_src.shape) # [2, C, H, W]
                            feature_src[t.item()][name] = _f_map_src.detach().cpu()
                            del _f_map_src

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        x_src = latents.detach().clone()

        # make the reference image (reconstruction)
        with torch.no_grad():
            image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))
            prompt_embeds_edit = self._encode_prompt(target_prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=None, negative_prompt_embeds=None,)
              
        x_tgt = latents_init
        # Second denoising loop for editing the text prompt
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        with self.progress_bar(total=2*num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                tp1 = t + self.scheduler.config.num_train_timesteps // num_inference_steps # t + 1
                tp2 = t + 2 * self.scheduler.config.num_train_timesteps // num_inference_steps # t + 2
                Tmt = timesteps[-i-1]
                if i != num_inference_steps - 1:
                    Tmtp1 = timesteps[-i-2]

                if i == 0:
                    alpha_T = self.scheduler.alphas_cumprod[t] # alpha_T
                    self.alpha_T = alpha_T
                    alpha_Tmt = self.inv_scheduler.alphas_cumprod[Tmt] # alpha_T-t
                    alpha_Tmtp1 = self.inv_scheduler.alphas_cumprod[Tmtp1]

                    fwd_guidance_input = torch.cat([x_src] * 2) if do_classifier_free_guidance else x_src
                    fwd_guidance_input = self.inv_scheduler.scale_model_input(fwd_guidance_input, Tmt)

                    _x_src = fwd_guidance_input.detach().clone()
                    _x_src.requires_grad = True
                    

                    if cycle_guidance != 0:                    
                        noise_pred_src = self.compute_noise(
                            do_classifier_free_guidance,
                            _x_src,
                            Tmt,
                            prompt_embeds,
                            cross_attention_kwargs,
                            guidance_scale
                        )
                    else:
                        with torch.no_grad():
                            noise_pred_src = self.compute_noise(
                                do_classifier_free_guidance,
                                _x_src,
                                Tmt,
                                prompt_embeds,
                                cross_attention_kwargs,
                                guidance_scale
                            )

                    x_tgt_T = self.get_x_T_hat(_x_src[0], alpha_T, alpha_Tmt, noise_pred_src)
                    

                    with torch.no_grad():
                        latent_model_input = torch.cat([x_tgt] * 2) if do_classifier_free_guidance else x_tgt
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        noise_pred_tgt = self.compute_noise(
                            do_classifier_free_guidance,
                            latent_model_input,
                            t,
                            prompt_embeds_edit.detach(),
                            cross_attention_kwargs,
                            guidance_scale
                        )
                        
                        x_tgt_0 = self.get_x_0_hat(x_tgt, alpha_T, noise_pred_tgt)

                    _cycle_loss = self.get_cycle_loss_src(x_tgt_0, 
                                                            x_tgt_T,
                                                            prompt_embeds_edit.detach().clone(), 
                                                            cross_attention_kwargs,
                                                            guidance_scale, 
                                                            extra_step_kwargs,
                                                            ) if cycle_guidance != 0 else 0


                    if cycle_guidance != 0: 
                        loss = (1.0 - i / num_inference_steps) * (cycle_guidance / 5) * _cycle_loss
                        loss.backward(retain_graph=False)
                        _grad = _x_src.grad.chunk(2)[0]
                    else:
                        _grad = 0.

                    with torch.no_grad():
                        x_src = _x_src.detach().chunk(2)[0]
                        x_src = self.inv_scheduler.step(noise_pred_src, Tmt, x_src, reverse=True, **extra_step_kwargs).prev_sample - _grad

                    
                    with torch.no_grad():
                        fwd_guidance_input = torch.cat([x_src] * 2) if do_classifier_free_guidance else x_src
                        fwd_guidance_input = self.inv_scheduler.scale_model_input(fwd_guidance_input, Tmtp1)
                        noise_pred_src = self.compute_noise(
                            do_classifier_free_guidance,
                            fwd_guidance_input,
                            Tmtp1,
                            prompt_embeds,
                            cross_attention_kwargs,
                            guidance_scale
                        )
                        x_tgt_T = self.get_x_T_hat(x_src, alpha_T, alpha_Tmtp1, noise_pred_src)

                    progress_bar.update()


                if i != 0:
                    alpha_t = self.scheduler.alphas_cumprod[t]
                    alpha_tp1 = self.scheduler.alphas_cumprod[tp1] # alpha_(t+1)
                    alpha_Tmt = self.inv_scheduler.alphas_cumprod[Tmt] # alpha_T-t
                    if i != num_inference_steps - 1:
                        alpha_Tmtp1 = self.inv_scheduler.alphas_cumprod[Tmtp1]
                    
                    latent_model_input = torch.cat([x_tgt] * 2) if do_classifier_free_guidance else x_tgt
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, tp1)

                    _x_tgt = latent_model_input.detach().clone()
                    _x_tgt.requires_grad = True

                    noise_pred_tgt = self.compute_noise(
                        do_classifier_free_guidance,
                        _x_tgt,
                        tp1,
                        prompt_embeds_edit.detach(),
                        cross_attention_kwargs,
                        guidance_scale
                    )

                    feature_tgt[tp1.item()] = {}
                    structure_loss = 0.0
                    for name, module in self.prep_unet.named_modules():
                        module_name = type(module).__name__
                        if ('down_blocks' in name or 'up_blocks' in name) and module_name == "ResnetBlock2D":
                            _f_map_tgt = module.f_map_value 
                            feature_tgt[tp1.item()][name] = _f_map_tgt.detach()
                            if i > 1:
                                _f_map_tgt_naive = feature_tgt[tp2.item()][name].detach().to(self._execution_device)
                            else:
                                _f_map_tgt_naive = None
                            _f_map_src = feature_src[tp1.item()][name].detach().to(self._execution_device)

                            structure_loss += self.get_structure_loss(_f_map_tgt, _f_map_tgt_naive, _f_map_src)
                            

                    if i > 1:
                        del feature_tgt[tp2.item()]
                        del feature_src[tp2.item()]
                    del _f_map_tgt
                    del _f_map_tgt_naive


                    # sample x_tgt_0
                    x_tgt_0 = self.get_x_0_hat(_x_tgt[0], alpha_tp1, noise_pred_tgt)
                    image_feature = clip_model.encode_image(self.decode_latents_for_grad(x_tgt_0))

                    # sample x_tgt_T
                    x_tgt_T = x_tgt_T.detach().clone() 

                    _cycle_loss = self.get_cycle_loss_tgt_1(x_tgt_0, 
                                                            x_tgt_T, 
                                                            prompt_embeds_edit.detach().clone(), 
                                                            cross_attention_kwargs,
                                                            guidance_scale, 
                                                            extra_step_kwargs,
                                                            ) if cycle_guidance != 0 else 0
                    
                    loss = (1 / self.get_lr(i, num_inference_steps)) * (1. - i / num_inference_steps) * cycle_guidance * _cycle_loss + clip_guidance * self.get_clip_loss(image_feature) + structure_guidance * structure_loss
                    loss.backward(retain_graph=False)
                    
                    _grad = _x_tgt.grad.chunk(2)[0]
                    with torch.no_grad():
                        x_tgt = _x_tgt.detach().chunk(2)[0]
                        x_tgt = self.scheduler.step(noise_pred_tgt, tp1, x_tgt, **extra_step_kwargs).prev_sample - self.get_lr(i, num_inference_steps) * _grad


                    with torch.no_grad():
                        latent_model_input = torch.cat([x_tgt] * 2) if do_classifier_free_guidance else x_tgt
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        noise_pred_tgt = self.compute_noise(
                            do_classifier_free_guidance,
                            latent_model_input,
                            t,
                            prompt_embeds_edit.detach(),
                            cross_attention_kwargs,
                            guidance_scale
                        )
                        x_tgt_0 = self.get_x_0_hat(x_tgt, alpha_t, noise_pred_tgt)

                    progress_bar.update()
                    
                    

                    fwd_guidance_input = torch.cat([x_src] * 2) if do_classifier_free_guidance else x_src
                    fwd_guidance_input = self.inv_scheduler.scale_model_input(fwd_guidance_input, Tmt)

                    _x_src = fwd_guidance_input.detach().clone()
                    _x_src.requires_grad = True

                    if cycle_guidance != 0:                    
                        noise_pred_src = self.compute_noise(
                            do_classifier_free_guidance,
                            _x_src,
                            Tmt,
                            prompt_embeds.detach(),
                            cross_attention_kwargs,
                            guidance_scale
                        )
                    else:
                        with torch.no_grad():
                              noise_pred_src = self.compute_noise(
                                  do_classifier_free_guidance,
                                  _x_src,
                                  Tmt,
                                  prompt_embeds.detach(),
                                  cross_attention_kwargs,
                                  guidance_scale
                              )

                    # sample x_tgt_0
                    x_tgt_0 = x_tgt_0.detach().clone() 

                    # sample x_tgt_T
                    x_tgt_T = self.get_x_T_hat(_x_src[0], alpha_T, alpha_Tmt, noise_pred_src)

                    _cycle_loss = self.get_cycle_loss_src(x_tgt_0, 
                                                          x_tgt_T,
                                                          prompt_embeds_edit.detach().clone(), 
                                                          cross_attention_kwargs,
                                                          guidance_scale, 
                                                          extra_step_kwargs,
                                                          ) if cycle_guidance != 0 else 0
                    
                    if cycle_guidance != 0: 
                        loss = (1.0 - i / num_inference_steps) * (cycle_guidance / 5) * _cycle_loss
                        loss.backward(retain_graph=False)
                        _grad = _x_src.grad.chunk(2)[0]
                    else:
                        _grad = 0.
                    

                    with torch.no_grad():
                        x_src = _x_src.detach().chunk(2)[0]
                        x_src = self.inv_scheduler.step(noise_pred_src, Tmt, x_src, reverse=True, **extra_step_kwargs).prev_sample - _grad
                    
                    if i != num_inference_steps - 1:
                        with torch.no_grad():
                            fwd_guidance_input = torch.cat([x_src] * 2) if do_classifier_free_guidance else x_src
                            fwd_guidance_input = self.inv_scheduler.scale_model_input(fwd_guidance_input, Tmtp1)

                            noise_pred_src = self.compute_noise(
                                do_classifier_free_guidance,
                                fwd_guidance_input,
                                Tmtp1,
                                prompt_embeds,
                                cross_attention_kwargs,
                                guidance_scale
                            )
                            x_tgt_T = self.get_x_T_hat(x_src, alpha_T, alpha_Tmtp1, noise_pred_src)
                    else:
                        x_tgt_T = x_src.detach().clone()
                    progress_bar.update()


                if i == num_inference_steps - 1:
                    alpha_t = self.scheduler.alphas_cumprod[t] # alpha_t

                    latent_model_input = torch.cat([x_tgt] * 2) if do_classifier_free_guidance else x_tgt
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    _x_tgt = latent_model_input.detach().clone()
                    _x_tgt.requires_grad_(True)

                    noise_pred_tgt = self.compute_noise(
                        do_classifier_free_guidance,
                        _x_tgt,
                        t,
                        prompt_embeds_edit,
                        cross_attention_kwargs,
                        guidance_scale
                    )

                    
                    feature_tgt[t.item()] = {}
                    structure_loss = 0.0
                    for name, module in self.prep_unet.named_modules():
                        module_name = type(module).__name__
                        if ('down_blocks' in name or 'up_blocks' in name) and module_name == "ResnetBlock2D":
                            _f_map_tgt = module.f_map_value 
                            feature_tgt[t.item()][name] = _f_map_tgt.detach()
                            if i > 1:
                                _f_map_tgt_naive = feature_tgt[tp1.item()][name].detach().to(self._execution_device)
                            else:
                                _f_map_tgt_naive = None
                            _f_map_src = feature_src[t.item()][name].detach().to(self._execution_device)

                            structure_loss += self.get_structure_loss(_f_map_tgt, _f_map_tgt_naive, _f_map_src)
                            

                    if i > 1:
                        del feature_tgt[tp1.item()]
                    del feature_src[tp1.item()]
                    del _f_map_tgt
                    del _f_map_tgt_naive

                    # sample x_tgt_0
                    x_tgt_0 = self.get_x_0_hat(_x_tgt[0], alpha_t, noise_pred_tgt)
                    image_feature = clip_model.encode_image(self.decode_latents_for_grad(x_tgt_0))

                    # sample x_tgt_T
                    x_tgt_T = x_tgt_T.detach().clone() 

                    _cycle_loss = self.get_cycle_loss_tgt_1(x_tgt_0, 
                                                            x_tgt_T, 
                                                            prompt_embeds_edit.detach().clone(), 
                                                            cross_attention_kwargs,
                                                            guidance_scale, 
                                                            extra_step_kwargs,
                                                            ) if cycle_guidance != 0 else 0

                    loss = (1 / self.get_lr(i, num_inference_steps)) * (1.0 - i / num_inference_steps) * cycle_guidance * _cycle_loss + clip_guidance * self.get_clip_loss(image_feature) + structure_guidance * structure_loss
                    loss.backward(retain_graph=False)

                    _grad = _x_tgt.grad.chunk(2)[0]
                    with torch.no_grad():
                        x_tgt = _x_tgt.detach().chunk(2)[0]
                        x_tgt = self.scheduler.step(noise_pred_tgt, t, x_tgt, **extra_step_kwargs).prev_sample - self.get_lr(i, num_inference_steps) * _grad
                    progress_bar.update()
                    


        # 8. Post-processing
        with torch.no_grad():
            image = self.decode_latents(x_tgt.detach())

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        with torch.no_grad():
            image_edit = self.numpy_to_pil(image)

        return image_rec, image_edit