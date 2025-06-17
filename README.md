# Diffusion-Based Conditional Image Editing through Optimized Inference with Guidance

Official implementation of "Diffusion-Based Conditional Image Editing through Optimized Inference with Guidance" (WACV 2025). [[paper](https://arxiv.org/pdf/2412.15798)]



### Environment installation

Please check `requirements.txt`.

```
pip install -r requirements.txt
```

### Edit synthetic images with OIG

We can generate an synthetic image using the pretrained Stable Diffusion and edit with OIG. Note that `posterior_guidance` is a hyperparameter related to guidance scale.


```
python src/edit_synthetic.py --source_prompt "a photo of a street with trees" \
                             --target_prompt "a photo of a street with trees in a snowy day" \
                             --num_ddim_steps 50 \
                             --random_seed 0 \
                             --results_folder 'output/synth_edit' \
                             --clip_guidance <LAMBDA_1> --structure_guidance <LAMBDA_2>
```

Reconstructed and edited images are saved as `reconstruction.png` and `edit.png` inside `--results_folder` directory. 

Note that the value of the arguments `--clip_guidance` and `--structure_guidance` correspond to the value of the hyperparameters $\lambda _1$ and $\lambda _2$, respectively. The values of $\beta _p$ and $\beta _f$ can be modified by using `--beta_p` and `--beta_f` argument, respectively.

We recommend setting `<LAMBDA_1>` to a value close to `0.1`. For `<LAMBDA_2>`, we suggest choosing a value from `{0.5, 1.0, 1.5}`. For the sensitivity analysis of `<LAMBDA_2>`, please refer to Figure 5 in the main paper.




### Edit real images with OIG 

To edit a real image with OIG, firstly do the DDIM inversion using the command:

```
python src/inversion.py --input_image "data/cat.png" \
                        --results_folder "output/test_cat" \
                        --num_ddim_steps 50
```

Then, edit the real image using the inverted latent and the source prompt. 

```
python src/edit_real.py --inversion "output/test_cat/inversion/" \
                        --prompt "output/test_cat/prompt/" --num_ddim_steps 50 \
                        --results_folder 'output/test_cat/' \
                        --task_name "cat2dog" \
                        --clip_guidance <LAMBDA_1> --structure_guidance <LAMBDA_2>
```


All files are saved in `--results_folder` directory:

```
output/test_cat
  ├── inversion
  │   ├── cat.pt
  │   └── ...
  ├── prompt
  │   ├── cat.txt
  │   └── ...
  ├── edit
  │   ├── cat.png
  │   └── ...
  └── reconstruction
      ├── cat.png
      └── ...
 ```
 
Reconstructed image from DDIM inversion is saved in `reconstruction/`, and edited image is saved in `edit/` directory.

### Edit images with OIG $^{+}$

We can use OIG $^{+}$ to edit the synthetic and real images by setting the value of the argument  `--cycle_guidance` to the value of hyperparameter $\lambda _4$. Note that the value of $\lambda _3$ depends on $\lambda _4$, i.e. $\lambda _3 = 0.2 \lambda _4$. For detailed explanation of OIG $^{+}$, please refer to the Appendix B of the paper.


### Acknowledgments

This method is implemented based on [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero/).



### Citation

``` bibtex

@inproceedings{lee2025diffusion,
  title={Diffusion-Based Conditional Image Editing through Optimized Inference with Guidance},
  author={Lee, Hyunsoo and Kang, Minsoo and Han, Bohyung},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}

```
