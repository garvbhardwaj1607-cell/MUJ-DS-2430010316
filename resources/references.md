# References

Dataset
1. D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," arXiv:1511.08060, 2015.
2. S. P. Mohanty, D. P. Hughes and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science* 7:1419, 2016. (Reports about 99.35 % with a fine-tuned GoogLeNet on a different split protocol; cited for context, not reproduced here.)
3. PlantVillage-Dataset repository: https://github.com/spMohanty/PlantVillage-Dataset (commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`).
4. M. A. Noyan, "Uncovering bias in the PlantVillage dataset," arXiv:2206.04374, 2022. (Motivation for the background-robustness experiment; please verify the exact claims in the paper before quoting them.)

Models and methods
5. A. Howard et al., "Searching for MobileNetV3," ICCV 2019.
6. M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," ICML 2019.
7. K. He et al., "Deep residual learning for image recognition," CVPR 2016. (ResNet-style reference model is a small re-implementation, not the published network.)
8. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," ICLR 2021. (Tiny ViT reference is a small re-implementation.)
9. J. Hu, L. Shen and G. Sun, "Squeeze-and-excitation networks," CVPR 2018. (Component A.)
10. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," ICCV 2017.
11. R. Wightman, "PyTorch Image Models (timm)," https://github.com/huggingface/pytorch-image-models.
12. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," ICLR 2019. (AdamW.)
13. L. N. Smith and N. Topin, "Super-convergence: very fast training of neural networks using large learning rates," 2019. (OneCycle schedule.)
14. C. Guo et al., "On calibration of modern neural networks," ICML 2017. (Expected calibration error.)
