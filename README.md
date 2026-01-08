
This script is designed to train a Wasserstein GAN (WGAN) for downscaling CMIP6 climate projections from approximately 50 km to 4 km spatial resolution. The domain covers 121°W–100°W in longitude and 48°N–61°N in latitude.

Users should update the directory paths to match their local file system. In the code, `y_train` corresponds to the original fine-resolution dataset, while `x_train` represents the upscaled (coarse-resolution) input data.

The dataset is split chronologically, with 80% used for training and 20% for testing. This split can be modified by the user if needed.

After training, the WGAN model is saved automatically. A pretrained model is also included in this repository for reference and reproducibility.

