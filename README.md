
## Overview
This script is designed to train a Wasserstein GAN (WGAN) for downscaling CMIP6 climate projections from approximately 50 km to 4 km spatial resolution. The domain covers 121°W–100°W in longitude and 48°N–61°N in latitude.

Users should update the directory paths to match their local file system. In the code, `y_train` corresponds to the original fine-resolution dataset, while `x_train` represents the upscaled (coarse-resolution) input data.

The dataset is split chronologically, with 80% used for training and 20% for testing. This split can be modified by the user if needed.

After training, the WGAN model is saved automatically. A pretrained model is also included in this repository for reference and reproducibility.

## UNET pretraining

This repository also includes code for pretraining a **UNET** model, which can be used as a deterministic baseline or as a precursor to the WGAN training.

The UNET is trained to map coarse-resolution precipitation fields (`x_train`) directly to fine-resolution targets (`y_train`) using supervised learning. The same preprocessing, spatial domain, and chronological train–test split (80% / 20%) are applied to ensure consistency with the WGAN experiments.

An example of a pretrained UNET model is provided in the repository for reference. Users may retrain the UNET using the supplied scripts by adjusting the data paths and training parameters as needed.

## Computational cost (example)

For reference, training the WGAN on a dataset consisting of approximately **15 years of hourly precipitation at 4 km resolution** (∼**1 TB** of data) required about **28 hours** of wall-clock time using **3 NVIDIA A100-SXM4 GPUs (40 GB each)**.

Actual training time will vary depending on data volume, model configuration, and hardware.


