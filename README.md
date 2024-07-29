# Deformable-Medical-Image-Registration-via-Multiveiw-Adversarial-Learning

Implement of "Deformable-Medical-Image-Registration-via-Multiveiw-Adversarial-Learning"

## Registration Network Architecture
<img src = "https://github.com/user-attachments/assets/2255c190-9c5b-4771-b50c-23319cfeabf4" width="100%" height="100%">

## Multiview Adversarial Learning
<img src = "https://github.com/user-attachments/assets/8d8e379c-22a0-402b-877a-9a1415fbb710" width="100%" height="100%">

## Prerequisites
- [python==3.8.8](https://www.python.org/)  <br/>
- [pytorch==1.8.1](https://pytorch.org/get-started/locally/)

## Installation
The required packages are located in ```requirements```.

    pip install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
    pip install -r requirement.txt

## Dataset
### Abdominal CT
BTCV Datasets Link : https://www.synapse.org/Synapse:syn3193805/wiki/217789 <br/>
Abdominal 1K Datasets Link : https://github.com/JunMa11/AbdomenCT-1K <br/>

Organs consist of Spleen, Kidney, Gallbladder, Esophagus, Liver, Stomach, Aorta, Inferior Vena Cava, Pancreas, Right and Left Adrenal Gland.

### Brain MRI
LPBA40 Link: https://www.loni.usc.edu/research/atlas_downloads <br/>

Brain MRI contains 40 T1-weighted MR images annotated with 56 subcortical ROIs. We combine the 56 labels into 7 region labels (i.e., Frontal Lobe, Parietal Lobe, Occipital Lobe, Temporal Lobe, Cingulate Lobe, Putamen, and Hippocampus) defined according to the main clinical structures of the brain.

## Training
- Before training, pre-processing and initial affine registration must be performed.
  - For pre-processing, reference ```preprocessing_BTCV_1K.py```, ```preprocessing_lpba40.py```.
  - For initial affine registration, reference ```Train_Affine.py```. <br/>
```python
python Train_M_Adv.py \
    --affine_model experiments/affine \
    --dataset_dir Dataset/BTCV_Abdominal_1k \
    --save_validation_img True \
    --max_epoch 100 \
```
## Inference
```python
python Inference_M_Adv.py --affine_model experiments/affine --dataset_dir Dataset/BTCV_Abdominal_1k
```

## Main results
### Abdominal CT
<img src = "https://github.com/user-attachments/assets/fd745f12-f4ab-4b31-b2e8-796cfd2dc2a7" width="100%" height="100%">

- **3D Visualization** <br/>
<img src = "https://github.com/user-attachments/assets/6daa787b-fc89-4c85-8d26-8821bb0e79d2" width="100%" height="100%">

- **2D Visualization** <br/>
<img src = "https://github.com/user-attachments/assets/c8ffb64a-9f13-4127-8a17-d31246958926" width="100%" height="100%">


### Brain MRI
<img src = "https://github.com/user-attachments/assets/f2bb6c3e-c5f1-497e-baa5-91d39026ddf4" width="100%" height="100%">

- **2D Visualization** <br/>
<img src = "https://github.com/user-attachments/assets/41f1e238-7695-4463-87e9-adc001378e81" width="100%" height="100%">

<!-- <img src = "https://github.com/user-attachments/assets/69ac1182-0bed-4e4a-afd4-4a62f55584d5" width="40%" height="40%"> -->

## Ablation Studied
<img src = "https://github.com/user-attachments/assets/9c9c7d42-150a-4547-8b86-e8c5885fd640" width="100%" height="100%">

## Reference Codes
- [https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- [https://github.com/voxelmorph/voxelmorph](https://github.com/voxelmorph/voxelmorph)
- [https://github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)
- [https://github.com/microsoft/CvT](https://github.com/microsoft/CvT)
