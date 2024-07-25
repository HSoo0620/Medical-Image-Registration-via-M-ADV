# Deformable-Medical-Image-Registration-via-Multiveiw-Adversarial-Learning

Implement of "Deformable-Medical-Image-Registration-via-Multiveiw-Adversarial-Learning"

## Registration Network Architecture
<img src = "https://github.com/user-attachments/assets/c919978b-5544-402b-b7cc-534099103107" width="100%" height="100%">

## Multiview Adversarial Learning
<img src = "https://github.com/user-attachments/assets/ad82f3d7-4bc9-4c1c-b9aa-e3b750f44b85" width="100%" height="100%">

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
<img src = "https://github.com/user-attachments/assets/7c3fc58c-2225-4d8f-806f-1c716b9a0fa1" width="100%" height="100%">

- **3D Visualization** <br/>
<img src = "https://github.com/user-attachments/assets/d8b4073b-2fb2-40de-8148-a22473ded7df" width="100%" height="100%">

- **2D Visualization** <br/>
<img src = "https://github.com/user-attachments/assets/c8ffb64a-9f13-4127-8a17-d31246958926" width="100%" height="100%">


### Brain MRI
<img src = "https://github.com/user-attachments/assets/6463d908-9b61-4216-adf3-85976b71a6f7" width="100%" height="100%">

- **2D Visualization** <br/>
<img src = "https://github.com/user-attachments/assets/97d6e91c-9a5d-44e6-9846-4a44a155382b" width="100%" height="100%">

<!-- <img src = "https://github.com/user-attachments/assets/69ac1182-0bed-4e4a-afd4-4a62f55584d5" width="40%" height="40%"> -->

## Ablation Studied
<img src = "https://github.com/user-attachments/assets/e4caba4c-1acd-43db-8737-3e3154125081" width="100%" height="100%">

## Reference Codes
- [https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- [https://github.com/voxelmorph/voxelmorph](https://github.com/voxelmorph/voxelmorph)
- [https://github.com/zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)
- [https://github.com/microsoft/CvT](https://github.com/microsoft/CvT)
