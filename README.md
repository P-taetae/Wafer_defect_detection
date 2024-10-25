# A Stratified Augmentation-Guided Approach for Detection and Classification of Wafer Defects

Official Pytorch implementation of [A Stratified Augmentation-Guided Approach for Detection and Classification of Wafer Defects] (On review).

Authors: Taekyeong Park, Yongho Son, Sanghyuk Moon, Seungju Han, Je Hyeong Hong


## Abstract
In semiconductor engineering, high yield of wafers relies on accurate detection and classification of wafer defects.
The primary challenges are: i) variable backgrounds and differing image scales, which complicate traditional classification techniques; ii) a significantly smaller amount of defect data compared to normal data; and iii) an imbalanced data distribution with a long-tail of defect types. Both ii) and iii) challenge traditional classification techniques.
To address these issues, we introduce a stratified framework called WaferDC, designed for detecting and classifying wafer defects from scanning electron microscope (SEM) images. 
Our framework comprises three modules, namely defect detection, defect classification and defect augmentation all of which are intertwined for optimal  performance.
For the detection phase, we introduce a multi-class memory bank to improve robustness of defect detection against different textures by separating input data based on texture characteristics.
For the classification stage, we pass defect-classified images through a PEFT-based classifier\cite{pel} utilizing a vision transformer.
We employ a new augmentation module called SegMix to create synthetic defect images utilizing anomaly heatmaps thereby raising the reliability of defect detection and classification under long-tailed imbalanced environment.

## Contribution
- We introduce WaferDC, a novel stratified augmentation-guided approach designed to address the imbalance in learning datasets for wafer defects, which typically exhibit a long-tailed distribution, aiming to enhance defect detection and classification performance in SEM image environments.

- To improve the performance of semiconductor image defect detection, we propose a multi-class memory bank. 
This approach can improve the defect detect performance on wafer data with various normal images, and this improvement contributes to the quality of the augmented data used later.

- We introduce SegMix, an advanced augmentation technique based on CutMix\cite{cutmix}, designed to refine wafer defect image augmentation.
SegMix, utilizing segmentation maps from a multi-class memory bank, precisely extracts and integrates defect areas, enhancing defect classification accuracy and aiding in adjusting thresholds for the multi-memory bank system.

## requirements:
Our results were computed using Python 3.8, with packages and respective version noted in requirements.txt

## How to run our code
1. defect_detection

Follow the 1step_defect_detection

Training :
```
python step1_1_train_k_means.py
python step1_2_normal_augmentation.py
python step1_3_run_patchcore.py
bash step1_4_abnormal_augmentation_magnetic.sh
python step1_5_reset_folder.py
```

Testing :
```
python test1_1_k_means.py
bash test1_2_load_and_evaluate_magnetic.sh 
```

2. defect_classification


After the 1step_defect_detection

Use main.py to train and test the model.

This code is written based on the following code : https://github.com/amazon-science/patchcore-inspection , https://github.com/shijxcs/LIFT

## Citation
---

## License


## Acknowledgement
