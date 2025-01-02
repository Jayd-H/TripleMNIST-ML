# ML Assignment 2024 - Triple MNIST Dataset

**ALL PYTHON SCRIPTS RAN FROM ROOT (./)**

This is quite a botch job, some example code shown favours the brief specification, rather than real-world efficiency. /data and /results for scripts have been omitted for brevity. The assignment doc has been omitted for confidentiality. This repository is meant to supplement the report. 

### Introduction

Throughout this report, we explore the progression from basic machine learning approaches to advanced neural networks in tracking multi-digit recognition. Each task builds upon previous findings, demonstrating how increasingly sophisticated techniques can handle the complexities of sequential digital classification. At each step, we will analyse the efficacy of the model implemented, explaining its shortcomings and positives. All code was run using PyTorch (with DirectML because of superior AMD support on Windows 11) with an AMD 6950 XT GPU and an AMD Ryzen 7800X3D CPU. 

# Example Plots

## Task 1 

![image](https://github.com/user-attachments/assets/e4752b83-eb0b-45ad-b13b-5818e7289911)

## Task 2: Logistic Regression

![error_distribution](https://github.com/user-attachments/assets/11ccd4fe-7b21-4165-9d7a-73292a4dcb7d)

![most_confused_pairs](https://github.com/user-attachments/assets/d644784b-fa2c-48fc-a715-293fea409627)

![training_history](https://github.com/user-attachments/assets/3121f4df-1ea9-4767-a2b0-1aedac5b278c)

## Task 2: Full-Image CNN

![error_distribution](https://github.com/user-attachments/assets/6a2d9d84-ebf8-4a20-8e56-5bc13cda9668)

![most_confused_pairs](https://github.com/user-attachments/assets/fcfe7f3e-3fc1-4275-b3d5-44bb4ff23929)

![training_history](https://github.com/user-attachments/assets/d1767fde-f1f4-4f66-9114-3cfc45eaa7b8)

![feature_map_epoch_10](https://github.com/user-attachments/assets/5f4f0f6e-7dc4-45e4-949f-37dc1427c54d)

## Task 3: Individual Digit Assessed CNN

![confusion_matrices](https://github.com/user-attachments/assets/be823fa2-5220-4a9d-8dcc-9b31757d430e)

![error_distribution](https://github.com/user-attachments/assets/eb370562-dfe7-48ed-a3f0-eb41cd3469c4)

![training_history](https://github.com/user-attachments/assets/f2e05ed4-b53a-49da-8b52-b0f370ebdc89)

## Task 4: Enchanced CNN

![training_history](https://github.com/user-attachments/assets/d0f79c1a-e317-484a-81d6-f3fef85f1f9b)

![error_distribution](https://github.com/user-attachments/assets/c694c00e-5d2e-40ca-87b7-0dcb39da256a)

![confusion_matrices](https://github.com/user-attachments/assets/c9f17953-2e57-463d-8ea8-d38c3b8bcb45)

![confidence_analysis](https://github.com/user-attachments/assets/e6b4ab1d-5c1c-44a5-aa73-f601b32d38be)

![augmentation_examples](https://github.com/user-attachments/assets/fef2ee10-fb62-46db-a193-af0b1f7bfbb1)

![attention_maps_epoch_250](https://github.com/user-attachments/assets/5cdaee9d-277e-4db0-b942-9ece2c9bd50e)

![layer_activations_epoch_250](https://github.com/user-attachments/assets/61e88ef1-8cea-425c-a3d2-2570ccaa0c70)

## Task 5: cDCGAN

![training_history](https://github.com/user-attachments/assets/d9db5356-e753-4062-b3ff-584346bf42bb)

![grid_epoch_200](https://github.com/user-attachments/assets/d081ec11-97b3-4850-8ead-1be2d0aa9814)

## Task 5: Augmented CNN

![training_comparison](https://github.com/user-attachments/assets/d596d3f4-c10e-4833-b0a0-e2e604d53867)

![final_comparison](https://github.com/user-attachments/assets/b66d27cc-eb8b-4aad-b0db-4e91df9c0c90)

#### Real Only

![confusion_matrices](https://github.com/user-attachments/assets/4b4bfc7c-535e-4938-a52d-1dfecfa4e413)

![training_history](https://github.com/user-attachments/assets/af4a80cb-471d-40ac-8571-642141801364)

#### Combined

![confusion_matrices](https://github.com/user-attachments/assets/73a1de56-1b0d-438e-8370-00a1c51158e1)

![training_history](https://github.com/user-attachments/assets/edb4f313-5edc-465a-99ad-d0a143ae7e9a)



