# Aimonk Multilabel Problem

Multi-label image classification project.

## Project Overview

This project implements a deep learning pipeline for multi-label classification on a dataset with four distinct attributes. The solution addresses real-world data challenges, including missing annotations (NA values) and significant class imbalance.

## Technical Approach

### 1) Framework and Architecture

- Framework: Implemented using TensorFlow 2.x.
- Model Architecture: Utilized ResNet50V2, an established deep residual network known for its stability and performance.
- Transfer Learning: The model is not trained from scratch. It uses ImageNet pre-trained weights as a foundation, followed by fine-tuning of the final dense layers to adapt to the specific 4-attribute classification task.

### 2) Handling Missing Data (NA Values)

To avoid wasting data by discarding rows with NA values, a masked loss function was developed.

- Mechanism: A binary mask is generated for every image (1 for available labels, 0 for NA).
- Integration: The binary cross-entropy (BCE) loss is multiplied by this mask element-wise.
- Outcome: The model is only graded on attributes where the ground truth is known, allowing it to learn from every image in the dataset regardless of partial missing information.
- The inspiration taken from the paper known as 1.Attention that all you need  2.Loss Mask is all you Need

### 3) Tackling Class Imbalance

The labels.txt file reveals a skewed distribution where some attributes are much more frequent than others. The following techniques were implemented or considered:

- Weighted Loss: simply give the weights for the each loss computed by the loss function during training so NA values does not affect the results.
- K-Fold Cross-Validation: To make the model consistent over the dataset here K-Fold cross validation is used with K=5.
- You can see the Loss graph for all K-Folds in this repo. 

## Deliverables

- Training Code: Modular Python script that processes labels.txt, verifies image existence, and produces a saved weights file (.h5 or .ckpt).
- Loss Curve: A plot titled "Aimonk_multilabel_problem" showing training_loss vs iteration_number for all K-folds to demonstrate convergence.
- Inference Code: A dedicated function or script that accepts a single image, applies a 0.5 probability threshold, and prints the list of detected attributes.

## Potential Improvements and Future Work

Due to time constraints, the following enhancements are proposed for future iterations.

### Pre-processing and Augmentations

- Data Augmentation: (All types of transformations can be used) Implementing RandomFlip, RandomRotation, and ColorJitter would increase the model's robustness to varying lighting and orientations.
- Mixup Training: Blending two images and their labels to help the model generalize better in multi-label scenarios.

### Advanced Imbalance Handling

- Iterative Stratification: Using multi-label stratification for K-fold splits to ensure the ratio of positive and negative samples for each attribute is identical across all folds.
- Focal Loss: Replacing standard BCE with focal loss to focus the training specifically on hard examples that the model frequently misclassifies.

### Optimization

- Threshold Tuning: Instead of a global 0.5 threshold, calculating the optimal F1-score threshold for each attribute individually based on the validation set.
- Learning Rate Scheduling: Implementing a ReduceLROnPlateau callback to fine-tune the weights more granularly once the loss curve flattens.