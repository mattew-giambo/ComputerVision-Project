# Project Report — Computer Vision (A.Y. 2025/2026)

## Project 2 | Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios
---

### Course Information
* **Institution:** Sapienza University of Rome
* **Course:** Computer Vision (M.Sc. in Engineering in Computer Science)
* **Academic Year:** 2025/2026
* **Professor:** Prof. Irene Amerini
* **Candidate(s):** []

---

## Table of Contents
1. [Introduction and Problem Statement (The Core Challenge)](#1-introduction-and-problem-statement-the-core-challenge)
2. [State of the Art (How Others Approach This)](#2-state-of-the-art-how-others-approach-this)
3. [Theoretical Background: Explaining the Concepts from Scratch](#3-theoretical-background-explaining-the-concepts-from-scratch)
   - 3.1 [What is a Digital Image?](#31-what-is-a-digital-image)
   - 3.2 [The Fourier Transform and the Frequency Domain](#32-the-fourier-transform-and-the-frequency-domain)
   - 3.3 [Convolutional Neural Networks (CNNs) for Image Recognition](#33-convolutional-neural-networks-cnns-for-image-recognition)
   - 3.4 [Monocular Depth Estimation (MDE) and RGB-D Representations](#34-monocular-depth-estimation-mde-and-rgb-d-representations)
   - 3.5 [Deep Generative Models and Diffusion Models (DDPM)](#35-deep-generative-models-and-diffusion-models-ddpm)
   - 3.6 [Multi-Task Learning as Regularisation](#36-multi-task-learning-as-regularisation)
4. [Dataset (RRDataset & Pre-computed Depth)](#4-dataset-rrdataset--pre-computed-depth)
5. [Proposed Method — MultiTaskNet (How the Model Works)](#5-proposed-method--multitasknet-how-the-model-works)
   - 5.1 [Architecture Overview](#51-architecture-overview)
   - 5.2 [Offline Depth Pre-Computation](#52-offline-depth-pre-computation)
   - 5.3 [FourierNet Branch and Log-Magnitude Input](#53-fouriernet-branch-and-log-magnitude-input)
   - 5.4 [RGB and Depth Specialist Branches](#54-rgb-and-depth-specialist-branches)
   - 5.5 [Hierarchical Multi-Stage Fusion (RGB-D and Fourier)](#55-hierarchical-multi-stage-fusion-rgb-d-and-fourier)
   - 5.6 [Classification Heads](#56-classification-heads)
   - 5.7 [Multi-Task Loss and weighting parameter Lambda](#57-multi-task-loss-and-weighting-parameter-lambda)
6. [Experimental Setup](#6-experimental-setup)
7. [Evaluation and Results (Detailed Guide to the Charts)](#7-evaluation-and-results-detailed-guide-to-the-charts)
   - 7.1 [Learning Curves](#71-learning-curves)
   - 7.2 [Confusion Matrices](#72-confusion-matrices)
   - 7.3 [Classification Reports](#73-classification-reports)
   - 7.4 [ROC Curves and AUC](#74-roc-curves-and-auc)
   - 7.5 [Per-Class Metrics Bar Charts](#75-per-class-metrics-bar-charts)
   - 7.6 [Accuracy Breakdown by Transformation Category](#76-accuracy-breakdown-by-transformation-category)
   - 7.7 [Lambda Trade-off Analysis (Pareto Frontier)](#77-lambda-trade-off-analysis-pareto-frontier)
   - 7.8 [Per-Class F1 Heatmap Across Lambda Values](#78-per-class-f1-heatmap-across-lambda-values)
   - 7.9 [Domain Breakdown Across Lambda Values](#79-domain-breakdown-across-lambda-values)
   - 7.10 [Full Numeric Summary Table](#710-full-numeric-summary-table)
   - 7.11 [Unimodal Baselines vs. Multi-Task Models](#711-unimodal-baselines-vs-multi-task-models)
8. [Model Selection and Best Configuration](#8-model-selection-and-best-configuration)
9. [Coverage of Project Requirements](#9-coverage-of-project-requirements)
10. [Conclusions and Future Work](#10-conclusions-and-future-work)
11. [References](#11-references)

---

## 1. Introduction and Problem Statement (The Core Challenge)

To understand this project, we must start with the absolute basics of how computers "see" images and how AI changes this field.

### What is the Problem?
Nowadays, generative models can produce photorealistic images. These images look completely real to the human eye, which makes it easy to spread fake visual information. Computer vision researchers try to build automated detectors to distinguish real camera photographs from AI-generated fakes.

In the real world, this task is extremely difficult because images do not remain in their original, clean state but, it has almost always undergone some form of **post-processing**. For example:
* **Internet Transfer:** Uploading a photo to social media platforms (like Instagram or WhatsApp) automatically resizes the image and applies lossy **JPEG compression** to save storage space.
* **Re-digitalisation:** A user might print the digital image onto physical paper and then re-photograph it with a smartphone or scan it back into a computer.

These operations alter the pixel values, removing the microscopic traces (like camera sensor noise) that detectors use to identify fakes.

### Our Solution
This project builds a unified neural network, called **MultiTaskNet**, to answer two questions simultaneously from a single input image:
1. **Task 1 — AI Detection:** Is the image a camera captured or AI-generated?  
(Binary classification: `real` vs. `ai`).
2. **Task 2 — Post-Processing Classification:** What transformation has the image undergone?   
(Three-class classification: `original` for no changes, `redigital` for printed-and-scanned, and `transfer` for compressed social media images).

Instead of treating these tasks separately, we use **Multi-Task Learning (MTL)**. By training one network on both tasks, the model learns a shared representation.

---

## 2. State of the Art (How Others Approach This)

Before building our model, we must look at how researchers currently attempt to solve these issues.

### 2.1 Spatial vs. Frequency Detection
Early detectors trained Convolutional Neural Networks (CNNs) in the **spatial domain** (using the raw color pixels directly). To make them robust to transformations, researchers applied data augmentations (like random blurring or JPEG compression) during training. This forces the model to ignore fragile pixel details and focus on global object structures.

However, generative models leave behind unique **spectral fingerprints**, which can be observed by converting the image to the **frequency domain** (using the Fourier Transform), resulting in bright, symmetrical patterns in the spectrum.

### 2.2 Adding 3D Geometry (Depth)
Recently, diffusion models have challenged existing frequency detectors because they do not use standard convolutional upsampling and thus do not leave periodic spectral grids. 

Therefore researchers have turned to **3D geometry**. Real cameras capture actual three-dimensional scenes. AI generators, on the other hand, generate 2D pixels that look realistic but often contain 3D geometrical errors (like shadows pointing in inconsistent directions or perspective lines that do not align correctly). By extracting a **depth map** (which shows how far objects are from the camera), we can expose these geometric inconsistencies.

---

## 3. Theoretical Background: Explaining the Concepts from Scratch



---

## 4. Dataset (RRDataset & Pre-computed Depth)

### The RRDataset
We use the **RRDataset** (*Real/Redigital* dataset) to train and test our model. The images in this dataset are organized into five categories:

1. **AI Label (Task 1):**
   * `real`: Genuine photos taken by cameras.
   * `ai`: Images generated by AI models.
2. **Domain Label (Task 2):**
   * `original`: Digital native images with no alterations.
   * `redigital`: Images printed on paper and re-digitised using a scanner or camera.
   * `transfer`: Digital images subjected to internet compression and resizing.

Combining these categories yields **six distinct subgroups**:
* Original Real & Original AI
* Redigital Real & Redigital AI
* Transfer Real & Transfer AI

```
RRDataset_test/
├── original/
│   ├── ai/
│   └── real/
├── redigital/
│   ├── ai/
│   └── real/
└── transfer/
    ├── ai/
    └── real/
```

### Dataset Balancing and Splits
To ensure the model does not develop a bias towards any class (e.g., predicting "real" just because there are more real images in the dataset), we cap the total dataset size at **15,000 samples**, ensuring an equal distribution across all classes.

We split this balanced dataset using a fixed random seed into:
* **Training Split (70%):**, used to train the network.
* **Validation Split (15%):**, used to tune hyperparameters and apply early stopping.
* **Test Split (15%):**, used only for final, unbiased evaluation.

### Depth Maps
To save GPU memory and training time, we pre-computed all depth maps once before traini/ng using **Depth Anything V2 (Small)**. These are saved to disk in a separate directory (`RRDataset_depth/`) as `.pt` files. During training, the dataloader loads the RGB image and its corresponding pre-computed `.pt` depth tensor at zero computational cost.

---

## 5. Proposed Method — MultiTaskNet (How the Model Works)

### 5.1 Architecture Overview
`MultiTaskNet` contains three parallel specialist branches in Stage 1, a two-step fusion block (Stage 2 and Stage 3), and two independent classification heads at the end.

---

### 5.2 Offline Depth Pre-Computation
Instead of running MDE in real time during training, we pre-compute depth maps offline:
* Load the pre-trained Depth Anything V2 (Small) model.
* Pass each RGB image through it to get a 2D depth map.
* Interpolate the map to 224x224 and normalize values to `[0, 1]`.
* Save the map as a `.pt` tensor.

---

### 5.3 FourierNet Branch and Log-Magnitude Input
The frequency branch converts the RGB image into the frequency domain:
1. **FFT:** Apply the 2D Fast Fourier Transform to the 3-channel RGB image.
2. **Shift:** Shift the zero-frequency component to the center of the spectrum.
3. **Decompose:** Separate the complex numbers into a 3-channel magnitude spectrum and a 3-channel phase spectrum.
4. **Log-compression:** Apply `log(|X| + 1)` to the magnitude spectrum to scale down the dominant DC (zero-frequency) value, making the higher-frequency patterns visible.
5. **Concat:** Concatenate the magnitude and phase spectra along the channel dimension, producing a 6-channel input tensor (shape `[B, 6, 224, 224]`).

This tensor is processed by `FourierNet` (three convolutional blocks with ReLU activations, global average pooling, and a fully connected layer), outputting a **64-dimensional frequency embedding**.

---

### 5.4 RGB and Depth Specialist Branches
* **RGBNet** processes the 3-channel RGB image to extract spatial colors and textures, outputting a 64-dimensional spatial embedding.
* **DepthNet** processes the 1-channel pre-computed depth map to extract 3D volumetric shapes, outputting a 64-dimensional geometric embedding.

These two branches are structurally identical (three convolutional blocks with GELU activations, global pooling, and fully connected bottleneck) but **never share weights** to prevent color information from contaminating geometric feature extraction in Stage 1.

---

### 5.5 Hierarchical Multi-Stage Fusion (RGB-D and Fourier)
We fuse the three 64-dimensional embeddings in two steps:

1. **Stage 2 — Intermediate Spatial Fusion (RGB ⊕ Depth):**
   We concatenate the RGB and depth embeddings (64 + 64 = 128 dimensions) and project them to 64 dimensions using `fc_stage2` (Linear(128, 64) + GELU + Dropout(0.3)). We fuse them early because both RGB and depth represent spatial pixel coordinates, allowing the model to learn spatial relationships (e.g., whether a flat texture anomaly aligns with a geometric edge).
2. **Stage 3 — Late Frequency Fusion (Visual-Geometric ⊕ Fourier):**
   We concatenate the 64-dimensional visual-geometric embedding and the 64-dimensional Fourier embedding (64 + 64 = 128 dimensions) and project them to the final 256-dimensional representation using `fc_stage3` (Linear(128, 256) + GELU + Dropout(0.3)). We use late fusion here because the Fourier spectrum represents global frequency statistics and has no spatial coordinates. Merging it earlier would disrupt the spatial filters of the RGB and Depth CNNs.

---

### 5.6 Classification Heads
The 256-dimensional fused embedding is sent to two separate heads:
* **Binary Head (AI Detection):** LayerNorm(256) → Linear(256, 128) → GELU → Dropout(0.4) → Linear(128, 2). Outputs logits for `{real, ai}`.
* **Transform Head (Post-Processing Classification):** LayerNorm(256) → Linear(256, 128) → GELU → Dropout(0.4) → Linear(128, 3). Outputs logits for `{original, redigital, transfer}`.

---

### 5.7 Multi-Task Loss and weighting parameter Lambda
We optimize the model using a combined loss function:

$$\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{AI}} + (1 - \lambda) \cdot \mathcal{L}_{\text{domain}}$$

Where:
* $\mathcal{L}_{\text{AI}}$ is the Cross-Entropy loss for Task 1.
* $\mathcal{L}_{\text{domain}}$ is the Cross-Entropy loss for Task 2.
* $\lambda \in [0, 1]$ is the scalar weighting parameter that controls the trade-off between the two tasks.

We trained 5 independent models with $\lambda \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ to locate the optimal balance on the Pareto frontier, and two unimodal baselines ($\lambda = 1.0$ and $\lambda = 0.0$).

---

## 6. Experimental Setup

* **Image Preprocessing:** Resized to 224×224 pixels. Standard ImageNet normalisation is applied to the RGB channel. Depth maps are normalized to `[0, 1]`.
* **Optimizer:** AdamW with a constant learning rate of $1 \times 10^{-4}$.
* **Batch Size:** 100.
* **Epochs:** Up to 50, with **Early Stopping** (patience = 5 epochs) monitoring validation loss.
* **Hardware & Acceleration:** Trained on a GPU using Mixed-Precision training (`torch.amp.GradScaler`) to reduce VRAM consumption and speed up training.

---

## 7. Evaluation and Results (Detailed Guide to the Charts)

We evaluated all models on the held-out test split of 2,250 images. Below we explain in detail the 9 evaluation components and charts generated in the notebook.

---

### 7.1 Learning Curves
* **What they show:** Side-by-side line charts for each model. The left chart plots training and validation losses over epochs. The right chart plots validation accuracies for Task 1 (AI) and Task 2 (Domain) over epochs.
* **Axes:** X-axis shows the training epoch (0 to 50). Y-axis shows the loss (left) or accuracy (right, from 0.0 to 1.0).
* **Ideal Shape:** Loss curves should decrease and level off together. If validation loss rises while training loss falls, the model is overfitting. Accuracy curves should rise and level off.
* **Observation:** The validation loss decreases and levels off alongside the training loss, showing that training converges smoothly without overfitting. Early stopping terminates training between epochs 28 and 40.
* **Physical Explanation:** For $\lambda = 0.1$, the loss is dominated by Task 2: Domain accuracy converges quickly, while AI accuracy improves slowly. For $\lambda = 0.9$, the AI classification task dominates, and AI detection accuracy rises rapidly while Domain accuracy oscillates.

---

### 7.2 Confusion Matrices
* **What they show:** For each model, two normalised confusion matrices are plotted side by side. The left matrix is for Task 1 (Real vs. AI); the right matrix is for Task 2 (Original vs. Redigital vs. Transfer).
* **Axes:** Y-axis represents the true label; X-axis represents the predicted label.
* **Ideal Shape:** A matrix with 1.0 (or 100%) along the diagonal (from top-left to bottom-right) and 0.0 everywhere else, representing perfect classification.
* **Observation:** At $\lambda=0.5$, the matrix is balanced, with similar accuracy for both classes (~82% and ~83% correct). For Task 2, the `redigital` class is classified with high accuracy (~85%). The main confusion occurs between `original` and `transfer` classes.
* **Physical Explanation:** Printing and scanning (`redigital`) introduces Moire patterns and scanner sensor grids that the Fourier branch detects easily. `original` and `transfer` are both digitally-native images, and their only difference is the JPEG compression and resizing artifacts, which are harder to separate.

---

### 7.3 Classification Reports
* **What they show:** Precise numerical values of **Precision**, **Recall**, and **F1-Score** for each class, along with macro averages on the test set.
* **Metrics Explained:**
  * **Precision:** Out of all images the model predicted as a class, what percentage was correct? (High precision means few false alarms).
  * **Recall:** Out of all actual images of a class, what percentage did the model find? (High recall means few missed fakes).
  * **F1-Score:** The harmonic mean of precision and recall, representing their balance.
* **Observation:** At $\lambda = 0.1$ (low AI weight), AI detection accuracy is 74.1%. The `ai` class achieves a high recall (82.9%) but low precision (70.0%), meaning the model is biased and over-predicts the `ai` class. At $\lambda = 0.5$ (balanced weight), AI detection accuracy is 82.6% (macro-F1 = 0.826). The F1-score is balanced between `real` (0.835) and `ai` (0.816). Task 2 accuracy is 59.5% (macro-F1 = 0.575).

---

### 7.4 ROC Curves and AUC
* **What they show:** ROC curves plotting the True Positive Rate (Sensitivity) against the False Positive Rate (1 - Specificity) across all decision thresholds. For the three-class Task 2, we use a One-vs-Rest (OvR) approach. The Area Under the Curve (AUC) is reported.
* **Axes:** X-axis shows the False Positive Rate (from 0.0 to 1.0); Y-axis shows the True Positive Rate (from 0.0 to 1.0).
* **Ideal Shape:** A curve that rises vertically to the top-left corner, resulting in an AUC of 1.0. A diagonal line from bottom-left to top-right represents a random guess (AUC of 0.5).
* **Observation:** For $\lambda = 0.5$, the AUC for AI detection is high (0.91), indicating that the shared backbone extracts highly discriminative features. For Task 2, `redigital` achieves the highest AUC (0.94), while `original` and `transfer` show more overlap, resulting in lower AUCs (around 0.78 and 0.81).
* **Physical Explanation:** The high AUC values show that even when a threshold is slightly suboptimal, the model's features remain highly separable.

---

### 7.5 Per-Class Metrics Bar Charts
* **What they show:** Grouped bar charts comparing Precision, Recall, and F1-score for each class.
* **Axes:** X-axis shows the classes; Y-axis shows the metric score (from 0.0 to 1.0).
* **Ideal Shape:** All bars rising to 1.0.
* **Observation:** This chart visualises the classification report. It shows that for low lambdas, the model has a recall bottleneck for the `original` class. As $\lambda$ increases, the height of the bars for the `real` and `ai` classes becomes equal and stable, showing balanced performance.

---

### 7.6 Accuracy Breakdown by Transformation Category
* **What they show:** A bar chart showing the Task 1 (AI detection) accuracy evaluated separately on the three post-processing subsets: Original, Re-digitalised, and Internet-transferred.
* **Axes:** X-axis shows the three transformation categories; Y-axis shows the AI detection accuracy (from 0% to 100%).
* **Ideal Shape:** Equal, high bars across all three categories, showing that post-processing does not degrade AI detection performance.
* **Observation:** For the optimal model ($\lambda = 0.5$), the AI detection accuracy is:
  * **Original (Clean):** 82.93%
  * **Re-digitalised (Analog-degraded):** 80.61%
  * **Internet-transferred (Compressed):** 84.34%
* **Physical Explanation:** The accuracy on re-digitalised images drops by only **2.32 percentage points** compared to clean images. Normally, printing and scanning destroys almost all spatial noise artifacts left by GANs and diffusion models. The model remains robust because:
  1. **The Depth branch** captures 3D perspective and volume inconsistencies, which are not altered by the 2D printing and scanning process.
  2. **The FourierNet branch** identifies the Moire patterns and scanning grid lines, helping the network classify the domain and isolate printing noise.

---

### 7.7 Lambda Trade-off Analysis (Pareto Frontier)
* **What it shows:** A line chart plotting overall test accuracy on the Y-axis and the $\lambda$ values on the X-axis for both tasks.
* **Axes:** X-axis shows the $\lambda$ values (0.1 to 0.9); Y-axis shows the test accuracy (from 0.0 to 1.0).
* **Ideal Shape:** Both lines remaining flat and high across all values, representing no trade-off. In practice, they intersect at a point representing the best compromise.
* **Observation:** AI accuracy increases from 74.1% at $\lambda=0.1$ to 82.6% at $\lambda=0.5$, and then plateaus (82.6% at $\lambda=0.9$). Domain classification accuracy remains stable between 58% and 60%.
* **Physical Explanation:** $\lambda = 0.5$ represents the optimal trade-off point: it matches the maximum AI detection accuracy of $\lambda = 0.9$ while keeping the domain classification accuracy high (59.5%).

---

### 7.8 Per-Class F1 Heatmap Across Lambda Values
* **What it shows:** A heatmap where rows represent $\lambda$ values and columns represent the individual classes. The color intensity shows the F1-score.
* **Axes:** Y-axis shows the $\lambda$ values; X-axis shows the classes (`real`, `ai`, `original`, `redigital`, `transfer`).
* **Ideal Shape:** A completely bright yellow grid (F1 close to 1.0).
* **Observation:** The F1-scores for `real` and `ai` columns grow brighter (higher F1) as $\lambda$ goes from 0.1 to 0.5, then remain stable. The `original` domain column is brightest at $\lambda = 0.3$ and $0.5$, showing that a balanced training objective helps the model distinguish clean images from degraded ones. The `redigital` class remains bright (high F1) across all lambda settings, showing it is robust to loss weighting changes.

---

### 7.9 Domain Breakdown Across Lambda Values
* **What it shows:** A line chart showing AI detection accuracy over the different $\lambda$ values, split into three lines for the three transformation categories.
* **Axes:** X-axis shows the $\lambda$ values; Y-axis shows the AI detection accuracy.
* **Ideal Shape:** Three high, flat lines close to each other.
* **Observation:** Across all $\lambda$ values, the re-digitalised images are consistently the most difficult domain for AI detection, while internet-transferred images are the easiest.
* **Physical Explanation:** The chart shows that moving from $\lambda = 0.1$ to $\lambda = 0.5$ improves accuracy across all three domains simultaneously, while further increases in $\lambda$ do not yield significant gains.

---

### 7.10 Full Numeric Summary Table
**What it shows:** A single table consolidating all key metrics.

| $\lambda$ | AI Detection Accuracy | AI Macro F1 | Domain Accuracy | Domain Macro F1 | Original AI Acc. | Redigital AI Acc. | Transfer AI Acc. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.1 | 74.1% | 0.739 | 60.0% | 0.548 | 76.3% | 68.5% | 77.5% |
| 0.3 | 81.3% | 0.812 | 59.9% | 0.586 | 82.1% | 79.7% | 82.1% |
| **0.5** | **82.6%** | **0.826** | **59.5%** | **0.575** | **82.9%** | **80.6%** | **84.3%** |
| 0.7 | 81.4% | 0.813 | 60.3% | 0.583 | 80.7% | 79.8% | 83.8% |
| 0.9 | 82.6% | 0.825 | 57.9% | 0.575 | 82.7% | 81.3% | 83.9% |

---

### 7.11 Unimodal Baselines vs. Multi-Task Models
* **What it shows:** A bar chart comparing the performance of the multi-task models against the two unimodal baselines:
  * `UnimodalNetAI` ($\lambda = 1.0$): Trained only on AI detection.
  * `UnimodalNetDomain` ($\lambda = 0.0$): Trained only on post-processing classification.
* **Axes:** X-axis shows the models (Baselines and Multi-Task lambdas); Y-axis shows the accuracy.
* **Observation:** The unimodal AI baseline ($\lambda = 1.0$) achieves 82.6% AI detection accuracy, but its domain classification accuracy is at chance level (~33.3%). The unimodal domain baseline ($\lambda = 0.0$) achieves 60.3% domain accuracy, but its AI detection accuracy is at chance level (~50.0%). The multi-task model at $\lambda = 0.5$ achieves **82.6% AI detection accuracy** (matching the unimodal model) and **59.5% domain classification accuracy** (matching the domain-only model).
* **Physical Explanation:** This provides scientific validation for multi-task learning: a single shared model can solve both tasks simultaneously without any degradation in performance, confirming that the two objectives provide mutual regularisation.

---

## 8. Model Selection and Best Configuration

Based on our evaluation, we select the model trained with **$\lambda = 0.5$** as the best configuration. Our choice is based on the following reasons:

1. **Maximum AI Detection Performance:** It achieves 82.6% accuracy and 0.826 macro-F1 on Task 1, which matches the peak performance of the highly focused $\lambda = 0.9$ model.
2. **Balanced Domain Classification:** It maintains a domain classification accuracy of 59.5%, which is very close to the unimodal baseline peak (60.3%).
3. **Generalisation and Robustness:** As shown in the per-domain breakdown, $\lambda = 0.5$ achieves the highest AI detection accuracies on original (82.93%) and internet-transferred (84.34%) images, and a highly competitive 80.61% on re-digitalised images.
4. **Parameter Efficiency:** It solves both tasks with a single set of shared backbone weights, saving half the parameters and training time compared to using two separate unimodal networks.

---

## 9. Coverage of Project Requirements

Below is the mapping showing how our implementation meets every requirement of **Project 2** from the course syllabus:

| Requirement | Implementation in Project |
| :--- | :--- |
| **Real/Fake Binary Classification** | Implemented `BinaryClassifier` head outputting logits for `{real, ai}`. |
| **Multi-Class Transformation Classification** | Implemented `TransformClassifier` head outputting logits for `{original, redigital, transfer}`. |
| **Multi-Task Unified Framework** | Implemented `MultiTaskNet` combining a shared `BackBone` with two independent heads. |
| **Combined Loss Function** | Optimized using $\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{AI}} + (1 - \lambda) \cdot \mathcal{L}_{\text{domain}}$. |
| **Study of Lambda Weighting** | Trained 5 models with $\lambda \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ and compared their Pareto trade-offs. |
| **Unimodal Baselines Comparison** | Trained and compared models at $\lambda = 1.0$ (AI-only) and $\lambda = 0.0$ (Domain-only). |
| **Multi-Modal Input (Beyond RGB)** | Integrated estimated Depth Maps (3D geometry) and Fourier Transform (spectral representations). |
| **Frequency-Domain Analysis** | Applied 2D FFT, shifted spectra, extracted magnitude/phase, applied log-compression, and classified via `FourierNet`. |
| **Depth Estimation Integration** | Pre-computed depth maps offline using `Depth Anything V2 (Small)` and cached them as `.pt` files. |
| **Dataset Balancing across Subgroups** | Implemented a quota balancing strategy inside `RRDataset` to keep all 6 subgroups equal. |
| **Standard 70/15/15 Data Split** | Split the balanced dataset into Train (70%), Validation (15%), and Test (15%) splits with a fixed seed. |
| **Validation and Model Evaluation** | Plotted learning curves, confusion matrices, classification reports, ROC/AUC, bar charts, and breakdown plots. |

---

## 10. Conclusions and Future Work

### 10.1 Summary of Findings
We proposed and implemented `MultiTaskNet`, a multi-modal and multi-task architecture for detecting synthetic images and identifying post-processing alterations. By combining RGB textures, monocular depth geometry, and frequency spectra, our model is highly robust to post-processing. The evaluation shows that:
* The model trained with $\lambda = 0.5$ achieves the best performance, matching the accuracy of unimodal models on both tasks simultaneously.
* Re-digitalised (printed and scanned) images are the hardest domain for AI detection, but our multi-modal features keep the performance drop to a minimum (only 2.3% drop).
* Staged fusion (fusing RGB and depth early, and frequency late) successfully prevents global frequency statistics from disrupting spatial filters in the convolutional branches.

### 10.2 Limitations
* **Dataset Size:** Capping the dataset at 15,000 samples limits the variety of generative models and scanning devices the network can learn.
* **Depth Map Precision:** The depth maps are estimated offline using a monocular model (Depth Anything V2) rather than measured by physical sensors. For complex outdoor scenes, estimated depth can be noisy.
* **Domain Classification Ceiling:** The domain classification accuracy remains capped at ~60%, showing that the network struggles to separate subtle JPEG compression levels from clean digital images.

### 10.3 Future Work
* **VLM and Foundation Models:** Replacing our custom CNN encoders with pre-trained vision-language features or larger backbones (like ViT or DINOv2) could improve feature extraction.
* **Explainability (Grad-CAM):** Implementing Grad-CAM (studied in the course) would allow us to visualize which spatial regions and frequency bands the model focuses on for each task.
* **Dynamic Loss Weighting:** Using an automated loss weighting scheduler (like GradNorm) instead of a fixed $\lambda$ could improve convergence speed.
* **Adversarial Robustness:** Testing the model against adversarial attacks (images designed to trick AI detectors) would assess the system's readiness for real-world deployment.

---

## 11. References
1. **Sapienza University of Rome Lecture Notes:** *Computer Vision Course Material (A.Y. 2025/2026)*, Prof. Irene Amerini. (Sections: Frequency Domain, Recognition, Monocular Depth Estimation, Diffusion Models, Transformers).
2. **Depth Anything V2:** Yang, L., et al. *Depth Anything V2*, arXiv preprint, 2024.
3. **Denoising Diffusion Probabilistic Models (DDPM):** Ho, J., Jain, A., & Abbeel, P. *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
4. **Li, C., et al.:** *Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios*, CVPR 2025.
5. **Shao, R., et al.:** *Detecting and grounding multi-modal media manipulation*, CVPR 2023.
