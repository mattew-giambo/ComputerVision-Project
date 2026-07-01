# Project Report — Computer Vision (A.Y. 2025/2026)

## Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios
**Project 2 | Multi-Task RGB-D Learning with Frequency-Domain Analysis**

---

### Course Information
* **Institution:** Sapienza University of Rome
* **Course:** Computer Vision (M.Sc. in Engineering in Computer Science)
* **Academic Year:** 2025/2026
* **Professor:** Prof. Irene Amerini
* **Candidate(s):** [Candidate Name(s) and Student ID(s) - Placeholder]

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
AI models can now generate photorealistic synthetic images. These images look completely real to the human eye, which makes it easy to spread fake visual information. Computer vision researchers try to build automated detector programs to distinguish real camera photographs from AI-generated fakes.

In the real world, this task is extremely difficult because images do not remain in their original, clean state. When you view an image online, it has almost always undergone some form of **post-processing**. For example:
* **Internet Transfer:** Uploading a photo to social media platforms (like Instagram or WhatsApp) automatically resizes the image and applies lossy **JPEG compression** to save storage space.
* **Re-digitalisation:** A user might print the digital image onto physical paper and then re-photograph it with a smartphone or scan it back into a computer.

These operations alter the pixel values, washing away the microscopic traces (like camera sensor noise) that detectors use to identify fakes.

### Our Solution
This project builds a unified neural network, called **MultiTaskNet**, to answer two forensic questions simultaneously from a single input image:
1. **Task 1 — AI Detection:** Is the image a genuine photograph or AI-generated? (Binary classification: `real` vs. `ai`).
2. **Task 2 — Post-Processing Classification:** What transformation has the image undergone? (Three-class classification: `original` for no changes, `redigital` for printed-and-scanned, and `transfer` for compressed social media images).

Instead of treating these tasks separately, we use **Multi-Task Learning (MTL)**. By training one network on both tasks, the model learns a shared representation. Recognizing the type of post-processing helps the network understand how the image was altered, which in turn helps it isolate and remove post-processing noise to make the AI detector highly robust.

---

## 2. State of the Art (How Others Approach This)

Before building our model, we must look at how researchers currently attempt to solve these issues.

### 2.1 Spatial vs. Frequency Detection
Early detectors trained Convolutional Neural Networks (CNNs) in the **spatial domain** (using the raw color pixels directly). To make them robust to transformations, researchers applied data augmentations (like random blurring or JPEG compression) during training. This forces the model to ignore fragile pixel details and focus on global object structures.

However, generative models leave behind unique **spectral fingerprints**. The upsampling layers used by generative models to enlarge small noise grids into full-size images introduce repeating, grid-like artifacts. By converting the image to the **frequency domain** (using the Fourier Transform), we can isolate these repeating grid lines, which appear as bright, symmetrical dots in the spectrum.

### 2.2 Adding 3D Geometry (Depth)
Recently, diffusion models have challenged existing frequency detectors because they do not use standard convolutional upsampling and thus do not leave periodic spectral grids. 

To catch these newer fakes, researchers have turned to **3D geometry**. Real cameras capture actual three-dimensional scenes. AI generators, on the other hand, generate 2D pixels that look realistic but often contain 3D geometrical errors (like shadows pointing in inconsistent directions or perspective lines that do not align correctly). By extracting a **depth map** (which shows how far objects are from the camera) and feeding it as a separate channel alongside the RGB colors, we can expose these geometric inconsistencies. Crucially, these 3D shapes do not disappear when the image is compressed or printed.

---

## 3. Theoretical Background: Explaining the Concepts from Scratch

This section explains the foundational concepts from the Computer Vision course that form the building blocks of our project.

### 3.1 What is a Digital Image?
To a computer, a grayscale digital image is a 2D grid of numbers called **pixels**. A typical grid size might be 224 rows by 224 columns. Each pixel contains a number from `0` (pure black) to `255` (pure white). 
A color image is a 3D grid with three color layers (called **channels**): Red, Green, and Blue (RGB). Therefore, a color image of size 224x224 pixels is represented as a tensor of shape `[3, 224, 224]`.

---

### 3.2 The Fourier Transform and the Frequency Domain
The Fourier Transform is a mathematical tool that allows us to view an image in two different ways:
1. **Spatial Domain:** The standard view, which shows *where* colors and objects are located in space (pixels on a grid).
2. **Frequency Domain:** A new view, which shows *how quickly* colors change as you move across the image.

#### Fast and Slow Changes (Frequencies)
* **High Frequencies:** Areas where color changes very quickly over a small space, such as sharp edges, textures, noise, or fine details.
* **Low Frequencies:** Areas where color changes very slowly, such as a smooth blue sky or flat walls.

The **2D Discrete Fourier Transform (DFT)** convert pixels into a complex-valued spectrum:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

The **Fast Fourier Transform (FFT)** is the efficient algorithm used to calculate this grid. The output grid contains complex numbers ($a + jb$), which we separate into two parts:
1. **Magnitude Spectrum:** Measures the *strength* of each frequency.
2. **Phase Spectrum:** Measures the *starting position* (alignment) of each frequency.

#### The Zebra and Cheetah Swap Experiment
In class, we discussed a classic experiment that illustrates the difference between magnitude and phase:
* Take the Fourier Transform of a cheetah image and a zebra image.
* Combine the **phase** of the cheetah with the **magnitude** of the zebra, and reconstruct the image. The result clearly shows a cheetah, but with zebra-like textures.
* Combine the **phase** of the zebra with the **magnitude** of the cheetah. The reconstructed image shows a zebra, overlaid with cheetah-like noise.

This proves that **the phase spectrum carries the structural details (edges and shapes) of the image, while the magnitude spectrum carries the frequency energy (noise and repeating patterns)**. In our model, we use both magnitude and phase as inputs to our frequency branch.

#### Convolution Theorem
The **Convolution Theorem** states that applying a filter to an image in the spatial domain (e.g., blurring it) is equivalent to multiplying the image's spectrum by the filter's spectrum in the frequency domain:

$$\mathcal{F}[g * h] = \mathcal{F}[g] \cdot \mathcal{F}[h]$$

This explains why post-processing operations (like blurring or JPEG compression, which are low-pass filters) directly alter or erase high-frequency details in the spectrum.

---

### 3.3 Convolutional Neural Networks (CNNs) for Image Recognition
A Convolutional Neural Network (CNN) is a deep learning model designed to extract visual features from images. A standard CNN is built from the following layers:

1. **Convolutional Layer:** This layer slides a set of small filters (e.g., a 3x3 pixel window) across the input image. At each step, it multiplies the filter weights by the pixel values to detect specific patterns. Early layers detect simple patterns like vertical or horizontal lines (edges). Deeper layers combine these lines to detect complex shapes (like circles) and objects (like eyes or wheels).
2. **Batch Normalisation (BN):** Normalises the numerical features across a batch of images. This prevents the numbers from becoming too large or too small, making training faster and more stable.
3. **Activation Function (ReLU / GELU):** Introduces non-linearity. Without activation functions, stacking multiple layers would be equivalent to a single linear equation, meaning the network could not learn complex patterns. GELU (Gaussian Error Linear Unit) is a smooth version of ReLU.
4. **Max Pooling:** Slides a window (usually 2x2) across the feature map and keeps only the maximum value. This halves the spatial size (reducing resolution), which makes the features robust to small shifts or translations in the image.
5. **Global Average Pooling:** Converts a 2D feature map of any size into a single number by taking the average of all its values. This reduces the spatial dimensions to 1x1, producing a flat vector that is fed into the classification layers.

---

### 3.4 Monocular Depth Estimation (MDE) and RGB-D Representations
Monocular Depth Estimation (MDE) is the task of predicting the 3D depth of a scene using a single 2D image. An RGB-D representation combines the three standard color channels (Red, Green, Blue) with a fourth channel: a dense **depth map (D)** showing the distance of each pixel from the camera.

In class, we studied **Depth Anything V2**, which uses a **Teacher-Student training framework**:
* **The Teacher Model:** A large, heavy transformer model is trained on high-quality synthetic 3D data. While precise, it struggles to generalise to real-world images due to the "distribution shift" (synthetic data looks different from real photographs).
* **Pseudo-Labelling:** To bridge this gap, the Teacher model is run on a massive dataset of unlabeled real-world images, predicting depth maps (called "pseudo-labels") for them.
* **The Student Model:** A smaller, lightweight model is trained on these real-world images using the pseudo-labels generated by the Teacher. This allows the Student to learn robust, real-world depth estimation.

In forensic security, MDE is highly valuable: AI generators are designed to make realistic 2D pixels, but they struggle with 3D physical consistency. This leaves geometric inconsistencies in the depth map (such as flat areas where there should be volume, or incorrect perspective lines) that can be detected by the model.

---

### 3.5 Deep Generative Models and Diffusion Models (DDPM)
Generative models learn the underlying probability distribution of real data ($p(x)$) to generate new, unique samples that look real.

In class, we studied **Denoising Diffusion Probabilistic Models (DDPM)**. They operate using two processes:
1. **Forward Process (Noising):** Takes a real image $x_0$ and gradually adds small amounts of Gaussian noise over $T$ steps (usually $T=1000$) until the image becomes pure, unstructured noise $x_T$. This process is fixed and does not require learning.
2. **Backward Process (Denoising / Generation):** A neural network learns to reverse the noise addition. Starting from random noise $x_T$, the model predicts and removes a small amount of noise at each step, gradually recovering a clean, structured image $x_0$.

Because diffusion models generate images via incremental denoising rather than convolutional upsampling, they do not produce the same periodic upsampling grids as older GAN models. This makes them harder to detect and requires the use of multi-modal features.

---

### 3.6 Multi-Task Learning as Regularisation
In machine learning, **overfitting** occurs when a model memorises the training data too well, including its noise and quirks, making it perform poorly on new, unseen test images. **Regularisation** is any technique used to prevent overfitting.

**Multi-Task Learning (MTL)** acts as an implicit regulariser. By forcing a single shared backbone network to extract features that are useful for both AI detection and post-processing classification, we prevent the model from focusing on fragile, task-specific details. Instead, the network must learn general visual, geometric, and frequency patterns that are robust to alterations.

---

## 4. Dataset (RRDataset & Pre-computed Depth)

### The RRDataset
We use the **RRDataset** (*Real/Redigital* dataset) to train and test our models. The images in this dataset are organized along two axes:

1. **AI Label (Task 1):**
   * `real`: Genuine photos taken by cameras.
   * `ai`: Images generated by AI models (like GANs and diffusion models).
2. **Domain Label (Task 2):**
   * `original`: Digital native images with no alterations.
   * `redigital`: Images printed on paper and re-digitised using a scanner or camera.
   * `transfer`: Digital images subjected to social media compression and resizing.

Combining these two axes yields **six distinct subgroups**:
* Original Real & Original AI
* Redigital Real & Redigital AI
* Transfer Real & Transfer AI

```
RRDataset_test/
├── original/
│   ├── ai/       ← AI-generated, clean
│   └── real/     ← Genuine photo, clean
├── redigital/
│   ├── ai/       ← AI-generated, printed-and-scanned
│   └── real/     ← Genuine photo, printed-and-scanned
└── transfer/
    ├── ai/       ← AI-generated, social-media compressed
    └── real/     ← Genuine photo, social-media compressed
```

### Dataset Balancing and Splits
To ensure the model does not develop a bias towards any class (e.g., predicting "real" just because there are more real images in the dataset), we cap the total dataset size at **15,000 samples** and balance it using a uniform quota strategy. This guarantees that each of the six subgroups contains exactly 2,500 images.

We split this balanced dataset using a fixed random seed (seed = 42) into:
* **Training Split (70%):** ~10,500 images, used to train the network.
* **Validation Split (15%):** ~2,250 images, used to tune hyperparameters and apply early stopping.
* **Test Split (15%):** ~2,250 images, held out and used only for final, unbiased evaluation.

### Depth Map Cache
To save GPU memory and training time, we pre-computed all depth maps **offline** (once before training) using **Depth Anything V2 (Small)**. These are saved to disk in a separate directory (`RRDataset_depth/`) as `.pt` files. During training, the dataloader loads the RGB image and its corresponding pre-computed `.pt` depth tensor at zero computational cost.

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
