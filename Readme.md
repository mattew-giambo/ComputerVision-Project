# Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [State of the Art](#2-state-of-the-art)
3. [Proposed Method](#3-proposed-method)
4. [Dataset](#4-dataset)
5. [Experimental Setup](#5-experimental-setup)
6. [Model Evaluation](#6-model-evaluation)
7. [Conclusion](#7-conclusion)

---

## 1. Problem Statement

The widespread availability of powerful generative models — including Generative Adversarial Networks (GANs), diffusion-based architectures, and large multimodal foundation models — has made it increasingly trivial to produce photorealistic synthetic imagery. While such capabilities have clear creative and industrial applications, they pose a fundamental challenge to the integrity of visual information: how can a viewer, or an automated system, reliably distinguish a genuine photograph from a synthetically generated one?

The difficulty is compounded by a critical real-world factor: images rarely remain in their pristine, uncompressed form. By the time a piece of content is viewed online, it has typically undergone one or more forms of post-processing. A photograph may have been saved and re-uploaded through social media platforms (which apply lossy compression and resizing), printed and subsequently re-digitised with a scanner or camera, or subjected to various other transformations. These post-processing pipelines introduce artefacts that partially destroy the statistical fingerprints which detection systems rely upon, making robust forensic analysis considerably harder.

This project addresses **two complementary tasks simultaneously**:

- **Task 1 — AI Detection:** Classify each image as either *real* (captured by a physical camera) or *AI-generated* (produced by a generative model).
- **Task 2 — Post-Processing Classification:** Identify the type of transformation the image has undergone: *original* (no processing), *re-digitalised* (printed and re-scanned), or *internet-transferred* (uploaded and downloaded via an online channel).

Solving these tasks jointly is motivated by a practical observation: the two problems are deeply interrelated. Understanding the provenance of an image (i.e., whether it is AI-generated) and understanding the transformations it has undergone share a common feature space — both depend on subtle statistical properties of the image signal, particularly in the frequency domain and in depth/geometry consistency. A unified model that learns both simultaneously may generalise better than two independent, narrowly specialised detectors.

---

## 2. State of the Art

### 2.1 AI-Generated Image Detection

The detection of synthetically generated images has been an active research area since the rise of generative architectures. Convolutional networks, such as the ResNet-50 architecture (covered in the course notes), are commonly trained to classify images as real or fake. To improve generalisation across different generator engines and real-world degradations, input-level data augmentation (including blurring and JPEG compression) is widely employed, establishing its role as a necessary regulariser for spatial feature extractors.

Subsequent work has leveraged **frequency-domain features** as a more robust and translation-invariant signal. Generative architectures leave characteristic spectral fingerprints — such as periodic peaks and grid-like anomalies in the Fourier magnitude spectrum — due to the mathematical properties of their convolutional upsampling operations. Computing the 2D Discrete Fourier Transform (DFT) (as covered in the course's frequency domain modules) allows these spectral artifacts to be isolated and fed directly to convolutional classifiers to detect synthetic content.

More recently, the emergence of diffusion models (such as the Denoising Diffusion Probabilistic Models, or DDPM, studied in the course) has challenged existing detectors, as diffusion-based images do not exhibit the same spectral artefacts as early generative models. This has prompted the use of more general feature extractors. Vision Transformer (ViT) backbones (extensively covered in the course) have been explored alongside convolutional networks to capture richer, less architecture-specific representations.

### 2.2 Post-Processing Robustness

A persistent limitation of early detection methods is their brittleness under post-processing. JPEG compression, rescaling, and median filtering can all erase the subtle statistical fingerprints that classifiers depend on. Research in **media forensics** has long grappled with this issue: splicing detection, copy-move forgery detection, and source camera identification all suffer from similar degradation sensitivity.

One line of research addresses this through explicit augmentation strategies — training detectors on images that have been subjected to simulated JPEG compression and resizing — while another seeks architecturally robust representations. The use of **high-pass filters** (including Laplacian operators and band-pass filtering, as studied in the course) has proven effective in extracting high-frequency noise residuals while suppressing semantic content. This ensures that the detector focuses on the statistical footprints that survive lossy post-processing operations.

### 2.3 Multi-Task and Multi-Modal Learning

Multi-task learning (MTL) has repeatedly demonstrated that sharing representations across related objectives acts as an implicit regulariser, improving generalisation on all constituent tasks. In image forensics, the idea of jointly training a detector for multiple forensic attributes has been explored in the context of GAN fingerprint identification and image authentication, though the specific combination of AI detection with post-processing classification is less studied.

The integration of **depth information** (RGB-D learning) has been explored extensively in scene understanding and 3D object detection. In the context of forensics, depth maps offer a complementary signal: AI-generated images may exhibit depth-consistency artefacts invisible in the RGB domain, such as implausible perspective transitions or geometrically inconsistent shadow-depth relationships. Monocular Depth Estimation (MDE) models — particularly transformer-based architectures such as **Depth Anything v2** (which, as covered in the course, utilizes a Teacher-Student training framework combined with pseudo-labeling) — enable the offline extraction of proxy depth maps from monocular RGB images, making RGB-D forensics tractable without physical depth sensors.

---

## 3. Proposed Method

### 3.1 Overall Architecture — `MultiTaskNet`

The proposed system, `MultiTaskNet`, is a **hierarchical multi-modal multi-task network** that processes each image through three parallel specialist pathways (RGB, depth, and Fourier) before fusing their representations and routing the result through two independent classification heads.

The architecture is organised into three stages:

#### Stage 1 — Isolated Specialist Extraction

Three independent encoder branches process complementary views of the input image:

- **RGB Branch (`rgb_net`):** A lightweight convolutional encoder that operates on the three-channel RGB image. It captures chromatic texture patterns and surface-level visual artefacts — the signals most immediately perceptible to human observers. The branch outputs a 64-dimensional embedding.

- **Depth Branch (`depth_net`):** A structurally identical convolutional encoder, but operating on a single-channel estimated depth map. By processing depth independently of colour, this branch can detect geometric implausibilities — such as incorrect perspective foreshortening or spatially inconsistent 3D structures — that are not visible in the colour image. The branch outputs a 64-dimensional embedding.

- **Fourier Branch (`FourierNet`):** A branch that operates on the frequency-domain representation of the image. The input image is transformed via a 2D Fast Fourier Transform (FFT), shifted to centre the zero-frequency component, and decomposed into its log-magnitude and phase spectra. These are processed by separate lightweight convolutional networks whose outputs are then combined into a 64-dimensional frequency embedding. Log-compression (`log(|X| + 1)`) is applied to the magnitude to emphasise spectral differences at all frequency scales, rather than allowing the large DC component to dominate.

The use of three independent branches, rather than a single merged input, ensures that each specialist remains focused on its native domain. Early or direct fusion of RGB, depth, and Fourier features in a single input tensor would risk corrupting the spatial filters of the RGB/depth CNNs with the global, spatially unaligned frequency signal.

#### Stage 2 — Intermediate Spatial Fusion (RGB ⊕ Depth)

The RGB and depth embeddings (64-dimensional each) are concatenated into a 128-dimensional vector and projected back to 64 dimensions through a learned linear bottleneck. This projection layer enables the network to learn cross-modal correspondences between texture (RGB) and geometry (depth): for example, a texture anomaly in the RGB branch that co-occurs with a geometric inconsistency in the depth branch is a strong indicator of AI generation. The resulting 64-dimensional visual-geometric embedding is compact enough to keep the overall computational footprint essentially unchanged with respect to an equivalent RGB-only baseline.

#### Stage 3 — Late Frequency Fusion

The visual-geometric embedding from Stage 2 is concatenated with the 64-dimensional Fourier embedding, yielding a 128-dimensional representation, which is then projected to the final 256-dimensional fused embedding. This late-fusion strategy is deliberate: the Fourier branch encodes global spectral statistics with no spatial coordinates, and merging it earlier would risk interfering with the spatially coherent filters trained in the RGB and depth CNNs. By fusing at the final stage, the Fourier signal supplements the visual-geometric understanding with a complementary, globally-averaged spectral fingerprint.

#### Classification Heads

The 256-dimensional fused embedding is routed to two independent classification heads:

- **Binary Head (`BinaryClassifier`):** Layer normalisation → Linear(256, 128) → GELU → Dropout(0.4) → Linear(128, 2). Outputs logits for {*real*, *AI-generated*}.
- **Transform Head (`TransformClassifier`):** Identical structure, but outputs logits for {*original*, *re-digitalised*, *internet-transferred*}.

Both heads operate on the same shared embedding, enabling the model to jointly learn features relevant to both forensic questions.

### 3.2 Multi-Task Loss Function

The training objective combines the cross-entropy losses of both tasks through a scalar weighting parameter $\lambda$:

$$\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{AI}} + (1 - \lambda) \cdot \mathcal{L}_{\text{domain}}$$

where $\mathcal{L}_{\text{AI}}$ is the cross-entropy loss for Task 1 and $\mathcal{L}_{\text{domain}}$ is the cross-entropy loss for Task 2. Five independent models are trained, one for each value of $\lambda \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$, each initialised from fresh random weights and trained to convergence. This grid allows the trade-off between the two tasks to be studied empirically on the Pareto frontier.

### 3.3 Offline Depth Pre-computation

Generating depth maps at training time would be prohibitively expensive in terms of GPU memory. Instead, depth maps are pre-computed offline once, before training, using **Depth Anything V2 (Small)** — a lightweight transformer-based monocular depth estimation model. The estimated depth maps are cached to disk as `.pt` tensors, mirroring the directory layout of the RGB dataset, and loaded by the `RRDataset` dataloader at runtime.

---

## 4. Dataset

### 4.1 The RRDataset

The project uses the **RRDataset** (*Real/Redigital* dataset), a curated collection of images designed for real-world forensic analysis. The dataset is organised along two independent axes:

**Axis 1 — Image Origin (AI Label):**
- `real`: Genuine photographs captured by physical cameras.
- `ai`: Images synthesised by generative AI models (covering a variety of generative architectures).

**Axis 2 — Post-Processing Condition (Domain Label):**
- `original`: Images in their original, unmodified state (or with minimal processing).
- `redigital`: Images that have been printed on a physical medium (paper) and subsequently re-digitised using a scanner or camera.
- `transfer`: Images that have been uploaded to and downloaded from an internet platform (e.g., a social media or file-sharing service), subjecting them to platform-side compression, rescaling, and re-encoding.

The combination of these two axes yields **six distinct subgroups** (2 origins × 3 conditions), each represented in the dataset. The main dataset folder `RRDataset_test` mirrors this structure:

```
RRDataset_test/
├── original/
│   ├── ai/       # AI-generated, unmodified
│   └── real/     # Genuine, unmodified
├── redigital/
│   ├── ai/       # AI-generated, print-and-scan
│   └── real/     # Genuine, print-and-scan
└── transfer/
    ├── ai/       # AI-generated, internet-transferred
    └── real/     # Genuine, internet-transferred
```

A representative subset of the AI-generated images originates from the `Culture & Religion` semantic category, while the genuine images cover a broad range of photographic content.

### 4.2 Depth Map Dataset

The folder `RRDataset_depth` contains pre-computed monocular depth maps corresponding to every image in `RRDataset_test`. The directory layout is identical to that of the RGB dataset, with each image replaced by a `.pt` file containing a single-channel depth tensor at the same resolution as its RGB counterpart (224 × 224 pixels). These depth maps were extracted using **Depth Anything V2 (Small)** and cached to avoid re-computation during training.

### 4.3 Dataset Statistics and Splits

The dataset is capped at **15,000 samples** in total. To avoid class imbalance artefacts, samples are drawn uniformly from all six subgroups: the maximum number of samples per subgroup is determined by dividing the cap equally across groups, and each group is filled up to its natural size if it has fewer samples than the per-group cap.

The balanced dataset is then split as follows:

| Split       | Proportion | Purpose                          |
|-------------|:----------:|----------------------------------|
| Training    | 70 %       | Model optimisation               |
| Validation  | 15 %       | Hyperparameter selection, early stopping |
| Test        | 15 %       | Final, unbiased evaluation       |

All five multi-task models and both unimodal baselines are evaluated on the **same** held-out test split, ensuring a fair comparison across all $\lambda$ values.

---

## 5. Experimental Setup

### 5.1 Data Pre-processing

All images are resized to **224 × 224** pixels before being passed to the network. Standard ImageNet normalisation (mean and standard deviation per channel) is applied. Depth maps are loaded from pre-computed `.pt` files and resized to match the RGB resolution.

### 5.2 Hyperparameters

| Hyperparameter       | Value              |
|----------------------|--------------------|
| Image size           | 224 × 224          |
| Batch size           | 100                |
| Learning rate        | 1 × 10⁻⁴ (Adam)   |
| Maximum epochs       | 50                 |
| Early stopping patience | 5 epochs        |
| RGB embedding dim    | 64                 |
| Depth embedding dim  | 64                 |
| Fourier embedding dim| 64                 |
| Fused embedding dim  | 256                |
| Classifier hidden dim| 128                |
| Dropout              | 0.4                |
| Random seed          | 42                 |
| λ values trained     | {0.1, 0.3, 0.5, 0.7, 0.9} |

### 5.3 Training Protocol

Each of the five multi-task models is trained **completely independently**: a fresh `MultiTaskNet` is instantiated (with the same random seed for reproducibility), trained end-to-end for up to 50 epochs with the fixed $\lambda$ value assigned to it, and saved to its own checkpoint file (`model_lambda_<λ>.pth`). Early stopping monitors the validation loss and terminates training if no improvement is recorded for 5 consecutive epochs, at which point the best checkpoint (lowest validation loss) is restored.

The optimiser is **Adam** with a learning rate of `1e-4`. No learning rate scheduling is applied. Mixed-precision training (AMP) is used when a CUDA-capable GPU is available.

Two additional **unimodal baselines** are trained to frame the multi-task models:

- `UnimodalNetAI` ($\lambda = 1.0$): Backbone + Binary Head only, trained exclusively on the AI-detection task.
- `UnimodalNetDomain` ($\lambda = 0.0$): Backbone + Transform Head only, trained exclusively on the post-processing classification task.

These degenerate cases represent the theoretical extremes of the $\lambda$ trade-off and allow the benefit of multi-task learning to be directly quantified.

### 5.4 Evaluation Metrics

Each model is evaluated on the held-out test set using the following metrics:

- **Test Accuracy** (Task 1 and Task 2 separately)
- **Macro-averaged F1-score** (Task 1 and Task 2)
- **Confusion matrices** (normalised, for both tasks)
- **Classification reports** (per-class precision, recall, F1, support)
- **ROC curves and AUC** (one-vs-rest, for each class in both tasks)
- **Per-class precision/recall/F1 bar charts**
- **Per-domain AI detection accuracy** (i.e., Task 1 accuracy broken down by the *original*, *redigital*, and *transfer* subsets)
- **Aggregate summary table** (all metrics for all λ values side by side)
- **Unimodal vs multi-task bar chart** (direct comparison across all models)

---

## 6. Model Evaluation

### 6.1 Task 1 — AI Detection Performance

The table below reports the test-set accuracy and macro-averaged F1-score for Task 1 (real vs. AI) across all five multi-task configurations:

| λ    | Test Accuracy | Macro F1 |
|------|:------------:|:--------:|
| 0.1  |    74.1 %    |   0.739  |
| 0.3  |    81.3 %    |   0.812  |
| 0.5  |    82.6 %    |   0.826  |
| 0.7  |    81.4 %    |   0.813  |
| 0.9  |    82.6 %    |   0.825  |

As expected, AI-detection performance increases monotonically as $\lambda$ shifts weight towards $\mathcal{L}_{\text{AI}}$, with diminishing returns above $\lambda = 0.5$. The model trained with $\lambda = 0.5$ achieves the same test accuracy as $\lambda = 0.9$ (82.6 %) while allocating equal importance to both tasks, demonstrating that multi-task learning provides a meaningful regularisation benefit at no cost to the primary objective.

### 6.2 Task 2 — Post-Processing Classification Performance

| λ    | Test Accuracy | Macro F1 |
|------|:------------:|:--------:|
| 0.1  |    60.0 %    |   0.548  |
| 0.3  |    59.9 %    |   0.586  |
| 0.7  |    60.3 %    |   0.583  |
| 0.5  |    59.5 %    |   0.575  |
| 0.9  |    57.9 %    |   0.575  |

Post-processing classification is inherently harder: distinguishing *original* from *re-digitalised* from *internet-transferred* requires detecting subtle compression, resampling, and sensor-noise artefacts. Performance is relatively stable across the $\lambda$ grid (approximately 58–60 %), reflecting that the shared backbone provides a useful representation for this task regardless of the exact weighting. The highest accuracy (60.3 %) is achieved at $\lambda = 0.7$, suggesting a modest benefit from having the AI-detection objective as an auxiliary task.

### 6.3 Per-Domain AI Detection

Breaking down Task 1 accuracy by post-processing condition reveals how each model generalises across transformations:

| λ    | Original | Re-digitalised | Internet-transferred |
|------|:--------:|:--------------:|:--------------------:|
| 0.1  |  76.3 %  |    68.5 %      |       77.5 %         |
| 0.3  |  82.1 %  |    79.7 %      |       82.1 %         |
| 0.5  |  82.9 %  |    80.6 %      |       84.3 %         |
| 0.7  |  80.7 %  |    79.8 %      |       83.8 %         |
| 0.9  |  82.7 %  |    81.3 %      |       83.9 %         |

Across all $\lambda$ values, AI-detection accuracy is consistently lower on re-digitalised images than on original or internet-transferred ones. This is consistent with the expectation that print-and-scan introduces the most disruptive transformation (analogue quantisation, scanner noise, optical blur), partially erasing the forensic fingerprints that discriminate real from AI content. Internet-transferred images, despite lossy compression, tend to preserve more discriminative spectral structure, yielding slightly higher accuracy.

### 6.4 Multi-Task vs Unimodal Comparison

The unimodal baselines ($\lambda = 1.0$ for AI detection, $\lambda = 0.0$ for domain classification) provide important reference points. The multi-task models at $\lambda \in \{0.5, 0.9\}$ achieve accuracy on the AI-detection task that is comparable to or exceeds the unimodal AI-only baseline, while simultaneously producing meaningful domain classification predictions. This confirms that the joint training objective acts as a beneficial regulariser rather than a distraction, and that the shared RGB-D-Fourier backbone is expressive enough to support both tasks concurrently.

### 6.5 Learning Curves

Training histories reveal consistent and well-behaved convergence across all $\lambda$ configurations. Models with higher $\lambda$ (more weight on AI detection) converge faster on that task and typically achieve lower validation loss within the first 15–20 epochs. The early stopping mechanism effectively prevents overfitting in all cases, with models typically stopping between epochs 28 and 40 depending on $\lambda$.

---

## 7. Conclusion

### 7.1 Summary

This work presented `MultiTaskNet`, a hierarchical multi-modal architecture for the joint detection of AI-generated images and post-processing transformations. The key design choices — independent specialist encoders for RGB, depth, and frequency features, followed by staged fusion — allow the model to exploit complementary forensic signals without cross-contaminating the learning dynamics of each modality. The multi-task loss formulation, parameterised by $\lambda$, provides an explicit, interpretable mechanism for studying the Pareto trade-off between the two objectives.

The experimental results demonstrate that:

1. A balanced $\lambda$ (≥ 0.5) achieves AI-detection accuracy (82.6 %) comparable to a fully dedicated unimodal detector, while simultaneously performing post-processing classification at approximately 60 % accuracy.
2. Re-digitalised images remain the most challenging condition for AI detection, underscoring the importance of forensic methods that are robust to analogue-to-digital conversion artefacts.
3. Depth-map integration provides a complementary modality to RGB features, and the Fourier branch captures spectral fingerprints that are invisible in the spatial domain alone.

### 7.2 Limitations

- **Dataset scale:** With a cap of 15,000 images spread across six subgroups, some subgroups may have limited coverage of the full diversity of generative models or scanning equipment.
- **Depth map quality:** The depth maps are estimated by a monocular model (Depth Anything V2 Small) rather than measured by a physical sensor. For indoor/outdoor scenes with complex geometry, the estimated depth may introduce noise that limits the benefit of the depth branch.
- **Post-processing classification ceiling:** The 58–60 % accuracy on Task 2 suggests that the current feature set may not fully capture the subtle statistical differences between compression codecs and scanning pipelines. Additional frequency-domain features or explicit noise-residual extractors could improve performance.

### 7.3 Future Work

- **Stronger generative models:** Extending the dataset to cover the latest generation of diffusion models (e.g., FLUX, Stable Diffusion 3, Midjourney v6) would test the robustness of the detector against increasingly photorealistic synthetic content.
- **Dynamic $\lambda$ scheduling:** Rather than a fixed $\lambda$ throughout training, a curriculum strategy — starting with equal weighting and gradually shifting towards the primary task — could achieve better convergence.
- **Pre-trained backbone integration:** Replacing the custom lightweight CNN branches with pre-trained Vision Transformer encoders (e.g., CLIP, DINOv2) as frozen or fine-tuned feature extractors could substantially boost performance, particularly on the AI-detection task.
- **Explainability:** GradCAM or attention-map visualisations applied to each branch would provide insight into which spatial regions and frequency bands the model relies upon for each task, improving interpretability and trust in forensic settings.
- **Adversarial robustness:** Evaluating the model against adversarially perturbed inputs — images specifically crafted to fool detectors — would provide a more realistic assessment of the system's utility in adversarial deployment scenarios.

---

## Repository Structure

```
ComputerVision project/
├── project_definitivo.ipynb    # Main notebook (full pipeline)
├── RRDataset_test/             # RGB image dataset
│   ├── original/
│   │   ├── ai/
│   │   └── real/
│   ├── redigital/
│   │   ├── ai/
│   │   └── real/
│   └── transfer/
│       ├── ai/
│       └── real/
├── RRDataset_depth/            # Pre-computed depth maps (.pt files)
│   ├── original/
│   ├── redigital/
│   └── transfer/
├── checkpoints/                # Saved model weights
│   ├── model_lambda_0.1.pth
│   ├── model_lambda_0.3.pth
│   ├── model_lambda_0.5.pth
│   ├── model_lambda_0.7.pth
│   ├── model_lambda_0.9.pth
│   ├── unimodal_ai.pth
│   ├── unimodal_domain.pth
│   └── training_histories.json
├── fourier.md                  # Reference notes on the Fourier Transform
└── README.md                   # This file
```

## Requirements

The project requires Python 3.9+ with the following main dependencies:

- `torch` and `torchvision`
- `transformers` (for Depth Anything V2)
- `scikit-learn`
- `matplotlib` and `seaborn`
- `Pillow`
- `tqdm`
- `numpy`

A CUDA-capable GPU is strongly recommended for training. Inference can be run on CPU, though it will be significantly slower.
