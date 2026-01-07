[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# NDVI-Guided Low-Rank Adaptation of Segment Anything for Agricultural Field Boundary Delineation from Sentinel-2 Imagery

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
  - [2.1 Datasets and Benchmarks for Agricultural Field Delineation](#21-datasets-and-benchmarks-for-agricultural-field-delineation)
  - [2.2 Classical Segmentation Methods and Transfer Learning](#22-classical-segmentation-methods-and-transfer-learning)
  - [2.3 Foundation Models for Remote-Sensing Segmentation](#23-foundation-models-for-remote-sensing-segmentation)
  - [2.4 Spectral Priors and NDVI for Boundary Delineation](#24-spectral-priors-and-ndvi-for-boundary-delineation)
  - [2.5 Efficiency, Data Cost, and Operational Scalability](#25-efficiency-data-cost-and-operational-scalability)

## 1. Introduction

Accurate delineation of agricultural field boundaries is foundational for crop monitoring, precision agriculture, and land resource management [1], [2]. Multispectral satellite imagery such as Sentinel-2 (10 m spatial resolution) provides globally consistent coverage at no per-scene cost, making it attractive for large-scale field mapping applications [3], [4]. However, the coarse spatial sampling of Sentinel-2 imagery, combined with the irregular and often narrow geometry of agricultural parcels—particularly in smallholder systems—leads to blurred or fragmented parcel edges. This significantly limits reliable boundary extraction from RGB composites alone [5], [6].

Classical and modern pixel-wise segmentation approaches (e.g., Random Forest, U-Net, DeepLab) achieve strong mask accuracy when dense labels and high-resolution inputs are available. Nevertheless, these methods typically require region-specific retraining and frequently fail to generalize across agro-ecological zones, crop types, and phenological stages [7], [8]. Large annotated benchmarks such as Fields of the World (FTW) provide essential cross-country testbeds for Sentinel-2 boundary segmentation, while AI4Boundaries uniquely pairs 1 m orthophotos with Sentinel-2 mosaics to quantify the fundamental resolution limits of 10 m imagery for parcel delineation [5], [9]. These dataset studies consistently highlight two core challenges:

- Many smallholder parcels fall below the minimum resolvable width for reliable 10 m boundary delineation.
- Models trained in one region often degrade substantially when transferred to other regions.

Foundation vision models—most notably the Segment Anything Model (SAM)—introduced a promptable segmentation paradigm capable of remarkable zero-shot generalization on natural images [10]. However, the direct application of SAM to remote sensing imagery is challenged by modality gaps related to spectral composition, spatial scale, and fine-grained boundary definitions that differ substantially from those in natural scenes [10], [11].

Recent efforts to adapt SAM for geospatial tasks have pursued complementary strategies. These include prompt diversification (e.g., GeoSAM), prompt optimization tailored to farmland geometry (fabSAM), frequency-domain priors for improved edge sensitivity (RSAM-Seg), and efficient parameter-efficient fine-tuning approaches such as Low-Rank Adaptation (LoRA) for aerial land-cover segmentation [12]–[14]. To date, however, most adaptations rely primarily on geometric or frequency-domain cues. Few explicitly inject spectral vegetation priors, such as the Normalized Difference Vegetation Index (NDVI), into either the prompt design or adaptation pipeline.

In this work, we introduce NDVI-Guided Low-Rank Adaptation of SAM for Sentinel-2 agricultural field boundary delineation. NDVI-guided prompting localizes candidate boundary regions, while LoRA fine-tuning adapts SAM’s mask decoder to correctly interpret multispectral boundary cues, improving boundary continuity and cross-region robustness. Specifically, our method:

- Automatically extracts NDVI-based spectral priors from Sentinel-2 time series,
- Converts these priors into structured prompt sets (including superpixel centroids and boundary-focused prompt bands) to guide SAM inference, and
- Performs lightweight LoRA fine-tuning on large FTW subsets to align SAM’s mask decoder with multispectral boundary signals.

We evaluate performance using both region-based metrics (IoU, Dice) and boundary-oriented metrics (Boundary F1 and mean boundary distance) across multiple FTW regions, as well as AI4Boundaries test cases where orthophoto comparisons are available. Our experiments quantify how NDVI-guided prompting and LoRA adaptation jointly improve boundary continuity and cross-region generalization while maintaining low computational and data costs.

## 2. Related Work

### 2.1 Datasets and Benchmarks for Agricultural Field Delineation

Large-scale annotated datasets have been critical enablers for training and benchmarking agricultural field boundary delineation models. The Fields of the World (FTW) dataset provides globally distributed field boundary masks across 24 countries and diverse agro-ecological contexts, enabling systematic cross-region transfer studies using Sentinel-2 imagery [5]. Similarly, the FieldSeg dataset established practical guidelines for field size thresholds and delineation performance under coarse spatial resolution conditions [6].

The AI4Boundaries dataset further highlights the spatial resolution limitations of Sentinel-2 imagery by benchmarking segmentation results against 1 m orthophotos across European and African agricultural landscapes [9]. Collectively, these benchmarks reveal two persistent constraints:

- Many agricultural parcels are smaller than the minimum resolvable size at 10 m resolution.
- Models trained in one geographic or phenological context often degrade significantly when transferred elsewhere.

### 2.2 Classical Segmentation Methods and Transfer Learning

Prior to the emergence of foundation models, agricultural field boundary extraction relied on classical machine learning and deep learning frameworks such as Random Forests, U-Net [7], and DeepLab [8]. These architectures can achieve strong pixel-level accuracy when extensive labeled data are available, but they typically require dense annotations and exhibit limited robustness under cross-region or cross-crop transfer scenarios [2]. Moreover, these approaches primarily exploit spatial context in RGB imagery while under-utilizing the rich multispectral information available in agricultural remote sensing.

### 2.3 Foundation Models for Remote-Sensing Segmentation

The introduction of the Segment Anything Model (SAM) [10] established a generic prompt-based segmentation paradigm with strong zero-shot generalization. However, SAM’s applicability to remote sensing imagery is limited by differences in spectral composition, spatial scale, and object granularity compared to natural images [10], [11].

Subsequent adaptations have explored various strategies:

- GeoSAM incorporated multi-modal prompts (text and visual) for infrastructure segmentation, demonstrating the feasibility of prompt diversification [13].
- fabSAM optimized prompt generation for farmland delineation but relied primarily on geometric cues and did not incorporate spectral vegetation indices [14].
- RSAM-Seg introduced frequency-domain features via Fourier transforms to enhance edge sensitivity in remote sensing imagery.
- Parameter-efficient fine-tuning approaches, including LoRA-based adaptation [11] and multi-stage multispectral adaptation pipelines [12], have been proposed to bridge domain gaps.

Despite these advances, none explicitly integrate vegetation indices such as NDVI as structured prompt signals within the SAM adaptation workflow.

### 2.4 Spectral Priors and NDVI for Boundary Delineation

Vegetation indices such as the Normalized Difference Vegetation Index (NDVI) are widely used proxies for canopy cover and crop vigor and have long been employed in agricultural remote sensing. Their ability to differentiate crop and non-crop regions makes them compelling candidates for boundary localization in agricultural mosaics. However, within the SAM adaptation literature, spectral indices are rarely used as prompt signals or fused priors. Instead, most methods emphasize geometric layout or texture-based cues, leaving a clear methodological gap for NDVI-guided prompt strategies.

### 2.5 Efficiency, Data Cost, and Operational Scalability

Operational deployment of agricultural field delineation systems depends not only on segmentation accuracy but also on computational efficiency, data cost, and domain generalizability. Methods such as FastSAM prioritize inference speed by replacing transformer backbones with lightweight CNNs, but do not address domain-specific spectral adaptation. Large-scale systems relying on high-resolution commercial imagery (<3 m) achieve strong accuracy but incur prohibitive acquisition costs and limited temporal revisit rates.

In contrast, the proposed NDVI-Guided LoRA-SAM framework leverages freely available Sentinel-2 imagery, applies efficient low-rank fine-tuning, and embeds spectral vegetation priors to improve cross-region generalization at minimal computational cost.