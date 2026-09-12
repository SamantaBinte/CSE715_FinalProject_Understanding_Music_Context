# Music Context Understanding with GNNs and Language Models

This project explores **music understanding using audio, textual context, Graph Neural Networks (GNNs), Transformer-based language models, and multimodal learning**.

The project consists of four progressively designed tasks, moving from individual modalities to cross-modal representation learning.

## Tasks

### Task 1 — BERT for Music Context

**Dataset:** MusicCaps
**Notebook:** `Task1_Music_Context_MusicCaps.ipynb`

Uses BERT to process MusicCaps captions and perform **multi-label music tag classification**. This serves as the text-based semantic baseline.

### Task 2 — GNN vs CNN for Music

**Dataset:** GTZAN
**Notebook:** `Task2_GNN_CNN_Music_Context_GTZAN.ipynb`

Represents audio as graphs of temporal segments and uses **GraphSAGE** for genre classification. A **CNN with log-mel spectrograms** is trained as a baseline for comparison.

### Task 3 — GNN + BERT Multimodal Fusion

**Dataset:** MagnaTagATune
**Notebook:** `Task3_GNN_BERT_Fusion_MagnaTagATune.ipynb`

Combines **GNN-based audio representations** with **DistilBERT-based semantic representations** for multi-label music tag prediction. Ablation experiments evaluate different fusion strategies, including concatenation and cross-attention.

### Task 4 — Cross-Modal Audio–Text Alignment

**Dataset:** MusicCaps
**Notebook:** `Task4_CrossModal_MusicCaps.ipynb`

Learns a shared embedding space for **music and text** using GraphSAGE, DistilBERT, and **InfoNCE contrastive learning**. The learned representations are evaluated through audio-to-text and text-to-audio retrieval using Recall@K.

---

## Overall Pipeline

```text
Task 1: Text
MusicCaps → BERT → Tag Classification

Task 2: Audio
GTZAN → Audio Graph → GraphSAGE
                 ↘ CNN Baseline

Task 3: Audio + Text
MTAT → GNN + DistilBERT → Multimodal Fusion → Tags

Task 4: Audio ↔ Text
MusicCaps → GNN + DistilBERT → Contrastive Learning
                              → Cross-Modal Retrieval
```

## Key Technologies

* **PyTorch**
* **Graph Neural Networks / GraphSAGE**
* **BERT / DistilBERT**
* **CNNs**
* **Librosa**
* **Hugging Face**
* **Multimodal Learning**
* **Contrastive Learning**
* **Cross-Modal Retrieval**

## Project Goal

The overall goal is to investigate how **musical structure and semantic context can be modeled independently, combined through multimodal learning, and ultimately aligned in a shared audio–text representation space.**
