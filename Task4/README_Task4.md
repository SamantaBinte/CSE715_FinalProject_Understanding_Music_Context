# Task 4: Cross-Modal MusicCaps Alignment

GNN-BERT contrastive dual-encoder for aligning audio structure graphs with
natural-language music captions (MusicCaps), per Task 4 of the project spec.

## What this does

- **Audio encoder**: GraphSAGE over per-clip segment graphs. Each node is a
  ~2s audio segment (chroma + MFCC features, from `audio_features.py`);
  edges are temporal adjacency + cosine-similarity links between segments.
- **Text encoder**: frozen DistilBERT + a small trainable projection head on
  the `[CLS]` embedding.
- **Training**: InfoNCE contrastive loss aligning (graph, caption) pairs,
  mixed precision, early stopping on validation R@10.
- **Evaluation**: Caption-to-Audio and Audio-to-Caption retrieval (R@1/5/10),
  10 qualitative retrieval examples, and zero-shot multi-label tag
  prediction (Macro-F1 / Micro-F1 / AUC-PR) using the same 50-tag vocabulary
  as Task 3, for a direct comparison against Task 3's supervised fusion
  model.

## How to run

Open `Task4_GNN_BERT_MusicCaps.ipynb` in Google Colab with a GPU runtime,
and run all cells top to bottom.

**Required uploads** (the notebook prompts for these when it reaches the
relevant cell):
1. `musiccaps-public.csv` -- official MusicCaps metadata (ytid, timestamps,
   caption, aspect list).
2. `audio_features.py` -- shared chroma/MFCC feature-extraction module.

**No YouTube download needed.** Audio clips are pulled from a public
Hugging Face dataset mirror (`nicolaus625/cmi`), which hosts pre-extracted
MusicCaps `.wav` files by YouTube ID. This avoids YouTube's bot-detection
and cookie/PO-token requirements entirely.

## Known limitation: dataset coverage

The HF mirror covers **2,085 of the official 5,521 MusicCaps clips**
(verified via `HfApi.list_repo_files`). The pipeline trains and evaluates
on this subset. This does not violate the task spec, which only requires
paired (graph, caption) data -- it does not require the full 5,521 clips.
If more coverage is needed later, the remaining clips would have to be
sourced from YouTube directly (requires cookies + a JS-challenge-capable
runtime, due to current YouTube bot-detection).

## Efficiency choices (kept training cheap on limited GPU time)

- BERT backbone is **frozen** -- only the projection heads and the GNN are
  trained (see the trainable-vs-total parameter count printed in the model
  cell).
- Audio features and graphs are **cached to disk** after first extraction;
  re-running the notebook does not redo this work.
- Mixed precision (`torch.cuda.amp`) and early stopping are used throughout
  training.

## Outputs produced

- `results/task4_retrieval_metrics.json` -- final test-set R@1/5/10, both
  directions.
- `results/task4_qualitative_retrieval.json` -- 10 example query captions
  with their top-3 retrieved clips.
- `results/task4_zero_shot_tag_metrics.json` -- zero-shot Macro-F1 /
  Micro-F1 / AUC-PR, comparable to Task 3's `ablation_results.csv`
  (`cross_attn` row).

## Notes on the Task 3 comparison

Task 3 trains a supervised GNN-BERT fusion classifier on MagnaTagATune
tags; Task 4 never sees a tag label at all, only captions via contrastive
learning. To make the two Macro-F1 numbers comparable despite the different
underlying datasets, Task 4's zero-shot evaluation reuses Task 3's exact
50-tag vocabulary, with ground-truth labels built as a proxy from each
MusicCaps clip's caption/aspect list. The final cell prints both numbers
side by side once Task 3's result is filled in.
