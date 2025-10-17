# MuseScorer: Idea Originality Scoring At Scale

This repository contains a fully automated pipeline to score the **originality of ideas** in Guilford’s Alternative Uses Test (AUT), using large language models (LLMs) and an externally orchestrated Retrieval-Augmented Generation (RAG) framework.

Preprint:
> [MuseScorer: Idea Originality Scoring At Scale](https://arxiv.org/pdf/2505.16232)
> A. S. Bangash, K. Veera, I. A. Islam, R. A. Baten, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025 (EMNLP 2025)
---

## 🚀 Overview

This system annotates whether a new idea is a **rephrased variant** of existing ideas or constitutes a **novel idea bucket**, enabling psychometrically valid frequency-based originality scoring at scale. It works across multiple objects (e.g., *shoe*, *button*) and supports various LLMs and embedding models.

---

## 🧐 Key Features

* **LLM-as-a-Judge** via Ollama API (`llama3.3:70B`, `phi4`, `qwen3`)
* **Embedding-based retrieval** with `sentence-transformers` (e.g., `e5`, `mpnet`, `bge`)
* **KNN-based comparison** using `scikit-learn`
* **Persistent codebook and annotation logs** (via `pickle` and `npy`)
* **Checkpointing and resumption** across multiple attempts
* **Auto-sorted and merged CSV exports** for downstream analysis

---

## 🧠 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**

* `ollama`
* `sentence-transformers`
* `scikit-learn`
* `torch`
* `numpy`, `pandas`, `tqdm`
* A working **Ollama server** with the chosen model pulled locally.

### 2. Input Files

Ensure the following input files exist:

* `input_files/ideas_<object>.csv` — with at least `id`, `idea_content`, `for_user_id` columns
* `input_files/forbidden_ideas.csv` — with `object_name` and `forbidden_idea` columns

---

## ⚙️ Configuration

Set the following parameters in `MuseRAG_annotator_primary_dataset.ipynb`:

```python
study_prefix = "simpl_prmpt"
llm_index = 2                # 1: llama3.3, 2: phi4, 3: qwen3
prompt_index = 2             # 1: baseline, 2: CoT
embedding_index = 2          # 1: mxbai, 2: e5, 3: mpnet, 4: bge
comparison_k = 10            # Number of comparison ideas
replication_id = 6           # Seed for shuffling
object_names = ["shoe", "button", "key", "wooden_pencil", "automobile_tire"]
```

Then run the notebook.

---

## 📁 Output Structure

* `databases/` — embeddings, codebooks, and annotations (as `.pkl` and `.npy`)
* `checkpoints/` — keeps track of annotated idea IDs and failed ones
* `exports/` — final sorted and merged CSVs

You will find both:

* `*_annotated_ideas.csv` (for each task)
* `*_codebook.csv` (for each task)
* `*_all.csv` (merged across all tasks)

---

## 📆 Example Output

CSV columns in `exports/` include:

* `idea_ids`, `idea_texts`, `idea_annotation_ids`
* `idea_for_user_ids`, `idea_object_names`, `idea_reasons`

These are ready for analysis (e.g., originality score calculation via frequency-based metrics).

---

## 🧪 Annotation Logic

1. Embed new idea
2. Retrieve `k` nearest neighbors from prior codebook
3. Combine with any *forbidden idea* for the object
4. Format as prompt to LLM
5. Parse and save LLM’s annotation
6. Update codebook if a new bucket is created

---

## 🧼 Notes

* LLM prompting supports both `baseline` (ID only) and `CoT` (ID + reason) modes.
* Codebook ID 0 is reserved for *forbidden* ideas.
* The system is **stateless**, ensuring repeatability and auditability.

---

## 📖 Citation

This tool was developed as part of a larger research initiative on **scalable human-AI creativity assessments**. For technical details, see:

> A. S. Bangash, K. Veera, I. A. Islam, R. A. Baten, MuseScorer: Idea Originality Scoring At Scale, <i>Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing</i>, 2025 (EMNLP 2025) [Preprint](https://arxiv.org/pdf/2505.16232)

---

## 📬 Contact

Maintainer: \[Raiyan Abdul Baten (rbaten@usf.edu]
For issues or feedback, open a GitHub issue or reach out via email.

---

## 📾 License

MIT License
