
<!-- ===== BADGES ===== -->

[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)

[![Colab](https://img.shields.io/badge/Colab-GPU%20Ready-orange?logo=googlecolab)](https://colab.research.google.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org)


# 🚀 TIRE: Data-Centric Relation Extraction for Cyber Threat Intelligence

🔐 **TIRE** (**T**hreat **I**ntelligence **R**elation **E**xtraction) is a **data-centric framework** for high-precision relation extraction (RE) in cybersecurity.  
Instead of increasing architectural complexity, TIRE demonstrates that **better data representations + task design outperform heavier models**, achieving **state-of-the-art accuracy with lower computational cost**.

📌 This repository provides a **reference implementation** of the TIRE framework and sample experiments to support reproducibility and adoption.

---

## 📄 Reference Publication

If you use this repository, **please cite**:

```bibtex
@inproceedings{mouiche2026tire,
  title={TIRE: Advancing Threat Intelligence Relation Extraction with a Novel Data-Centric Framework},
  author={Mouiche, Inoussa and Saad, Sherif},
  booktitle={Applied Cryptography and Network Security Workshops (ACNS 2025)},
  series={Lecture Notes in Computer Science},
  volume={15655},
  year={2026},
  publisher={Springer},
  doi={10.1007/978-3-032-01823-6_1}
}
🎯 Why TIRE?
Most RE systems are model-centric:

deeper architectures

attention stacking

higher latency

marginal gains

💡 TIRE flips the paradigm by showing that:

Well-engineered data representations + entity-aware masking + lightweight classifiers outperform complex neural architectures.

🧠 Key Idea: Data-Centric vs Model-Centric RE
❌ Model-Centric RE (Traditional)
Complex architectures (CNNs, BiLSTMs, CRFs, Attention stacks)

Long training and inference time

Hard to deploy in real-time CTI pipelines

✅ Data-Centric RE (TIRE)
Entity-aware masking and pooling

Explicit entity type embeddings

Simple dense classifiers

Faster, more explainable, more accurate

🏗️ TIRE Architecture
🧩 Core Components

Pretrained Language Model (BERT / RoBERTa / SecureBERT / SecureBERT⁺)

Entity Mask Pooling

Entity Type Embeddings

Lightweight RE Dense Layer

📌 The model focuses learning only on relevant entity spans, eliminating noise and redundancy.

(See paper for full architectural details)

📊 Performance Highlights

<p align="center">
<img width="464" height="467" alt="Image" src="https://github.com/user-attachments/assets/d3dc1cba-b12a-4c02-9ce9-51cff1c7e928" />

</p>
📊 Performance Highlights

<p align="center">
<img width="600" height="361" alt="Image" src="https://github.com/user-attachments/assets/0e5efd0e-b70d-4367-a9af-6d105db2a760" />
</p>

✅ Higher accuracy
✅ ~2× faster training
✅ Simpler architecture

🧪 Sample Implementation
This repository includes:

✅ A BERT-based TIRE implementation

✅ Entity masking & pooling logic

✅ Entity-type embedding integration

✅ End-to-end RE training & evaluation

⚠️ Note
Most full experimental pipelines remain under WASP Lab permissions.
This repo is provided as a reference implementation for learning, reproduction, and extension.

🖥️ How to Run
Requirements
Python ≥ 3.8

PyTorch

HuggingFace Transformers

GPU-enabled environment (recommended)

Recommended Setup
✔ Google Colab (GPU)
✔ VS Code / Jupyter

🌍 Applications
🛡️ Cyber Threat Intelligence (CTI)

📊 Security Knowledge Graphs

🧠 AI-assisted Threat Attribution

🏥 Transferable to Biomedical RE

🏭 Supply-Chain Intelligence

📬 Questions & Collaboration
💬 I’ve received multiple requests for TIRE implementations — happy to help!

Open an issue

Reach out for research collaboration

Just remember to cite the paper

⭐ If this work helped you, consider starring the repo!
