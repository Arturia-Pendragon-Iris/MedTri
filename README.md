# 🏥 MedTri  

## **MedTri: A Platform for Structured Medical Report Normalization to Enhance Vision–Language Pretraining**

MedTri is a lightweight, locally deployable platform for structured normalization of radiology reports.  
It transforms heterogeneous free-text reports into unified, anatomically grounded triplets, improving data consistency and fine-grained alignment for medical vision–language pretraining (VLP).

> **Paper:** *MedTri: A Platform for Structured Medical Report Normalization to Enhance Vision–Language Pretraining*  
> **Status:** Accepted by **MICCAI 2026**

<p align="center">
  <img src="https://github.com/Arturia-Pendragon-Iris/MedTri/blob/main/Figure_2.png" width="800"/>
</p>

---

## 📌 Overview

Medical vision–language pretraining (VLP) depends heavily on paired image–report data.  
However, raw radiology reports commonly exhibit:

- **Stylistic heterogeneity** across institutions and radiologists  
- **Inconsistent length and verbosity**  
- **Image-irrelevant content** (e.g., clinical history, recommendations)  
- **Weak fine-grained image–text alignment**

Existing normalization approaches typically:

- Focus on **limited schema extraction** (e.g., RadGraph-style NER pipelines), or  
- Rely on **cloud-based LLM rewriting**, raising privacy and scalability concerns  

---

## 🚀 What MedTri Provides

MedTri offers a structured, anatomy-grounded normalization framework that is:

- ✅ Lightweight  
- ✅ Privacy-preserving (fully local deployment)  
- ✅ Scalable for large-scale pretraining  
- ✅ Designed for fine-grained image–text alignment  

---

## 🧠 Core Concept: Structured Triplet Normalization

MedTri converts free-text reports into a unified schema:

```text
[Anatomical Entity: Radiologic Description + Diagnosis Category]
```

---

## 🛠️ Usage

### 🔹 1. Report Normalization

Use the normalization script:

```bash
python perform_normalization.py
```

**Input format:** `.xlsx` radiology reports  

Optional augmentation modules:

```bash
python explanation_augment.py   # MedTri-K
python counterfact_augment.py   # MedTri-C
```

---

### 🔹 2. Model Training

To train MedTri-based models:

```bash
python MedTri_train.py
```

A complete runnable example is provided in:

```bash
/example_train
```

---

## 📦 Model Release

Now you can freely download the pretrained checkpoints through the [Google Drive](https://drive.google.com/file/d/1TTeYbKIVbGaBP_g73j9bb4C3peDiCLiG/view?usp=sharing) link and update the path in perform_normalization.py to point to your local checkpoint directory.

---

## 🔬 Design Principles

MedTri is developed under three core principles:

1. **Anatomy-grounded structuring** instead of free-form rewriting  
2. **Local deployment** instead of cloud dependency  
3. **Pretraining-oriented alignment** instead of surface-level extraction  

---

## 🙏 Acknowledgement

This project references and builds upon:

- [RadGraph](https://github.com/Stanford-AIMI/radgraph)  
- [MedFILIP](https://github.com/PerceptionComputingLab/MedFILIP)

We acknowledge their open-source contributions to the medical AI community.

## 📖 Citation
If you find MedTri useful for your research, please consider citing our paper:
'''
@misc{chu2026medtriplatformstructuredmedical,
      title={MedTri: A Platform for Structured Medical Report Normalization to Enhance Vision-Language Pretraining}, 
      author={Yuetan Chu and Xinhua Ma and Xinran Jin and Gongning Luo and Xin Gao},
      year={2026},
      eprint={2602.22143},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.22143}, 
}
'''


