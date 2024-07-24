
# Evaluating Robustness in Extractive Question Answering Systems with Unanswerable Queries

This repository contains the code and resources for the project "Computational Semantics," based on the article "A Lightweight Method to Generate Unanswerable Questions in English." The project extends the evaluation of the method to various model architectures and tasks, focusing on both in-domain and out-of-domain applications.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Datasets](#datasets)
4. [Models](#models)
5. [Experimental Setup](#experimental-setup)
6. [Results](#results)
7. [Contact](#contact)

## Introduction
This project explores Extractive Question Answering (EQA) with a focus on handling unanswerable questions. We evaluate methods to generate unanswerable questions and assess their impact on various models, including BERT, FLAN-T5, and Llama 3, using datasets such as SQuAD 2.0, TyDi QA, and ACE-whQA.

## Project Structure
```
computational_semantics_project/
│
├── data/
│   ├── squad2.0/
│   ├── tydiqa/
│   └── ace_whqa/
│
├── models/
│   ├── bert/
│   ├── flan_t5/
│   └── llama3/
│
├── src/
│   ├── encoder.py
│   ├── decoder.py
│   └── encoder-decoder.py
│
├── eval/
│   ├── Meta-Llama-3-8B-Instruct/
│   │   ├── heatmaps/
│   │   └── Tables/
│   ├── bert-large-uncased/
│   │   ├── heatmaps/
│   │   └── Tables/
│   └── flan-t5-base/
│       ├── heatmaps/
│       └── Tables/
│
├── README.md
└── requirements.txt
```

## Datasets
We use three primary datasets:
1. **SQuAD 2.0**: A dataset with both answerable and unanswerable questions.
2. **TyDi QA**: A multilingual QA dataset, using only the English portion.
3. **ACE-whQA**: Comprises subsets to test various aspects of QA systems, including answerable, unanswerable, competitive, and non-competitive questions.

## Models
We fine-tune and evaluate the following models:
1. **BERT**: An encoder model known for its effectiveness in various NLP tasks.
2. **FLAN-T5**: An encoder-decoder model optimized for instruction-based learning.
3. **Llama 3**: A decoder model designed for text generation tasks.

## Experimental Setup
The experiments include:
1. **Dataset Preparation**: Preprocessing the datasets for training and evaluation.
2. **Model Fine-Tuning**: Fine-tuning the models on the datasets using various methods, including Antonym Augmentation and Entity Augmentation.
3. **Evaluation**: Assessing model performance on both in-domain and out-of-domain tasks.

## Results
Our experiments indicate that the Entity method significantly improves performance in in-domain tasks for the BERT model, while the UNANSQ method excels in out-of-domain scenarios. Detailed results and analysis can be found in the `eval` directory.

## Contact
For any inquiries, please contact the authors:
- Noam Azulay: noamaz@post.bgu.ac.il
- Daniel Samira: samirada@post.bgu.ac.il
- Yuval Gorodissky: yuvalgor@post.bgu.ac.il
- Alon Neduva: neduva@post.bgu.ac.il

---

This project is part of the Computational Semantics course at Ben-Gurion University of the Negev, Israel, Department of Software and Information Systems Engineering.

We have documented our findings and methodologies in a detailed paper. You can find the paper [here](Computational_Semantics_Final_Report.pdf).

---
