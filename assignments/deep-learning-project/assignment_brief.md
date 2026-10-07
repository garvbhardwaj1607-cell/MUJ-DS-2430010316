# Deep Learning Project Assignment – 20 Marks

Total Marks: 20
Nature of Submission:** Individual Contribution
**Initial Evaluation Submission Deadline:** **Monday, 14 September 2026**

## 1. Evaluation Structure

The project will be evaluated in three stages/evaluations. The marks will be distributed as follows:

| Evaluation Component           |        Marks | Requirement                                                                                                    |
| ------------------------------ | -----------: | -------------------------------------------------------------------------------------------------------------- |
| **CO3 – Apply**                | **10 Marks** | Implementation of the proposed Deep Learning model and successful application to the selected dataset          |
| **CO4 – Comparative Analysis** |  **5 Marks** | Comparison of your results with other state-of-the-art (SOTA) methods on the **same dataset**                  |
| **CO5 – Innovation**           |  **5 Marks** | Novel/innovative contribution, modification, optimization, architecture, methodology, or experimental approach |
| **Total**                      | **20 Marks** |                                                                                                                |

### Important

* A candidate who completes only the **CO3 (Apply)** component can receive a maximum of **10 marks**.
* Candidates who additionally perform a proper **comparative analysis with state-of-the-art methods on the same dataset** can earn **5 additional marks under CO4**.
* Candidates demonstrating a meaningful **innovation/novel contribution** can earn **5 additional marks under CO5**.
* If a candidate completes the entire project in the first evaluation, it is appreciated. However, the candidate may be given an opportunity to **optimize, improve, and refine the project** during subsequent evaluations.

---

## 2. Submission Deadline

The initial submission deadline has been extended by **2 days**.

**Final deadline for the initial evaluation: Monday, 14 September 2026.**

Late submissions will be subject to an **appropriate penalty in marks for each day of delay**.

Therefore, students are strongly advised to submit the project within the prescribed deadline.

---

## 3. Project Upload Instructions

**Project Upload Link:**
[Insert Project Upload Link]

### File Format

Only a **ZIP folder** will be accepted.

### Naming Convention

The ZIP file **must be named using the Registration Number only**:

```text
RegistrationNumber.zip
```

### Example

```text
202612345.zip
```

**Do NOT include:**

* Student name
* Project name
* Model name
* Dataset name

in the ZIP file name.

---

## 4. Individual Contribution

This is an **individual project**.

Each candidate must be able to explain and demonstrate their own:

* Dataset
* Preprocessing
* Model architecture
* Training procedure
* Hyperparameter selection
* Evaluation
* Results
* Comparative analysis
* Innovation/contribution

---

## 5. Required Publication-Oriented Results

Students are expected to prepare proper experimental results. The following **matrices/tables** should be included wherever applicable:

### A. Dataset Analysis

| Matrix/Table                       | Requirement                        |
| ---------------------------------- | ---------------------------------- |
| Dataset distribution               | Number of samples in each class    |
| Train/Validation/Test distribution | Number and percentage of samples   |
| Class distribution                 | Class imbalance analysis           |
| Dataset preprocessing              | Preprocessing steps and parameters |

### B. Model Performance

| Matrix/Table         | Requirement                     |
| -------------------- | ------------------------------- |
| **Confusion Matrix** | Actual vs. predicted classes    |
| Accuracy             | Overall classification accuracy |
| Precision            | Per-class and/or macro/weighted |
| Recall               | Per-class and/or macro/weighted |
| F1-score             | Per-class and macro/weighted F1 |
| Specificity          | Where applicable                |
| ROC-AUC              | Where applicable                |

### C. Model Comparison – CO4

Students attempting CO4 must compare their proposed model with **existing state-of-the-art methods on the same dataset**.

Recommended comparison matrix:

| Model          | Accuracy | Precision | Recall | F1 | AUC | Parameters |
| -------------- | -------: | --------: | -----: | -: | --: | ---------: |
| SOTA Model 1   |          |           |        |    |     |            |
| SOTA Model 2   |          |           |        |    |     |            |
| SOTA Model 3   |          |           |        |    |     |            |
| Proposed Model |          |           |        |    |     |            |

The comparison should use **reliable published methods/papers** wherever possible.

### D. Ablation Study – CO5

For an innovation claim, students should demonstrate the contribution experimentally.

Example:

| Experiment   | Component A | Component B | Component C | F1 | Accuracy |
| ------------ | ----------- | ----------- | ----------- | -: | -------: |
| Baseline     | ✗           | ✗           | ✗           |    |          |
| Experiment 1 | ✓           | ✗           | ✗           |    |          |
| Experiment 2 | ✓           | ✓           | ✗           |    |          |
| Proposed     | ✓           | ✓           | ✓           |    |          |

### E. Training Analysis

Where applicable, include:

* Training vs. validation loss
* Training vs. validation accuracy
* Learning curves
* Convergence analysis
* Hyperparameter comparison
* Computational cost
* Number of trainable parameters

### F. Innovation/CO5 Results

Students claiming innovation should provide evidence such as:

* Ablation study
* Improved architecture
* New preprocessing strategy
* New feature extraction technique
* Optimization technique
* Attention mechanism
* Ensemble strategy
* Transfer-learning strategy
* Parameter-efficient fine-tuning
* Improved generalization
* Robustness analysis
* Other clearly justified novel contribution

**Simply changing a few hyperparameters will generally not be considered sufficient innovation.**

---

## 6. Mandatory README File

Every ZIP folder **must contain a `README` file**.

The README should include the following:

### 1. Project Title

Clearly mention the project title.

### 2. Student Details

* Name
* Registration Number
* Section

### 3. Dataset

Mention:

* Dataset name
* Dataset source
* Number of samples
* Number of classes
* Train/validation/test split
* Preprocessing performed

### 4. Model

Mention:

* Model/architecture name
* Architecture description
* Pretrained model, if applicable
* Proposed modifications

### 5. Hyperparameters

Include:

| Hyperparameter   | Value |
| ---------------- | ----- |
| Learning rate    |       |
| Batch size       |       |
| Number of epochs |       |
| Optimizer        |       |
| Loss function    |       |
| Scheduler        |       |
| Dropout          |       |
| Weight decay     |       |
| Other parameters |       |

### 6. Hyperparameter Tuning

Mention:

* Parameters considered
* Values tested
* Final selected values
* Reason for selection

### 7. Results

Mention the major evaluation results and provide references to the result files/figures.

### 8. Innovation

Clearly explain:

> **What is your innovation compared with the baseline/state-of-the-art methods?**

### 9. Folder Structure

The README must explain the project folder structure.

Example:

```text
RegistrationNumber/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── dataset_information.txt
│
├── src/
│   ├── train.py
│   ├── test.py
│   ├── model.py
│   └── preprocessing.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   ├── results.csv
│   └── comparison.csv
│
├── models/
│   └── model_description.txt
│
└── figures/
    └── ...
```

---

## 7. requirements.txt

A **`requirements.txt` file is mandatory**.

It must contain the Python libraries and versions required to reproduce the project.

Example:

```text
python==3.x
torch==x.x.x
torchvision==x.x.x
numpy==x.x.x
pandas==x.x.x
scikit-learn==x.x.x
matplotlib==x.x.x
seaborn==x.x.x
opencv-python==x.x.x
```

Students should specify the actual versions used in their project.

---

## 8. Minimum Requirement for CO3 – 10 Marks

To achieve the **CO3 component**, the candidate should demonstrate:

1. Dataset selection and preparation
2. Data preprocessing
3. Deep Learning model implementation
4. Model training
5. Testing/evaluation
6. Appropriate performance metrics
7. Confusion matrix
8. Training/validation analysis
9. Working source code
10. Reproducible project structure

---

## 9. Additional Requirement for CO4 – 5 Marks

For **CO4 – Comparative Analysis**, the candidate should:

* Select relevant state-of-the-art methods.
* Use the **same dataset** for comparison wherever possible.
* Report appropriate performance metrics.
* Clearly compare the proposed model against existing methods.
* Explain why the proposed model performs better, similarly, or worse.
* Provide references to the compared research papers.

---

## 10. Additional Requirement for CO5 – 5 Marks

For **CO5 – Innovation**, the candidate should demonstrate a meaningful contribution beyond straightforward implementation.

The innovation must be:

* Clearly defined
* Technically justified
* Experimentally evaluated
* Supported by quantitative results

An **ablation study** is strongly recommended to demonstrate the contribution of the proposed innovation.

---

## 11. Final Submission Checklist

Before uploading, verify that your ZIP contains:

* [ ] `README.md`
* [ ] `requirements.txt`
* [ ] Source code
* [ ] Dataset information
* [ ] Model information
* [ ] Hyperparameters
* [ ] Hyperparameter tuning details
* [ ] Training results
* [ ] Test results
* [ ] Confusion matrix
* [ ] Performance metrics
* [ ] Comparison with SOTA methods, if attempting CO4
* [ ] Innovation/ablation experiments, if attempting CO5
* [ ] Figures and tables
* [ ] Folder structure documentation

### File name:

```text
RegistrationNumber.zip
```

**No student name or project name should be included in the ZIP file name.**

**Submission Deadline: 14 September 2026**

Late submission will attract an appropriate penalty in marks.