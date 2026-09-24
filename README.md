# Benchmark Analysis of Data Balancing Techniques on k-NN Classifiers for Imbalanced Datasets

This project investigates the impact of synthetic oversampling techniques (**SMOTE** and **Geometric SMOTE**) combined with various **k-Nearest Neighbors (k-NN)** classifier variants on class-imbalanced benchmark datasets.

All data balancing algorithms, distance-weighted decision mechanisms, and cross-validation pipelines were implemented **from scratch in MATLAB** without relying on black-box external machine learning toolboxes.

---

## 📌 Problem Overview
In real-world applications (e.g., medical diagnosis, fraud detection), datasets frequently exhibit severe class imbalance. Conventional classifiers optimized for overall accuracy often fail to detect critical minority instances. This study evaluates how synthetic data generation affects decision boundaries across different neighborhood topologies.

---

## ⚙️ Methodology & Pipeline

1. **Pre-processing:** Min-Max Feature Normalization to ensure distance metric consistency across dimensions.
2. **Train-Test Split:** 80/20 Holdout evaluation with dynamic class frequency detection.
3. **Hyperparameter Tuning:** 5-Fold Cross-Validation on the training set ($k \in [1, 10]$) with nested oversampling to prevent **data leakage**.
4. **Balancing Techniques Evaluated:**
   - **Baseline:** Original distribution (No oversampling)
   - **SMOTE:** Linear interpolation between nearest minority neighbors
   - **G-SMOTE:** Directional vector sampling around minority centers within a geometric hyper-sphere
5. **Classifier Architectures:**
   - **Standard k-NN:** Majority voting
   - **Modified (Distance-Weighted) k-NN:** Inverse distance weighting ($w = \frac{1}{d + \epsilon}$)
   - **Fuzzy k-NN:** Distance-based fuzzy class membership assignment
   - **Radius-based k-NN (r-NN):** Fixed-radius neighborhood majority voting

---

## 📊 Datasets & Benchmark Results

The pipeline was benchmarked across 5 binary classification datasets:
* **Yeast**
* **Parkinson**
* **Heart**
* **CTO**
* **BCWD (Breast Cancer Wisconsin Diagnostic)**

### Key Takeaways:
* **Trade-off Dynamic:** Balancing techniques significantly enhance minority class recall in datasets where baseline models fail (e.g., *Yeast* and *Heart*), at the expense of a slight decrease in majority class accuracy.
* **Geometric SMOTE vs. SMOTE:** G-SMOTE generally provides a more controlled boundary expansion, mitigating excessive over-generalization into majority territory compared to standard SMOTE.
* **Separability Matters:** For naturally well-separated distributions (e.g., *Parkinson*, *BCWD*), synthetic oversampling offers minimal benefit and can unnecessarily degrade overall performance.

---

## 📁 Repository Structure

```text
├── main_benchmark.m       # Full experimental pipeline (Pre-processing, CV, Models, Evaluation)
├── report.pdf             # Comprehensive technical paper detailing methodology and results
└── README.md              # Project documentation
