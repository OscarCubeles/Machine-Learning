# Machine Learning

This repository contains the coursework completed for the **Introduction to Machine Learning** subject. It brings together three practical projects covering supervised classification, unsupervised clustering, dimensionality reduction, and data visualization.

## Projects

1. **Work 2 - Classification with lazy learning and SVM:** Implements and evaluates k-nearest neighbors and support vector machines on UCI classification datasets using predefined 10-fold cross-validation. The project explores distance metrics, voting schemes, feature weighting, instance reduction, statistical comparison, and the trade-offs between predictive performance, efficiency, and storage. See the [`project documentation`](w2/README.md), [`source code`](w2), [`assignment brief`](w2/Work2_IML_2024_v1.pdf), and [`supporting material`](w2/SuportWork2_2024_wide_ALL_s1.pdf).

2. **Work 3 - Clustering techniques:** Compares OPTICS, Spectral Clustering, K-Means, K-Means++, X-Means, and Generalized Fuzzy C-Means across the Hepatitis, MX, and Breast Cancer Wisconsin datasets. The implementations are assessed with internal and external clustering metrics, including silhouette score, purity, F1 score, Jaccard index, and adjusted Rand index. See the [`project documentation`](w3/README.md), [`source code`](w3), and [`report`](w3/IML_Work3.pdf).

3. **Work 4 - PCA and data visualization:** Implements Principal Component Analysis from scratch and compares it with scikit-learn's implementation. The study measures how dimensionality reduction affects K-Means++ and OPTICS clustering and compares PCA visualizations with UMAP on the Hepatitis and Breast Cancer Wisconsin datasets. See the [`project documentation`](w4/README.md), [`source code`](w4), and [`report`](w4/IML_Work4.pdf).

## Repository Structure

```text
.
├── w2/                         # k-NN and SVM classification
│   ├── algorithms/               # Classifier implementations
│   ├── preprocessing/            # Dataset-specific preprocessing
│   ├── datasets/                 # ARFF and prepared datasets
│   └── results/                  # SVM evaluation results
├── w3/                         # Clustering-technique comparison
│   ├── algorithms/               # Clustering and evaluation methods
│   ├── preprocessing/            # Dataset preprocessing
│   └── IML_Work3.pdf             # Project report
└── w4/                         # PCA, clustering, and visualization
    ├── algorithms/               # PCA and clustering implementations
    ├── preprocessing/            # Dataset preprocessing
    └── IML_Work4.pdf             # Project report
```

Each project is independent and includes its own implementation, datasets or preprocessing utilities, evaluation workflow, and supporting documentation.
