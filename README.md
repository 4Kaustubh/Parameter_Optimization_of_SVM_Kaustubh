# ShivaneKapoor_102203191_Parameter-Optimization-of-SVM

# 🔍 SVM Hyperparameter Tuning on Letter Recognition Data

This project explores **Support Vector Machine (SVM)** optimization using `RandomizedSearchCV` on the classic **Letter Recognition** dataset. The goal is to identify the best-performing model configurations using various randomized train-test splits, evaluate generalization performance, and track optimization behavior visually.

---

## 📂 Project Overview

- Load and preprocess the **Letter Recognition** dataset from UCI (via OpenML)
- Generate **10 unique train-test splits** with stratified sampling (70% train, 30% test)
- Perform **SVM tuning** using `RandomizedSearchCV` (10 iterations, 3-fold CV)
- Track simulated **convergence accuracy trends** across 100 steps
- Generate tabular performance summaries and a convergence graph
- Save detailed results including analytics and model metrics

---

## 📊 Dataset Summary

| Attribute        | Description                  |
|------------------|------------------------------|
| **Dataset Name** | Letter Recognition           |
| **Samples**      | 20,000                       |
| **Features**     | 16 (real-valued)             |
| **Target Classes** | 26 (Alphabets A–Z)         |
| **Source**       | UCI Repository / OpenML      |

---

## 🛠️ Technologies & Libraries

- Python
- scikit-learn
- pandas, numpy
- matplotlib
- RandomizedSearchCV

---

## 🧠 Model Configuration & Tuning

Each of the 10 dataset splits was trained using the `SVC` model with the following hyperparameter grid:

| Parameter   | Values Tested            |
|-------------|---------------------------|
| **Kernel**  | `'rbf'`, `'linear'`       |
| **C**       | `0.1`, `1`, `10`          |
| **Gamma**   | `'scale'`, `'auto'`       |
| **CV Folds**| 3                         |
| **Iterations** | 10 random combinations |

> ✨ Convergence is **simulated** by interpolating sorted cross-validation scores over 100 steps for visualization.

---

## 📈 Sample Convergence Visualization

Below is the simulated convergence plot for the **best-performing sample**, showing how accuracy improved during parameter exploration.

![image](https://github.com/user-attachments/assets/21accb83-15e3-47a3-9266-57a1bcaa9d79)


---

## 🧾 Summary of SVM Optimization Results

Each row in the results table corresponds to the **best configuration** for one of the 10 randomized data splits.

![image](https://github.com/user-attachments/assets/821eea06-26d2-4e98-b5b0-9655f9a74727)


Includes:
- Accuracy on test set
- Optimal hyperparameters: kernel, C, gamma

---

## 📌 Additional Analysis

A basic analytics summary is generated and saved:

- Dataset size and number of classes
- Best-performing sample ID
- Highest achieved accuracy
- Corresponding SVM parameters

![image](https://github.com/user-attachments/assets/6999ec7f-a827-4ced-aa65-869b581a347a)


---

## 🚀 Future Enhancements

- Add support for additional kernels (`poly`, `sigmoid`)
- Compare `GridSearchCV` vs `RandomizedSearchCV`
- Visualize misclassifications using confusion matrices
- Track training time for each configuration
- Analyze feature importance (e.g., using PCA)

---

## ✅ Conclusion

This experiment demonstrates how **SVM**, even with a basic feature set, can perform well on complex multiclass classification tasks when tuned effectively. The workflow is modular and extensible, making it easy to adapt for future machine learning experiments.

Shivane Kapoor_3C15_(102203191)

---
