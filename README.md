# Digits Recognition Case Study - Support Vector Machine (SVM) Classification

## Table of Contents
* [SVM Overview](#svm-overview)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Approach for SVM](#approach-for-svm)
* [Classification Outcome](#classification-outcome)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

## SVM Overview

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks where target is a categorical variable. The primary goal of SVM is to find the optimal hyperplane that separates data points of different classes (categories) with the maximum margin.

### Key Concepts

1. **Hyperplane**:
   - In an n-dimensional space, a hyperplane is a flat affine subspace of dimension (n-1) that separates the data points.
   - For 2D space, it is a line; for 3D space, it is a plane.

2. **Support Vectors**:
   - Data points that are closest to the hyperplane and influence its position and orientation.
   - The margin is defined by these support vectors.

3. **Margin**:
   - The distance between the hyperplane and the closest data points (support vectors) from either class.
   - SVM aims to maximize this margin.

### Equations

1. **Equation of the Hyperplane**:
   - For a set of features $\mathbf{x}$ and weights $\mathbf{w}$, the hyperplane is defined as:

   $$ \mathbf{w} \cdot \mathbf{x} + b = 0 $$

   where $\mathbf{w}$ is the weight vector, $\mathbf{x}$ is the feature vector, and $b$ is the bias term.

2. **Decision Function**:
   - The decision function determines the class of a data point $\mathbf{x}$:

   $$ f(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b) $$

   - If $f(\mathbf{x}) \ge 0$, the data point belongs to one class; otherwise, it belongs to the other class.

3. **Optimization Problem**:
   - The objective is to maximize the margin, which can be formulated as a constrained optimization problem:

   $$ \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 $$

   subject to the constraint:

   $$ y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 \quad \forall i $$

   where $y_i$ is the class label of the $i$-th data point ($y_i \in \{-1, +1\}$).

4. **Kernel Trick**:
   - SVM can be extended to handle non-linear separable data by mapping the data into a higher-dimensional space using a kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$.
   - Common kernels include linear, polynomial, radial basis function (RBF), and sigmoid.

### Key Advantages

- Effective in high-dimensional spaces.
- Versatile due to the use of different kernel functions.
- Robust to overfitting, especially in high-dimensional space, due to the regularization term $\|\mathbf{w}\|^2$.

### Limitations

- Computationally intensive for large datasets.
- The choice of the kernel and regularization parameters can significantly affect performance.
- Does not perform well with overlapping classes and noisy data.

<br>

**SVM Classification** has been utilized in this case study in a step-by-step manner to understand, analyse, transform and model the data provided for the analysis. The approach described here represent the practical process utilised in industry to predict categorical single or multi-class target parameters for business.


## Problem Statement

A classic problem in the field of pattern recognition is that of handwritten digit recognition. Here we have images of handwritten digits ranging from 0-9 written by various people in boxes of a specific size - similar to the application forms in banks and universities. The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image.

### Objectives

You need to develop a model using **Support Vector Machine** which should correctly classify the handwritten digits from 0-9 based on the pixel values given as features. Thus, this is a 10-class classification problem.

### Data Description

For this case study, we use the **MNIST** data which is a large database of handwritten digits. The 'pixel values' of each digit (image) comprise the features, and the actual number between 0-9 is the label. 

Since each image is of 28 x 28 pixels, and each pixel forms a feature, there are 784 features.
<a href='http://yann.lecun.com/exdb/mnist/'>Here is a detailed description of the dataset.</a>


## Technologies Used

Python Jupyter Notebook with Numpy, Pandas, Matplotlib and Seaborn libraries are used to prepare, analyse and visualise data. The following SVM specific classes & methods from `scikit-learn` library have been used in the case study:

- GridSearchCV
- RandomizedSearchCV
- KFold


## Approach for SVM

A four-phase approach is used for this modelling exercise:

* <span style="font-size: 18px;">Data Understanding and Cleaning</span>
    1. Import & Understand Data
    2. Address Class-Imbalance in Dataset
    3. Handle Missing & Bad Quality Data

* <span style="font-size: 18px;">Data Preparation for Model Building</span>

    4. Split Data into Train & Test

* <span style="font-size: 18px;">Model Building</span>

    5. Build Initial Linear Model & Analyse Performance
    6. Building Non-Linear SVM Model
    7. Tune Hyperparameters thru' Visualisation
    8. Finalise Model with Optimum Hyperparameters

* <span style="font-size: 18px;">Evaluating Model</span>

    9. Perform Prediction and Evaluate Final Model
    10. Explain Final Model for Business Use

<br>

Some important steps in this approach include,

- Visualise the data (in this case for a digit, using the pixels for one observation) for better understanding

- Check if the dataset is `balanced` by reviewing the % of data from each class. In this case, each class has a fraction of 9-11% in the dataset

- Split the available data into `Training` and `Test` data sets. Since the data set is larger, only 15% of the data is used for training

- Scale the training data set using appropriate `Scaler` to bring all the independent variables to same range

- Build the initial model as `Linear Model` with C=1 using training data set and assess the model prediction performance parameters like `Accuracy, precision, recall, f1-score` etc.

- Subsequently, increase model complexity by training with non-linear models with `rbf` kernel and check if the model performance is increasing

- Perform cross-validation by splitting the training dataset in `K-folds` by setting-up `GridSearchCV` with a combination of hyperparameters, `C` (the regularisation parameter) and `Gamma` (extent of non-linearity)

- Visualise the impact of `C` and `Gamma` on model parameters (e.g. `Accuracy`) between training and test data sets and determine the optimum values for the hyperparameters

- Optionally, finetune the hyperparameters using `RandomisedSearchCV` by providing an optimal range of values to the algorithm 

- Finalise the model using the best scroes from the cross-validation results

- Perform prediction using `Test` data set by scaling it using previously established `Scaler`

- Evaluate the model using `confusion matrix` and expected business metric (e.g. Accuracy, Sensitivity, Precision etc.) and confirm if the final model performs better than the linear model

- Finally explain the model in business terms for easy comprehension of Business Users

## Classification Outcome

The developed model is able to predict a handwritten digit with a sensitivity of 94% which is very good.

$$Sensitivity = \frac{No.\ of\ correctly\ predicted\ specific\ digit}{Total\ no.\ of\ actual\ observations\ of\ the\ same\ digit}$$

Each digit is represented by a set of features (like pixel brightness at different positions). The model multiplies these features by the learned weights (dual coefficients) and adds the bias to decide which digit it is.

## Conclusion

Support Vector Machines (SVMs) are powerful tools for classifying categorical targets, aiding in crucial business decision-making processes. By finding the optimal hyperplane that separates different categories, SVMs ensure high accuracy in classification tasks. This ability to distinguish between classes helps businesses segment their customers, detect fraud, and predict outcomes with greater precision. Additionally, SVMs can handle high-dimensional data efficiently, making them suitable for complex datasets often encountered in business environments.


## Acknowledgements

This case study has been developed as part of Post Graduate Diploma Program on Machine Learning and AI, offered jointly by Indian Institute of Information Technology, Bangalore (IIIT-B) and upGrad.