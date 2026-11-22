# Logistic Regression

## Overview

Logistic Regression is a supervised learning algorithm primarily used for **classification**. In binary classification, it estimates the probability that an observation belongs to class `1`, then converts that probability into a class label using a decision threshold.

Despite its name, Logistic Regression is a classification algorithm rather than a method for predicting continuous targets. It combines a **linear score** with the **sigmoid function** to produce probabilities between `0` and `1`.

This module covers Logistic Regression from mathematical intuition to practical implementation, including model evaluation, decision thresholds, and L2 regularization.

## Topics Covered

1. Logistic Regression and binary classification
2. Linear scores and the sigmoid function
3. Probabilities, odds, and log-odds
4. Decision thresholds and decision boundaries
5. Binary cross-entropy (log loss)
6. Gradient Descent intuition and parameter updates
7. Implementing Logistic Regression with scikit-learn
8. Feature scaling and train-test splitting
9. Accuracy, precision, recall, F1-score, and confusion matrices
10. Threshold tuning and precision-recall trade-offs
11. L2 regularization and the `C` parameter

## Notebook

**Implementation:** [Module_14_Logistic_Regression.ipynb](Module_14_Logistic_Regression.ipynb)

The notebook contains a study-hours classification example, a sigmoid function implementation, a breast cancer dataset classification experiment, evaluation metrics, threshold comparisons, and an L2 regularization experiment.

## 1. Mathematical Intuition

### 1.1 From Features to a Linear Score

A Logistic Regression model first combines the input features into a linear score:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = \mathbf{w}^{T}\mathbf{x} + b
$$

Where:

- $x_1, x_2, \ldots, x_n$ are input features.
- $w_1, w_2, \ldots, w_n$ are learned feature weights.
- $b$ is the bias or intercept.
- $z$ is the raw model score, which can be any real number.

Unlike a probability, $z$ is not restricted to the interval from `0` to `1`. The sigmoid function converts it into a probability.

### 1.2 The Sigmoid Function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The predicted probability of class `1` is:

$$
P(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^{T}\mathbf{x} + b)
$$

The sigmoid creates an S-shaped curve with these properties:

- If $z$ is strongly negative, the probability approaches `0`.
- If $z=0$, the probability is exactly `0.5`.
- If $z$ is strongly positive, the probability approaches `1`.

Examples from the notebook:

- $\sigma(-5) \approx 0.0067$
- $\sigma(-2) \approx 0.1192$
- $\sigma(0) = 0.5$
- $\sigma(2) \approx 0.8808$
- $\sigma(5) \approx 0.9933$

The output is a probability estimate; the final class prediction requires a threshold.

### 1.3 Turning Probabilities into Class Predictions

Using the default threshold of `0.5`:

$$
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1\mid\mathbf{x}) \geq 0.5 \\
0 & \text{if } P(y=1\mid\mathbf{x}) < 0.5
\end{cases}
$$

**Worked example from the mathematical notes**

Suppose a simple study-hours model has $w=0.8$ and $b=-3$. For a student who studied for $5$ hours:

$$
z = wx+b = (0.8)(5)-3 = 1
$$

$$
P(\text{pass}) = \sigma(1) = \frac{1}{1+e^{-1}} \approx 0.7311
$$

Since $0.7311 \geq 0.5$, the predicted class is `1` (pass).

This is a mathematical illustration using assumed coefficients, not the fitted coefficients from the notebook.

### 1.4 Odds and Log-Odds

Probability tells us how likely an event is. **Odds** compare the probability of the event with the probability that it does not occur:

$$
\text{Odds} = \frac{p}{1-p}
$$

For example:

- If $p=0.5$, odds $=0.5/0.5=1$ (1-to-1 odds).
- If $p=0.8$, odds $=0.8/0.2=4$ (4-to-1 odds).
- If $p=0.2$, odds $=0.2/0.8=0.25$ (1-to-4 odds).

Taking the natural logarithm produces **log-odds**, also called the **logit**:

$$
\log\left(\frac{p}{1-p}\right) = \mathbf{w}^{T}\mathbf{x} + b
$$

This is a defining relationship in Logistic Regression: the **log-odds are linear in the input features**, even though the predicted probability follows a nonlinear sigmoid curve.

For $p=0.7$:

$$
\text{Odds} = \frac{0.7}{0.3} \approx 2.33,
\qquad
\text{Log-odds} = \ln(2.33) \approx 0.847
$$

### 1.5 Decision Boundary

For a `0.5` probability threshold, the decision boundary occurs where:

$$
\sigma(z)=0.5 \quad \Longleftrightarrow \quad z=0
$$

Therefore:

$$
\mathbf{w}^{T}\mathbf{x}+b=0
$$

For a single feature, using $w=0.8$ and $b=-3$:

$$
0.8x-3=0 \quad \Longrightarrow \quad x=3.75
$$

With these assumed coefficients:

- $x > 3.75$ gives a probability greater than `0.5`, so the prediction is class `1`.
- $x < 3.75$ gives a probability below `0.5`, so the prediction is class `0`.
- $x=3.75$ gives a probability of exactly `0.5`, which the stated threshold rule assigns to class `1`.

In multiple dimensions, the default-threshold boundary is a linear hyperplane in feature space.

### 1.6 Binary Cross-Entropy (Log Loss)

To learn appropriate weights and bias, Logistic Regression minimizes a loss function that penalizes incorrect probability estimates.

For $m$ training examples, binary cross-entropy is:

$$
J(\mathbf{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}
\left[y^{(i)}\log\left(p^{(i)}\right)
+ \left(1-y^{(i)}\right)\log\left(1-p^{(i)}\right)\right]
$$

Where $y^{(i)}$ is the actual class (`0` or `1`) and $p^{(i)}$ is the model's predicted probability for class `1`.

For one observation, the formula becomes:

$$
L(y,p) =
\begin{cases}
-\log(p) & \text{if } y=1 \\
-\log(1-p) & \text{if } y=0
\end{cases}
$$

**Intuition:** If the actual label is `1`, assigning a high probability to class `1` gives a smaller loss. If the actual label is `0`, assigning a high probability to class `1` gives a larger loss.

For example, a prediction of $p=0.8$ gives:

- Actual class `1`: $-\ln(0.8) \approx 0.223$ (low loss).
- Actual class `0`: $-\ln(0.2) \approx 1.609$ (higher loss).

### 1.7 Gradient Descent: How Weights Improve

Gradient Descent updates the model parameters in the direction that reduces the loss.

For unregularized Logistic Regression, the gradients are:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}
\left(p^{(i)}-y^{(i)}\right)x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}
\left(p^{(i)}-y^{(i)}\right)
$$

The parameter updates are:

$$
w_j \leftarrow w_j-\alpha\frac{\partial J}{\partial w_j},
\qquad
b \leftarrow b-\alpha\frac{\partial J}{\partial b}
$$

Here $\alpha$ is the learning rate.

**Worked example from the mathematical notes**

Consider one training observation with:

- Initial weight: $w=0.5$
- Initial bias: $b=-1$
- Input: $x=2$
- Actual class: $y=1$
- Learning rate: $\alpha=0.1$

**Step 1: Compute the linear score.**

$$
z=(0.5)(2)-1=0
$$

**Step 2: Calculate the predicted probability.**

$$
p=\sigma(0)=0.5
$$

**Step 3: Compute the gradients.**

$$
p-y=0.5-1=-0.5
$$

$$
\frac{\partial J}{\partial w}=(p-y)x=(-0.5)(2)=-1
$$

$$
\frac{\partial J}{\partial b}=p-y=-0.5
$$

**Step 4: Update the parameters.**

$$
w_{\text{new}}=0.5-(0.1)(-1)=0.6
$$

$$
b_{\text{new}}=-1-(0.1)(-0.5)=-0.95
$$

**Step 5: Recalculate the prediction.**

$$
z_{\text{new}}=(0.6)(2)-0.95=0.25
$$

$$
p_{\text{new}}=\sigma(0.25)\approx0.5622
$$

The predicted probability for the actual class `1` increased from `0.5` to approximately `0.5622` after one update. This single-observation example illustrates the direction of learning; the scikit-learn implementation handles optimization internally.

## 2. Practical Implementation

### 2.1 Study-Hours Toy Dataset

The notebook begins with a simple binary classification task:

- **Input feature:** `hours_studied`, ranging from `1` to `9` hours.
- **Target:** `pass_exam`, where `0` means fail and `1` means pass.

A `LogisticRegression` model is fitted to these observations, and a smooth probability curve is plotted over `200` points between `0` and `10` hours.

The fitted model outputs in the notebook are approximately:

- Intercept: `-5.2956`
- Coefficient for hours studied: `1.1781`

Example predictions from the notebook:

- `2` hours: probability of passing `0.0502`, predicted class `0`.
- `4` hours: probability of passing `0.3582`, predicted class `0`.
- `6` hours: probability of passing `0.8548`, predicted class `1`.
- `8` hours: probability of passing `0.9842`, predicted class `1`.

These predictions are based on the toy dataset; they are not a general model of real examination outcomes.

### 2.2 Sigmoid and Threshold Experiments

The notebook implements the sigmoid function from scratch, plots its S-shaped curve, and experiments with converting probabilities into class predictions.

In the short threshold example, the chosen threshold is `0.25`. Thus, probabilities of `0.30`, `0.49`, `0.50`, `0.70`, and `0.90` are assigned class `1`, while `0.10` is assigned class `0`.

**Notebook detail:** The output column in this example is named `class_at_threshold_0.5`, but the actual threshold used in the code is `0.25`. The predictions reflect `0.25`.

### 2.3 Breast Cancer Dataset

The main classification experiment uses scikit-learn's built-in Breast Cancer dataset.

Dataset characteristics:

- **Samples:** `569`
- **Features:** `30` numerical measurements
- **Class `0`:** malignant (`212` samples)
- **Class `1`:** benign (`357` samples)

The features cover measurements of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, along with corresponding error and worst-value features.

### 2.4 Data Preparation

The notebook uses an `80/20` train-test split with `random_state=42` and `stratify=y_full` to approximately preserve the class proportions.

Before fitting the classifier, it scales the input features using `StandardScaler`:

$$
x_{\text{scaled}}=\frac{x-\mu}{\sigma}
$$

Important implementation details:

- `fit_transform(X_train)` learns the mean and standard deviation from training features and scales the training set.
- `transform(X_test)` applies the **same training statistics** to the test set, avoiding data leakage.
- The binary target labels are not scaled because they represent classes rather than continuous measurements.

### 2.5 Training and Prediction

The model is trained with scikit-learn's `LogisticRegression` on the scaled training features.

Two prediction interfaces are used:

- `predict(X_test_scaled)` returns predicted class labels.
- `predict_proba(X_test_scaled)[:, 1]` returns the predicted probability of **class `1` (benign)**.

This distinction matters when interpreting threshold experiments and class-specific evaluation metrics.

## 3. Model Evaluation

### 3.1 Baseline Test Results

The notebook records these results on the `114`-sample test set:

- **Accuracy:** `0.982456` (approximately **98.25%**)
- **Precision for class 1:** `0.986111` (approximately **98.61%**)
- **Recall for class 1:** `0.986111` (approximately **98.61%**)
- **F1-score for class 1:** `0.986111` (approximately **98.61%**)

The confusion matrix is:

```text
                 Predicted 0   Predicted 1
Actual 0               41             1
Actual 1                1            71
```

Rows are actual labels and columns are predicted labels. Since `0` represents malignant and `1` represents benign:

- `41` malignant samples were correctly predicted as malignant.
- `1` malignant sample was incorrectly predicted as benign.
- `1` benign sample was incorrectly predicted as malignant.
- `71` benign samples were correctly predicted as benign.

Overall, the model correctly classified `112` of `114` test observations.

### 3.2 Understanding Classification Metrics

**Accuracy** measures the proportion of all predictions that are correct:

$$
\text{Accuracy}=\frac{\text{Correct predictions}}{\text{Total predictions}}
$$

**Precision** measures how many observations predicted as the selected positive class actually belong to that class:

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

**Recall** measures how many actual observations of the selected positive class were identified:

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

**F1-score** is the harmonic mean of precision and recall:

$$
F_1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

**Support** is the number of actual test examples belonging to each class. The classification report includes support of `42` for malignant and `72` for benign.

In scikit-learn's binary metric functions, class `1` is the positive class by default. Therefore, the reported `precision_score`, `recall_score`, and `f1_score` values in this notebook measure performance for **benign** cases. To focus on malignant cases, the positive label would need to be set to `0` or the class-`0` row of the classification report examined.

### 3.3 Threshold Tuning

Instead of always using `0.5`, the notebook evaluates thresholds from `0.1` through `0.9` on the predicted probability of class `1`.

Selected recorded outcomes:

- **Threshold `0.1`:** Accuracy `95.61%`, precision for class 1 `93.51%`, recall for class 1 `100%`.
- **Threshold `0.2`:** Accuracy `98.25%`, precision for class 1 `97.30%`, recall for class 1 `100%`.
- **Threshold `0.5`:** Accuracy `98.25%`, precision for class 1 `98.61%`, recall for class 1 `98.61%`.
- **Threshold `0.7`:** Accuracy `94.74%`, precision for class 1 `98.53%`, recall for class 1 `93.06%`.
- **Threshold `0.9`:** Accuracy `89.47%`, precision for class 1 `98.39%`, recall for class 1 `84.72%`.

**Interpretation:** Lowering the threshold generally predicts class `1` more often, which can increase recall for class `1` while reducing precision. Raising the threshold generally predicts class `1` less often.

**Important class-label detail:** Here class `1` means **benign**, not malignant. Lowering the threshold for `P(benign)` is therefore not the same as lowering a threshold for `P(malignant)`. In a medical context, evaluating missed malignant cases requires careful attention to which class is treated as positive.

## 4. L2 Regularization

Regularization discourages excessively large coefficients, helping control model complexity and reduce overfitting.

For L2 regularization, a penalty based on squared coefficients is added to the loss:

$$
J_{\text{regularized}} = J_{\text{log loss}} + \frac{\lambda}{2}\sum_{j=1}^{n}w_j^2
$$

Here $\lambda$ represents regularization strength in this conceptual form of the objective. The intercept is typically not included in the L2 penalty.

**Intuition:** Larger coefficients incur a larger penalty, encouraging the model to find a balance between fitting the training data and keeping coefficients reasonably small.

### The `C` Parameter

In scikit-learn's `LogisticRegression`, the `C` parameter is inversely related to regularization strength:

- **Smaller `C`:** Stronger regularization, generally smaller coefficient magnitudes.
- **Larger `C`:** Weaker regularization, allowing larger coefficient magnitudes.

L2 regularization is the default penalty in the Logistic Regression model used in this notebook.

### Regularization Experiment

The notebook compares two L2-regularized models on the scaled breast cancer data:

- **Weak regularization:** `C=1000`, test accuracy approximately **92.98%**.
- **Strong regularization:** `C=0.01`, test accuracy approximately **94.74%**.

It also compares the learned coefficients and visualizes the absolute values of the first ten feature coefficients. Stronger regularization produces much smaller coefficient magnitudes in this experiment.

For comparison, the earlier default-`C` baseline model achieved approximately **98.25%** test accuracy on the same split. The results show that **stronger regularization is not automatically better**; both very weak and very strong regularization can perform worse than an intermediate setting. Choosing `C` through validation or cross-validation is preferable to assuming that one extreme will work best.

## 5. Key Learning Outcomes

After working through this module, I practiced how to:

- Explain the relationship between a linear score, sigmoid, probability, and class prediction.
- Interpret odds, log-odds, and linear decision boundaries.
- Understand why binary cross-entropy is used for probability-based classification.
- Follow a worked Gradient Descent update for a Logistic Regression model.
- Build and evaluate a binary classifier using scikit-learn.
- Scale training and test features without leaking test-set statistics into training.
- Read a confusion matrix and distinguish precision, recall, F1-score, and accuracy.
- Test different classification thresholds and interpret the precision-recall trade-off.
- Examine how L2 regularization and the `C` parameter affect coefficient magnitudes and test performance.

## 6. Tools and Libraries

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook / Google Colab

To run the notebook locally, install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

Open the notebook with Jupyter or a compatible editor and run the cells in order.

## 7. Notes and Limitations

- The study-hours example is a small illustrative dataset, not a real-world prediction system.
- The breast cancer experiment uses a built-in educational dataset and a single held-out test split. Its metrics should not be interpreted as evidence of clinical readiness.
- Accuracy alone can hide important class-specific mistakes. The missed malignant case in the confusion matrix warrants attention when interpreting model performance.
- Threshold selection depends on the meaning of the positive class and the consequences of false positives versus false negatives.
- Regularization strength should be selected through appropriate validation rather than by looking only at a single test-set result.

## Learning Materials

The documentation is based on the supplied **Module 14 Logistic Regression** slides, handwritten mathematical intuition notes, and `Module_14_Logistic_Regression.ipynb` notebook. All reported implementation results are taken from the saved notebook outputs.
