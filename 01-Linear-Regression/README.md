# Linear Regression: Concepts, Implementation, and Evaluation

This repository documents my learning and practical exercises on **Linear Regression**, a supervised machine learning algorithm used to predict continuous numerical values. The material focuses on understanding the best-fit line, measuring prediction errors, comparing optimization approaches, evaluating regression models, and recognizing their assumptions and limitations.

The exercises use a small **study hours vs. exam marks** dataset to build intuition before moving on to more complex problems.

## Topics Covered

### 1. Introduction to Linear Regression

Linear Regression models the relationship between one or more input features and a continuous target. For a single input feature, the prediction equation is:

$$
\hat{y} = mx + c
$$

- **$x$**: input feature (for example, study hours)
- **$\hat{y}$**: predicted target (for example, exam marks)
- **$m$**: slope or coefficient, indicating the change in predicted output for a one-unit increase in the input
- **$c$**: intercept, the predicted output when the input is zero

A scatter plot shows the observed data, while the **best-fit line** summarizes the overall linear trend.

### 2. Cost Function and Gradient Descent

A regression model needs a way to measure how far its predictions are from the actual values. **Mean Squared Error (MSE)** is one such cost function:

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Squaring prevents positive and negative errors from canceling and gives larger errors greater influence.

**Gradient Descent** is an iterative optimization method that adjusts parameters to reduce the cost. Its core idea is to compute the direction of increasing cost and move in the opposite direction. The **learning rate** determines the update size: a very small rate can make learning slow, while a very large rate can prevent stable convergence.

The practical notebook compares Scikit-learn's `LinearRegression` with `SGDRegressor`, which uses stochastic gradient descent.

### 3. Regression Evaluation Metrics

The exercises calculate several ways to measure model performance:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values. Easier to interpret in the target's original units.
- **MSE (Mean Squared Error):** Average squared prediction error. Gives larger mistakes more weight.
- **RMSE (Root Mean Squared Error):** Square root of MSE. Expresses error in the target's original units while emphasizing larger mistakes.
- **R² (Coefficient of Determination):** Measures how well predictions account for variation in the target relative to predicting its mean.

These metrics describe different aspects of performance; no single metric tells the whole story.

### 4. Assumptions and Limitations

The theory material also introduces important considerations when using Linear Regression:

- **Linearity:** The expected relationship should be adequately represented by a linear model.
- **Independence of errors:** Residuals should not systematically depend on one another.
- **Homoscedasticity:** Residual spread should remain reasonably constant across fitted values.
- **Normality of residuals:** Important particularly for conventional statistical inference.
- **Multicollinearity:** Highly correlated input features can make individual coefficient estimates difficult to interpret.

Other limitations discussed include sensitivity to outliers and difficulty representing complex nonlinear patterns or interactions with a simple straight-line model.

## Practical Notebooks

The following notebooks are part of this module. I maintain the executable notebooks separately from this README.

### [`Module_10_Part_01.ipynb`](Module_10_Part_01.ipynb) — Regression and Best-Fit Line

- Create a small dataset of study hours and marks.
- Fit a `LinearRegression` model using Scikit-learn.
- Inspect the learned slope and intercept.
- Plot the observed points and fitted regression line.
- Predict marks for a new study-hour value.

### [`Module_10_Part_02.ipynb`](Module_10_Part_02.ipynb) — Cost Function and Gradient Descent

- Fit a baseline `LinearRegression` model.
- Calculate prediction cost using `mean_squared_error`.
- Fit an `SGDRegressor` to the same dataset.
- Compare coefficients, mean squared errors, and fitted lines.
- Observe how optimization settings affect the result.

### [`Module_10_Part_03.ipynb`](Module_10_Part_03.ipynb) — Regression Evaluation

- Generate predictions from a fitted Linear Regression model.
- Compute MAE, MSE, RMSE, and R².
- Visualize actual observations against the regression line.
- Interpret the model's errors and overall fit.

## Dataset Used

The notebooks use this illustrative dataset:

| Study hours | Exam marks |
| ---: | ---: |
| 2 | 55 |
| 4 | 60 |
| 6 | 70 |
| 8 | 80 |
| 10 | 88 |

In this example, study hours are the input feature (`X`) and marks are the continuous target (`y`). The small dataset is intended for learning and visualization, not for making reliable real-world predictions.

## Example Results

From the saved notebook outputs:

- **Learned slope:** `4.3`
- **Learned intercept:** `44.8`
- **Fitted equation:** `Predicted marks = 4.3 × study hours + 44.8`
- **Predicted marks for 7 study hours:** `74.9`

The evaluation notebook reports the following metrics on the same five observations used for fitting:

| Metric | Value |
| --- | ---: |
| MAE | 1.04 |
| MSE | 1.52 |
| RMSE | 1.2329 |
| R² | 0.9898 |

**Evaluation note:** These are **training-data (in-sample)** metrics, not test-set scores. They show how closely the fitted line matches the example observations; they do not measure performance on unseen students.

**Optimization note:** In the saved gradient-descent notebook, `SGDRegressor` reports a higher MSE (approximately `20.91`) than `LinearRegression` (`1.52`) and produces a convergence warning. This is a useful learning observation: a gradient-descent-based model can require additional tuning or scaling to converge well, even on a small dataset.

## Tools and Libraries

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

To install the main dependencies:

```bash
pip install numpy matplotlib scikit-learn notebook
```

To open the notebooks from the project directory:

```bash
jupyter notebook
```

## What I Learned

Through this module, I practiced connecting the mathematical intuition behind Linear Regression with actual Python implementations. In particular, I learned how a model estimates slope and intercept, how predictions relate to observed data, why cost functions are necessary, how gradient-based optimization differs from direct fitting, and how evaluation metrics describe prediction quality. I also studied assumptions that matter when interpreting or applying a linear model.

## Scope

This repository focuses **only on Linear Regression**, including its best-fit line, cost function, gradient descent, evaluation metrics, assumptions, and limitations.
