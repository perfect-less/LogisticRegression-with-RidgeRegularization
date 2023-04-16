# Logistic Regression with Ridge Regularization
This is a simple implementation of Logistic Regression algorithm trained using gradient descent on binary cross entropy with ridge regularization. 

# Test using generated data
To test the algorithm, I generated data with 500 samples where the y values is 1 if x is larger than 0.7 and 0 otherwise. The generated data were as follows:


Generated Data             |  Fitted Curve
:-------------------------:|:-------------------------:
![generated data](images/GeneratedData.png#gh-light-mode-only) ![generated data](images/GeneratedData_dark.png#gh-dark-mode-only) |  ![fitted curve](images/FittedCurve.png#gh-light-mode-only) ![fitted curve](images/FittedCurve_dark.png#gh-dark-mode-only)

Then I run the algorithm on previous data and the result is the plot on the right.


# How to use
1. Import `LogisticRegression` class from `logreg` module

```python
from logreg import LogisticRegression
```

2. Prepare your data, the shape of X is `(n, n_features)`, and the shape of y is either `(n,)` or `(n, 1)`.

3. Initialize the classifier object
```python
logReg = LogisticRegression(
            lr=0.1,         # learning rate
            ld=0.00001,     # lambda for ridge regularization
            batch_size=20,  # batch size
            epochs=1000,    # number of epochs
        )
```

4. Train the model on your data
```python
losses, accuracy = logReg.train(X_train, y_train)
```

5. You can calculate accuracy on test data by using `calc_accuracy` method from the classifier.
```python
logReg.calc_accuracy(X_test, y_test)
```

# References
- https://towardsdatascience.com/the-basics-logistic-regression-and-regularization-828b0d2d206c
- https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
- https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
- https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b
