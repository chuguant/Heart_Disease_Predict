## Run
```bash
python app.py
```

Then open http://127.0.0.1

## Design
### Part 1
Visualize the statistics for basic information(Attribute) No. 3 - 13 by groups of age and sex with appropriate charts.

There are two types of attributes:
- Categorical, e.g., chest pain type, resting electrocardiographic results...
- Numerical:, e.g., resting blood pressure, serum cholestoral in mg/dl...

For categorical attributes, I draw the bar plot for each (age, sex) group, the number of bar is corresponding the the number of categorical values, and the height of bar is corresponding to the percentage of that value in the (age, sex) group.

For numerical attributes, I draw the line plot and bar plot for each (age, sex) group with their mean values.

### Part 2
I tried three algorithms to find the potential important factors(attributes) related to heart diseases.

1. Correlation Matrix with Heatmap

Correlation states how the features are related to each other or the target variable.

Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)

Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

2. Decision Tree Feature Importance

We can get the feature importance by using the feature importance property of the decision tree model.

Feature importance gives us a score for each feature of the data, the higher the score more important or relevant is the feature towards the target attribute.

3. Statistical method

Statistical tests can be used to select those features that have the strongest relationship with the target atrribute.

The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.

## Part 3 Track 2 - Heart Disease Prediction

### Data cleaning

The dataset has some missing value marked as '?', we need to drop those records with missing values.

### Normalization

Some algorithm requires data normalization, while some others not. I compare each algorithm with unnormalized data and z-score normalized data, and then pick the best one.

### Experiments

I compare the following classification algorithms:
- Logistic Regression
- SVM
- Decision Tree
- Adaboost
- Random Forest
- KNN


Using 5-fold cross validation and some parameter search methods. I find the best model is **Random Forst** with `n_estimators=100, max_depth=2, criterion='entropy'`, the test accuracy reachs 85.60%.

The full results are as follows:

|     Algorithm      |      Accuracy     |
|--------------------|-------------------|
| LR                 | 0.8314 (+/- 0.08) |
| LR bagging         | 0.8415 (+/- 0.09) |
| LR norm            | 0.8247 (+/- 0.06) |
| SVM                | 0.6737 (+/- 0.09) |
| SVM norm           | 0.8212 (+/- 0.09) |
| Decision Tree      | 0.7235 (+/- 0.17) |
| Decision Tree norm | 0.7572 (+/- 0.13) |
| Adaboost           | 0.7739 (+/- 0.13) |
| Adaboost norm      | 0.7739 (+/- 0.13) |
| Random Forest      | 0.8560 (+/- 0.06) |
