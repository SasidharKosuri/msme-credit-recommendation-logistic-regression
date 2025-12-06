The credit recommendation system is created with the help of numpy,pandas and seaborn and matplotlib libraries of python.

Firstly, we take a dataset into consideration and create a dataframe out of it using pandas.
This dataset that we have taken into account consists of 7 features required inorder to predict the credit recommendation outcome.

These 7 features are : 
1. revenue
How much income the business generates. Higher revenue usually means a stronger business.

2. profit_margin
Percentage of profit the business keeps after expenses. Higher margin = more efficient business.

3. employees
Number of workers in the company. Larger companies often have more stability.

4. age
How long the business has been running (in years). Older businesses tend to be more reliable.

5. existing_loans
Number of active loans the business already has. More loans = higher financial risk.

6. credit_score
Business’s credit rating (similar to personal credit score). Higher score = more trustworthy borrower.

7. label
Target:
1 = Approve (loan should be approved)
0 = Reject (loan should not be approved)

Then we analyze the dataset using Seaborn (Visual Explanatory Data-Analysis). We are using pair-plot and heatmap to achieve this purpose.

pair-plot: 
Visualizes pairwise feature relationships and how they differ between approved and 
rejected MSMEs using color-coded labels.

heatmap:
Shows the correlation between all features, helping identify important predictors 
and relationships in the dataset.

Then we split the dataframe into numpy arrays X and y
```
X = df.drop("label", axis=1).values
````
Take all columns except label → convert to NumPy
→ X becomes your feature matrix (10×6)

```y = df["label"].values.reshape(-1, 1)
```

Take ONLY the label column → convert to NumPy → reshape
→ y becomes your target vector (10×1)

We convert the dataframe to numpy arrays because Logistic regression math requires arrays, not DataFrames.
our implementation uses:
- matrix multiplication
- transpose
- dot products
- element-wise operations
- gradient descent

Pandas cannot do this kind of math reliably.
NumPy is designed for linear algebra.

the next step is normalization:

```
X = (X - X.mean(axis=0)) / X.std(axis=0)
```

We do normalization because different features have different feature scales

Example:
credit_score ~ 700
existing_loans ~ 0 or 1
employees ~ 40 or 120
age ~ 1–20


Because of this:
👉 Features with large numbers dominate gradient descent
👉 Features with small numbers get ignored

Without normalization, the model behaves like:
```
“credit_score is super important because it's HUGE”
“existing_loans is not important at all because it's tiny”
```
This is not true — it's just a scaling problem.

and also because Normalization makes training MUCH faster & Logistic regression assumes features are on similar scales

the next step is adding a bias-column/intercept:
```
X = np.hstack((np.ones((X.shape[0], 1)), X))
```
basically, what we are doing here is adding a horizontally stacking a column of 1's and X.We do this because the bias term allows the model to make predictions even when all features = 0.

