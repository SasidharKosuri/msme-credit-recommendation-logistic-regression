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

```
y = df["label"].values.reshape(-1, 1)
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

🌟 What the Bias/Intercept Actually Does

The bias term allows the model to make predictions even when all features = 0.

Without a bias:
<img width="572" height="235" alt="image" src="https://github.com/user-attachments/assets/6784e290-dbfb-46c1-b181-0ba1310c6377" />
That means the model ALWAYS predicts 50% probability when features are close to zero.
This is WRONG for almost all datasets.
The bias fixes this by shifting the decision boundary up or down.

In other words,the bias allows the model to shift the sigmoid curve
With bias:
<img width="282" height="54" alt="image" src="https://github.com/user-attachments/assets/26ba1342-4107-4ee5-afde-fdb922a7cdc9" />
Here, 
w₀ is the intercept.

It shifts the sigmoid left or right so predictions fit the data.

Without w₀ → model is forced to pass through the origin.
With w₀ → model can fit real-world data.

Here, z is the z-score; i.e a raw score before applying the sigmoid function.
The sigmoid then converts z into a probability (0–1).

Suppose a row of features (after normalization + bias) looks like:
```
[1, -0.39, -0.24, -0.51, -0.41, 0.26, -0.19]
```
Suppose weights look like:
```
[w0, w1, w2, w3, w4, w5, w6]
```
Weights tell how important each feature is.
| Feature        | Meaning                 | Weight               |
| -------------- | ----------------------- | -------------------- |
| credit_score   | very important          | w₆ = large positive  |
| existing_loans | reduces approval chance | w₅ = negative        |
| revenue        | somewhat important      | w₁ = small positive  |
| profit_margin  | moderately important    | w₂ = medium positive |

Interpretation:
- Higher positive weight → increases probability of “Approve”
- Negative weight → decreases probability
- Zero weight → feature has no effect

So,Weights = what the model learns.
Then:
```
z = 1*w0
   + (-0.39)*w1
   + (-0.24)*w2
   + (-0.51)*w3
   + (-0.41)*w4
   + (0.26)*w5
   + (-0.19)*w6
```
and This gives one number, be something like:
```
z = 1.47
```
which is the z-score for that row and corresponding weights

Logistic Regression
Logistic regression is a machine-learning method used for YES/NO predictions, like:
Approve loan or Reject?
Spam email or Not?
Fraudulent transaction or Normal?

In this scenario,"Will this MSME get credit? → Yes (1) or No (0)"
<img width="467" height="289" alt="image" src="https://github.com/user-attachments/assets/b97d6403-bf43-41d5-8396-c18900e9a9c5" />

The next step is training the logistic regression model to arrive at accurate weights, we do this with the help of the train_logistic_regression function. The inputs for this function are X,y,lr and epochs
lr and epochs are two of the MOST important hyperparameters in gradient descent.
1. lr (learning rate)

This controls how big the weight updates are during training.
You update weights using:
```
weights -= lr * gradient
```
So:
If lr is large, weights change a lot each step
If lr is small, weights change slowly

In our code, lr = 0.1
0.1 is a good balance. Sinc it is:
- Stable
- Fast enough
- Works well for normalized data

If lr was 10 → model explodes
If lr was 0.0001 → training will be extremely slow

2. epochs (training cycles)

An epoch means:
One full pass over the entire dataset.

In our code:
```
epochs = 2000
```

Meaning:
- We compute predictions
- Compute gradients
- Update weights
- Repeat this 2000 times
- Also, for every 200th time we compute loss and analyze if it is gradually reducing(if the loss is reducing with increasing epochs, it means that the model is working well)
  
Each epoch improves the model a little.

Now, coming to the train_logistic_regression function:

```
np.random.seed(42)
weights = np.random.randn(X.shape[1], 1) * 0.01
```
- np.random.seed(42)

Locks NumPy's random generator so that it ALWAYS produces the same sequence of random numbers.

- np.random.randn(X.shape[1], 1)

Generates a 7×1 array of random values (because X has 7 features).

- * 0.01

Makes the weights small. We need to understand why we make the weights small: 


Derivative of the sigmoid (important!)

Sigmoid:

<img width="197" height="72" alt="image" src="https://github.com/user-attachments/assets/73505f9a-c05d-4e6d-b60e-07049eaf8869" />


Derivative of sigmoid:

<img width="227" height="47" alt="image" src="https://github.com/user-attachments/assets/d1676181-5ccb-4011-a47d-76a8199ef121" />


This measures how much the sigmoid output changes if z changes a little.

Key insight:

👉 When sigmoid output is close to 0 or 1,
its derivative becomes almost zero.

This is called sigmoid saturation.

When derivative ≈ 0 → no learning happens.

This is why we want weights to start SMALL.
Otherwise z becomes too large since z = XW→ sigmoid output becomes too close to 0 or 1 → derivative becomes tiny → learning stops.

now within the train_logistic_regression function we are using 2 important functions:
1. sigmoid function :
   <img width="197" height="72" alt="image" src="https://github.com/user-attachments/assets/73505f9a-c05d-4e6d-b60e-07049eaf8869" />
- It takes ANY number — negative, zero, or positive — and converts it into a probability between 0 and 1.
  Examples:
  
| z    | sigmoid(z) |
| ---- | ---------- |
| -100 | ~0.0000    |
| -5   | 0.0067     |
| 0    | 0.5        |
| 2    | 0.88       |
| 10   | 0.9999     |


  So sigmoid is a probability generator.
