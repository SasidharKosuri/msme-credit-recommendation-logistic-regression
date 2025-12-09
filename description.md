# 📌 MSME Credit Recommendation System
The credit recommendation system is created with the help of numpy,pandas and seaborn and matplotlib libraries of python.

Firstly, we take a dataset into consideration and create a dataframe out of it using pandas.
This dataset that we have taken into account consists of 7 features required inorder to predict the credit recommendation outcome.

## 📊 Features : 
1. **revenue**
How much income the business generates. Higher revenue usually means a stronger business.

2. **profit_margin**
Percentage of profit the business keeps after expenses. Higher margin = more efficient business.

3. **employees**
Number of workers in the company. Larger companies often have more stability.

4. **age**
How long the business has been running (in years). Older businesses tend to be more reliable.

5. **existing_loans**
Number of active loans the business already has. More loans = higher financial risk.

6. **credit_score**
Business’s credit rating (similar to personal credit score). Higher score = more trustworthy borrower.

7. **label**
Target:
1 = Approve (loan should be approved)
0 = Reject (loan should not be approved)

## 🔍 Dataset Analysis (EDA)
We analyze the dataset using Seaborn (Visual Explanatory Data-Analysis). We are using pair-plot and heatmap to achieve this purpose.

- pair-plot: 
Visualizes pairwise feature relationships and how they differ between approved and 
rejected MSMEs using color-coded labels.

- heatmap:
Shows the correlation between all features, helping identify important predictors 
and relationships in the dataset.

## 🔢 Splitting DataFrame into NumPy Arrays

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
Our implementation uses:
- matrix multiplication
- transpose
- dot products
- element-wise operations
- gradient descent

Pandas cannot do this kind of math reliably.
NumPy is designed for linear algebra.

## ⚖️ Normalization
the next step is normalization:

```
X = (X - X.mean(axis=0)) / X.std(axis=0)
```

We do normalization because different features have different feature scales

Example:<br>
credit_score ~ 700<br>
existing_loans ~ 0 or 1<br> 
employees ~ 40 or 120<br> 
age ~ 1–20<br>


Because of this:<br>
- Features with large numbers dominate gradient descent
- Features with small numbers get ignored

Without normalization, the model behaves like:
```
“credit_score is super important because it's HUGE”
“existing_loans is not important at all because it's tiny”
```
This is not true as it's just a scaling problem.<br>
and also because Normalization makes training MUCH faster & Logistic regression assumes features are on similar scales<br>

## ➕ Adding the Bias Column (Intercept)
the next step is adding a bias-column/intercept:<br>
```
X = np.hstack((np.ones((X.shape[0], 1)), X))
```
<br>
basically, what we are doing here is adding a horizontally stacking a column of 1's and X.We do this because the bias term allows the model to make predictions even when all features = 0.

### 🎯 What the Bias/Intercept Actually Does
---
The bias term allows the model to make predictions even when all features = 0.

### Without a bias:
 ---
<img width="572" height="235" alt="image" src="https://github.com/user-attachments/assets/6784e290-dbfb-46c1-b181-0ba1310c6377" /><br>

- That means the model ALWAYS predicts 50% probability when features are close to zero.<br>
- This is WRONG for almost all datasets.<br>
- The bias fixes this by shifting the decision boundary up or down.<br>

In other words,the bias allows the model to shift the sigmoid curve<br>

### With bias:<hr>
<img width="282" height="54" alt="image" src="https://github.com/user-attachments/assets/26ba1342-4107-4ee5-afde-fdb922a7cdc9" /><br>

 > Here, w₀ is the intercept.<br>
 > It shifts the sigmoid left or right so predictions fit the data.
 > Without w₀ → model is forced to pass through the origin.
 > With w₀ → model can fit real-world data.<br>

### Understanding z
 - Here, z is the z-score; i.e a raw score before applying the sigmoid function.
 - The sigmoid then converts z into a probability (0–1).

Suppose a row of features (after normalization + bias) looks like:<br>

```
[1, -0.39, -0.24, -0.51, -0.41, 0.26, -0.19]
```
<br>
Suppose weights look like:

```
[w0, w1, w2, w3, w4, w5, w6]
```
<br>
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

## 📈 Logistic Regression
Logistic regression is a machine-learning method used for YES/NO predictions, like:
- Approve loan or Reject?
- Spam email or Not?
- Fraudulent transaction or Normal?

In this scenario,**"Will this MSME get credit? → Yes (1) or No (0)"**<br>

<img width="467" height="289" alt="image" src="https://github.com/user-attachments/assets/b97d6403-bf43-41d5-8396-c18900e9a9c5" /><br>

## 🏋️ Training the Model
- The next step is training the logistic regression model to arrive at accurate weights, we do this with the help of the train_logistic_regression function.<br>
- The inputs for this function are X,y,lr and epochs.<br>
- lr and epochs are two of the MOST important hyperparameters in gradient descent.

 **1. lr (learning rate)**:
This controls how big the weight updates are during training.
You update weights using:
  ```
weights -= lr * gradient
  ```
So:
- If lr is large, weights change a lot each step
- If lr is small, weights change slowly

In our code, lr = 0.1
0.1 is a good balance. Since it is:
- Stable
- Fast enough
- Works well for normalized data

If lr was 10 → model explodes
If lr was 0.0001 → training will be extremely slow

**2. epochs (training cycles)**
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

## train_logistic_regression function
### 🧪 Weight Initialization
```
np.random.seed(42)
weights = np.random.randn(X.shape[1], 1) * 0.01
```
- np.random.seed(42)

Locks NumPy's random generator so that it ALWAYS produces the same sequence of random numbers.

- np.random.randn(X.shape[1], 1)

Generates a 7×1 array of random values (because X has 7 features).

- *0.01

Makes the weights small. We need to understand why we make the weights small:

Derivative of the sigmoid (important!)

**Sigmoid:**

<img width="197" height="72" alt="image" src="https://github.com/user-attachments/assets/73505f9a-c05d-4e6d-b60e-07049eaf8869" />

**Derivative of sigmoid:**

<img width="227" height="47" alt="image" src="https://github.com/user-attachments/assets/d1676181-5ccb-4011-a47d-76a8199ef121" />

This measures how much the sigmoid output changes if z changes a little.

**Key insight:**

> When sigmoid output is close to 0 or 1,
> its derivative becomes almost zero.

- This is called sigmoid saturation.
- When derivative ≈ 0 → no learning happens.
- This is why we want weights to start SMALL.
- Otherwise z becomes too large since z = XW→ sigmoid output becomes too close to 0 or 1 → derivative becomes tiny → learning stops.

## 🧮 Three Important Functions in Training
Within the train_logistic_regression function we are using 3 important functions:

**1. sigmoid function:**
   <br><img width="197" height="72" alt="image" src="https://github.com/user-attachments/assets/73505f9a-c05d-4e6d-b60e-07049eaf8869" /><br>

When z is:
- Large and positive → e^-z → becomes almost zero → result → 1
- Large and negative →  e^-z → becomes huge → result → 0
- Zero → result = 0.5
- It takes ANY number — negative, zero, or positive — and converts it into a probability between 0 and 1.
  Examples:
  
| z    | sigmoid(z) |
| ---- | ---------- |
| -100 | ~0.0000    |
| -5   | 0.0067     |
| 0    | 0.5        |
| 2    | 0.88       |
| 10   | 0.9999     |

So sigmoid is a probability generator.We need sigmoid in logistic regression because logistic regression predicts things like:
- Approve (1)
- Reject (0)

We need probabilities like:
- 0.93 → 93% chance approve
- 0.18 → 18% chance approve
- Sigmoid gives us exactly that.

**2. predictions:**<br>
It just returns the sigmoid value of the dot product of X and weights.<br>
> np.dot(X, weights)

This performs matrix multiplication:
```
z=X⋅W
```
Where:
- X is your (10, 7) matrix
- W is (7, 1) weight vector

The result is:
- z shape = (10, 1) i.e 1 z-score per MSME.
- We then apply sigmoid function to each of these 10 values and it results in 10 predictions, 1 prediction for each MSME.

**3.compute_loss:**
- takes X,y & weights array as inputs
- Loss tells us how wrong the model is.
- High loss → model makes bad predictions
- Low loss → model makes good predictions
- Training tries to reduce this loss over time.

```
m = len(y)
```
> In our code y has 10 rows (10,1) → m = 10.<br>
> This is used to compute the average loss across all MSMEs.<br>
> It then gets the prediction array for each MSME using
```
predictions = predict(X, weights)
```
then it uses the Binary Cross-Entropy formula for loss,which is given by:
```
loss = - (1/m) * np.sum(y*np.log(predictions) + (1-y)*np.log(1 - predictions))
```
i.e
<br>
<img width="408" height="67" alt="image" src="https://github.com/user-attachments/assets/a0f652ab-2c0e-47ef-92b1-92a0482f6841" /><br>

The real meaning of the loss formula:
**1. Case 1 : When y = 1** <br>
<img width="454" height="173" alt="image" src="https://github.com/user-attachments/assets/ed6dc3fe-ac8a-4f4c-8724-a24ee17dd0eb" /><br>

**2. Case 2 : When y = 0** <br>
<img width="488" height="173" alt="image" src="https://github.com/user-attachments/assets/b2b7fb8a-22dd-4d88-81ca-c35c90f4f137" /><br>

**Why negative sign?**
Logs are negative numbers (log ≤ 0).So:
- Small negative (good prediction)
- After multiplying by -1 → becomes small positive loss
- Big negative (bad prediction)
- After multiplying by -1 → becomes large positive loss
- Loss should always be positive.

**Why take the mean? (1/m)**
We average the loss across all 10 samples.
This makes the scale consistent regardless of dataset size.

**Why do we multiply y with log(pred) and (1-y) with log(1-p)?**

```
If loan should be approved (y=1)
We only care about: log(pred)

→ wants pred to be high
→ punishes model if pred is low
```
If loan should be rejected (y=0)
```
We only care about: log(1 - pred)

→ wants pred to be low so that 1-pred will be high
→ punishes model if pred is high
```
The formula is built so that:
- Wrong predictions → big loss
- Correct predictions → small loss

this is based on the understanding that if pred tends to 1 log(pred) moves to 0, i.e loss will be nearly 0.
and if pred tends to 0, log(pred) tends to large negative values, implies log(1-pred) tends to 0 <br>

<img width="346" height="124" alt="image" src="https://github.com/user-attachments/assets/8ade2b92-4748-4edf-b6dc-bbb7adbc18ec" /><br>

in the range of epochs,we calculate predictions followed by gradient.
```
gradient = np.dot(X.T, (predictions - y)) / len(y)
```
✔ Step A — Compute error
```
(predictions - y)
```
For each sample:
If prediction too high → positive error
If prediction too low → negative error

Example:
If true = 1 and pred = 0.7 → error = -0.3
If true = 0 and pred = 0.8 → error = +0.8

This error tells us how wrong each prediction is.
✔ Step B — Multiply error with X.T

X.T shape is (7, 10)

Why transpose?
So each weight gets updated using all samples.

This dot product does:
<img width="238" height="85" alt="image" src="https://github.com/user-attachments/assets/e39e1d4c-86e2-40ce-8a61-ed6f65275f13" />
Meaning:

“For each weight j, compute how much it contributed to the error.”
This produces a 7×1 vector — one gradient for each weight.

✔ Step C — Divide by number of samples
```
/ len(y)
```
This gives the average gradient (better stability).

✔ Step D - weights -= lr * gradient

This is the actual learning step.

 We move weights in the opposite direction of the gradient:
```
weights = weights - (learning_rate × gradient)
```

Because:

If gradient is positive → prediction too high → reduce weight
If gradient is negative → prediction too low → increase weight

This is exactly how gradient descent works.

Also, we calculate loss for every 200 epochs and analyze if its decreasing.
return weights in the end.

Now the final step is predicting the probability and Approval/Rejection outcomw for a new MSME when the training is done.
We do this with the help of the recommend_credit function.

lets say this is the sample data for which we need to predict
```
sample = [75, 18, 15, 6, 0, 710]
decision, probability = recommend_credit(sample)
```

the sample is the input here and the recommend_credit function takes it as a parameter (new_data)
We then convert the new_data python list to a numpy array

For the next step we perform normalization with the help of our previously calculated dataframe X
```
new_data = (new_data - df.drop("label", axis=1).mean().values) / \
                df.drop("label", axis=1).std().values
```
This line:

Takes the same columns used for training (revenue, profit_margin, etc.)
Computes their mean and std from the original df
Applies the z-score normalization:

Why?

Because during training we did:

```
X = (X - X.mean(axis=0)) / X.std(axis=0)
```

<img width="217" height="53" alt="image" src="https://github.com/user-attachments/assets/1f99dc7b-cbe6-4251-9003-fc10971bd5ad" />

So the model learned on normalized features.
At prediction time, you MUST feed values in the same scale, otherwise the weights don’t make sense.

Then we add the bias term (intercept),
We insert 1 at index 0, so X has 7 values now the first one is the bias and rest are feature values.

Next, we compute the approval probability
```
prob = sigmoid(np.dot(new_data, weights))[0]

```

Decoding it:
```
np.dot(new_data, weights)
```
new_data → shape (7,)
weights → shape (7,1)
Result → shape (1,) (a single z value wrapped in an array)

This is:

sigmoid(z) turns that raw score into a probability between 0 and 1.
[0] just extracts the scalar from the array.
So if prob is something like 0.87 → 87% chance this MSME should be approved.

Now for the last step, we turn probability into decision
```
return "Approve" if prob >= 0.5 else "Reject", prob
```

If prob >= 0.5 → "Approve"
Else → "Reject"
So the function returns either of two things:

("Approve", probabilty of approval)
# or
("Reject", the probabilty of approval)
```
eg :
("Approve", 0.8734)
# or
("Reject", 0.2412)
```
We then print these as decision and probability respectively.
