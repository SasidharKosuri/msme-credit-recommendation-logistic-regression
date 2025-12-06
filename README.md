# 📌 MSME Credit Recommendation System (Logistic Regression from Scratch)

This project implements a credit approval prediction system for Micro, Small, and Medium Enterprises (MSMEs) using Logistic Regression built entirely from scratch — without using scikit-learn.
It uses NumPy, Pandas, Matplotlib, and Seaborn, along with manual Gradient Descent, to train a logistic regression classifier that predicts whether a business should be approved for a loan based on key financial and operational features.

---

## 🚀 Features

- Logistic Regression implemented manually
- Feature normalization (z-score standardization)
- Bias term handling
- Custom sigmoid, prediction, and loss functions
- Gradient Descent optimization
- Heatmaps & visualizations using Seaborn/Matplotlib
- Real-time credit approval recommendation function
- Clean, minimal, reproducible code
- Great for learning, interviews, and portfolio projects

## 📊 Dataset Features

The model uses the following MSME attributes:
- revenue
- profit_margin
- employees
- age
- existing_loans
- credit_score

### Target label:
- 1 → Approve
- 0 → Reject

## 🧮 Technologies Used

- Python
- NumPy – numerical computations
- Pandas – data handling
- Matplotlib – plotting
- Seaborn – heatmaps & EDA visualizations

## 🖥️ Steps to Run Locally
- Follow these steps to run the MSME Credit Recommendation System on your local machine:
- 
  ### 1️⃣ Clone the Repository
  
'''
git clone https://github.com/SasidharKosuri/msme-credit-recommendation-logistic-regression.git
cd msme-credit-recommendation-logistic-regression
'''

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

On Windows:

'''
python -m venv venv
venv\Scripts\activate
'''

On macOS/Linux:
'''
python3 -m venv venv
source venv/bin/activate
'''

### 3️⃣ Install Dependencies

'''
pip install numpy pandas matplotlib seaborn
'''

No scikit-learn required — the logistic regression model is implemented entirely from scratch.

### 4️⃣ Run the Script or Notebook

If you're using Jupyter/Colab Notebook (.ipynb):
'''
jupyter notebook
'''
Then open your notebook file and run all cells.

If you're using a Python script (.py):
'''
python main.py
'''
Or whatever your script name is.

### 5️⃣ Test the Recommendation System

At the bottom of the script/notebook, you can test with a new MSME input:
'''
sample = [75, 18, 15, 6, 0, 710]
decision, probability = recommend_credit(sample)

print("Decision:", decision)
print("Approval Probability:", probability)
'''

You’ll see output such as:
'''
Decision: Approve
Approval Probability: 0.87
'''
