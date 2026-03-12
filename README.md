

# 🍷 Wine Quality Prediction using Machine Learning

## 📌 Project Overview

This project builds a **Wine Quality Prediction System** using **Machine Learning**.
The goal is to classify wine as **Good Quality** or **Bad Quality** based on its chemical properties.

The dataset used is the **Red Wine Quality dataset**, and the model is trained using the **Random Forest Classifier**.

If the wine quality score is **7 or higher**, it is classified as **Good Quality**, otherwise it is classified as **Bad Quality**.

---

# 📂 Dataset Information

The dataset contains physicochemical properties of red wine samples.

### Features in the Dataset

| Feature              | Description                                  |
| -------------------- | -------------------------------------------- |
| fixed acidity        | Amount of fixed acids in wine                |
| volatile acidity     | Acetic acid amount                           |
| citric acid          | Citric acid content                          |
| residual sugar       | Sugar remaining after fermentation           |
| chlorides            | Salt content                                 |
| free sulfur dioxide  | Free SO₂ level                               |
| total sulfur dioxide | Total SO₂ level                              |
| density              | Density of wine                              |
| pH                   | Acidity level                                |
| sulphates            | Wine additive contributing to sulfur dioxide |
| alcohol              | Alcohol percentage                           |
| quality              | Quality score (0–10)                         |

---

# 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

# 🔄 Project Workflow

## 1️⃣ Importing Dependencies

Libraries used for data manipulation, visualization, and machine learning.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# 2️⃣ Data Collection

The dataset is loaded into a **Pandas DataFrame**.

```python
wine_dataset = pd.read_csv('winequality-red.csv')
```

Then the structure of the dataset is checked using:

* `shape()` → number of rows and columns
* `head()` → first few rows
* `isnull()` → missing values

---

# 3️⃣ Data Analysis & Visualization

### Distribution of Wine Quality

```python
sns.catplot(x='quality', data=wine_dataset, kind='count')
```

This shows how many wines belong to each quality level.

### Feature Relationship with Quality

Example:

```python
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
```

Used to understand how chemical properties affect wine quality.

---

# 4️⃣ Correlation Analysis

A **correlation matrix** is created to understand relationships between variables.

```python
correlation = wine_dataset.corr()
sns.heatmap(correlation)
```

The heatmap helps identify **positive and negative correlations**.

---

# 5️⃣ Data Preprocessing

The dataset is split into:

* **Features (X)** → Chemical properties
* **Label (Y)** → Wine quality

```python
X = wine_dataset.drop('quality', axis=1)
```

---

# 6️⃣ Label Binarization

Wine quality is converted into **binary classification**.

```python
Y = wine_dataset['quality'].apply(lambda y: 1 if y >= 7 else 0)
```

* **1 → Good Quality Wine**
* **0 → Bad Quality Wine**

---

# 7️⃣ Train-Test Split

The dataset is divided into training and testing data.

```python
train_test_split(X, Y, test_size=0.2, random_state=3)
```

* 80% Training Data
* 20% Testing Data

---

# 8️⃣ Model Training

The model used is **Random Forest Classifier**.

```python
model = RandomForestClassifier()
model.fit(X_train, Y_train)
```

Random Forest works by building multiple decision trees and combining their results.

---

# 9️⃣ Model Evaluation

Model performance is evaluated using **Accuracy Score**.

```python
accuracy_score(X_test_prediction, Y_test)
```

Example output:

```
Accuracy : 0.88
```

---

# 🔮 Predictive System

The trained model can predict wine quality for new input data.

Example input:

```python
input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)
```

Prediction output:

```
Good Quality Wine
```

or

```
Bad Quality Wine
```

---

# 🚀 How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 2: Download Dataset

Place the dataset file:

```
winequality-red.csv
```

in the project folder.

### Step 3: Run the Python Script

Run the code in **Jupyter Notebook** or **Python environment**.

---

# 📊 Project Results

* Random Forest Classifier successfully classifies wine quality.
* Model achieves **high accuracy (~85–90%)** depending on data split.

