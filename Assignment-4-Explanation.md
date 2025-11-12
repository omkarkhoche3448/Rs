# Assignment 4: Breast Cancer Survival Prediction - Complete Explanation

## Overview
This code predicts whether breast cancer patients will **survive or die** based on medical data. It's a **binary classification problem** using machine learning.

**3 Algorithms Used:**
1. **Logistic Regression** - Simple statistical model
2. **Random Forest Classifier** - Ensemble of decision trees
3. **XGBoost Classifier** - Advanced gradient boosting

**Dataset:** SEER (Surveillance, Epidemiology, and End Results) Breast Cancer Dataset

---

## Step-by-Step Breakdown

### 1. Importing Libraries
**File:** [rs-ass4.ipynb:1](rs-ass4.ipynb#cell-0) (Lines 1-11)

```python
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

#### What Each Import Does:

**Data Handling:**
- `pandas` - Load and manipulate datasets
- `os` - File system operations

**Preprocessing:**
- `train_test_split` - Split data into training and testing sets
- `OneHotEncoder` - Convert categorical data (text) to numbers
- `StandardScaler` - Normalize numerical features to same scale
- `ColumnTransformer` - Apply different preprocessing to different column types
- `Pipeline` - Chain preprocessing steps with model training

**Models (Classifiers):**
- `LogisticRegression` - Simple linear classification model
- `RandomForestClassifier` - Ensemble of decision trees
- `XGBClassifier` - Gradient boosting (advanced technique)

**Evaluation Metrics:**
- `accuracy_score` - Overall correctness percentage
- `precision_score` - How trustworthy are positive predictions?
- `recall_score` - How many actual positives did we catch?
- `f1_score` - Balanced metric combining precision and recall

---

### 2. Loading the Dataset
**File:** [rs-ass4.ipynb:1](rs-ass4.ipynb#cell-0) (Lines 16-17)

```python
csv_path = "/kaggle/input/seer-breast-cancer-data/SEER Breast Cancer Dataset .csv"
df = pd.read_csv(csv_path)
```

#### Dataset Information:

**Total Records:** 4,024 breast cancer patients

**Columns (16 total):**

**Patient Demographics:**
1. **Age** - Patient age in years (e.g., 45, 62, 78)
2. **Race** - White, Black, Asian, Other
3. **Marital Status** - Married, Single, Divorced, Widowed, Separated

**Tumor Characteristics:**
4. **T Stage** - Tumor size/extent (T1, T2, T3, T4)
   - T1: Small tumor (≤2cm)
   - T2: Medium tumor (2-5cm)
   - T3: Large tumor (>5cm)
   - T4: Tumor invading chest wall/skin
5. **Tumor Size** - Exact size in millimeters
6. **Grade** - How abnormal cancer cells look (1, 2, 3, 4)
   - Grade 1: Well differentiated (slow growing)
   - Grade 2: Moderately differentiated
   - Grade 3: Poorly differentiated (fast growing)
   - Grade 4: Undifferentiated (very aggressive)

**Cancer Spread:**
7. **N Stage** - Lymph node involvement (N1, N2, N3)
   - N0: No lymph node spread
   - N1: 1-3 lymph nodes
   - N2: 4-9 lymph nodes
   - N3: 10+ lymph nodes
8. **6th Stage** - Overall cancer stage (I, II, III, IV)
   - Stage I: Early, localized
   - Stage II: Locally advanced
   - Stage III: Regional spread
   - Stage IV: Distant metastasis
9. **A Stage** - Regional vs Distant spread
10. **Regional Node Examined** - Number of lymph nodes checked (0-60)
11. **Reginol Node Positive** - Number that tested positive for cancer (0-40)

**Hormone Receptors:**
12. **Estrogen Status** - Positive or Negative
    - Positive: Cancer cells have estrogen receptors
    - Negative: No estrogen receptors
13. **Progesterone Status** - Positive or Negative
    - Similar to estrogen status

**Outcome:**
14. **Survival Months** - How long patient survived
15. **Status** - **TARGET VARIABLE: "Alive" or "Dead"**

**Junk Column:**
16. **Unnamed: 3** - Empty column (will be removed)

---

### 3. Data Cleaning
**File:** [rs-ass4.ipynb:3](rs-ass4.ipynb#cell-2), [rs-ass4.ipynb:5](rs-ass4.ipynb#cell-4), [rs-ass4.ipynb:6](rs-ass4.ipynb#cell-5) (Lines 1-4)

#### Step 1: Remove Junk Column
```python
df.drop(columns=["Unnamed: 3"], inplace=True)
```
- Removes empty/useless column

#### Step 2: Clean Column Names
```python
df.columns = df.columns.str.strip()
```
- Removes leading/trailing spaces from column names
- Example: `" Age "` → `"Age"`

#### Step 3: Prepare Features and Target
```python
df = df.dropna(subset=["Status"])
y = df["Status"].map({"Dead": 1, "Alive": 0}).astype(int)
X = df.drop(columns=["Status"])
```

**What it does:**
1. **Remove missing values:** Drop rows where Status is missing (can't predict without target)
2. **Convert target to numbers:**
   - "Dead" → 1 (positive class - what we're trying to detect)
   - "Alive" → 0 (negative class)
3. **Separate features from target:**
   - **X** = All columns except Status (predictors)
   - **y** = Status column (what we're predicting)

**Final dataset shape:** 4,024 patients × 15 features

---

### 4. Identifying Column Types
**File:** [rs-ass4.ipynb:6](rs-ass4.ipynb#cell-5) (Lines 6-11)

```python
num_cols = ["Age", "Tumor Size", "Regional Node Examined",
            "Reginol Node Positive", "Survival Months"]

cat_cols = ["Race", "Marital Status", "T Stage", "N Stage", "6th Stage",
            "Grade", "A Stage", "Estrogen Status", "Progesterone Status"]
```

#### Why Separate Column Types?

Machine learning models only work with numbers, but we have two types of data:

**Numerical Columns (5 features):**
- Continuous values with meaningful magnitude
- Examples:
  - Age: 25, 45, 65, 85
  - Tumor Size: 10mm, 35mm, 80mm
  - Regional Node Examined: 5, 15, 30
  - Reginol Node Positive: 0, 3, 10
  - Survival Months: 12, 36, 60

**Need:** Scaling (normalize ranges so no feature dominates)

**Categorical Columns (9 features):**
- Discrete categories without numerical meaning
- Examples:
  - Race: White, Black, Asian
  - T Stage: T1, T2, T3, T4
  - Grade: 1, 2, 3, 4
  - Estrogen Status: Positive, Negative

**Need:** Encoding (convert text/categories to numbers)

**Different preprocessing for different types!**

---

### 5. Preprocessing Pipeline
**File:** [rs-ass4.ipynb:6](rs-ass4.ipynb#cell-5) (Lines 13-21)

```python
numeric_pipe   = Pipeline([("scaler", StandardScaler())])
categorical_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

pre = ColumnTransformer(
        [("num", numeric_pipe, num_cols),
         ("cat", categorical_pipe, cat_cols)])
```

---

#### Numerical Pipeline: StandardScaler

**What is StandardScaler?**
- Normalizes numerical features to have **mean = 0** and **standard deviation = 1**
- Formula: `scaled_value = (original_value - mean) / std`

**Why is it needed?**

Without scaling:
```
Age: 25-95 (range: 70)
Tumor Size: 1-140 (range: 139)
Survival Months: 1-107 (range: 106)
```

Problems:
- Tumor Size has much larger values → Model thinks it's more important
- Features with larger ranges dominate the learning
- Distances become skewed

After scaling (all features have comparable ranges):
```
Age: -1.5 to 2.0 (mean=0, std=1)
Tumor Size: -1.5 to 2.0 (mean=0, std=1)
Survival Months: -1.5 to 2.0 (mean=0, std=1)
```

**Example Calculation:**
```
Age column: [25, 45, 65, 85]
Mean = 55
Std = 24.5

Scaled values:
25 → (25-55)/24.5 = -1.22
45 → (45-55)/24.5 = -0.41
65 → (65-55)/24.5 =  0.41
85 → (85-55)/24.5 =  1.22
```

Now all features are on the same scale!

---

#### Categorical Pipeline: OneHotEncoder

**What is OneHotEncoder?**
- Converts categorical variables into binary (0/1) columns
- Each unique category becomes its own column

**Why is it needed?**
Machine learning can't understand text directly:
- Can't do math on "White", "Black", "Asian"
- Can't calculate: "White" + "Black" = ?

**How it works:**

**Before encoding:**
```
Race
White
Black
Asian
White
Other
```

**After one-hot encoding:**
```
Race_White | Race_Black | Race_Asian | Race_Other
    1      |     0      |     0      |     0
    0      |     1      |     0      |     0
    0      |     0      |     1      |     0
    1      |     0      |     0      |     0
    0      |     0      |     0      |     1
```

**Each row has exactly one "1" (mutually exclusive categories)**

**Another Example - T Stage:**
```
T Stage → T_Stage_T1 | T_Stage_T2 | T_Stage_T3 | T_Stage_T4
T2      →      0     |     1      |     0      |     0
T1      →      1     |     0      |     0      |     0
T4      →      0     |     0      |     0      |     1
```

**Parameter:**
- `handle_unknown="ignore"`
  - If test data has a new category not seen in training (e.g., new race)
  - Don't crash - just ignore it (set all race columns to 0)

---

#### ColumnTransformer: Combining Pipelines

**What it does:**
- Applies **different preprocessing** to different column types
- Like having two assembly lines in a factory:
  - Line 1: Numerical columns → StandardScaler
  - Line 2: Categorical columns → OneHotEncoder
- Combines both outputs into one feature matrix

**Visual Representation:**
```
Original Data:
Age | Race  | Tumor Size | T Stage
65  | White |    35      |   T2
45  | Black |    50      |   T3

After ColumnTransformer:
Age(scaled) | Race_White | Race_Black | Tumor(scaled) | T_T2 | T_T3
   0.5      |     1      |     0      |     -0.3      |  1   |  0
  -0.5      |     0      |     1      |      0.8      |  0   |  1
```

All numerical columns are scaled, all categorical are one-hot encoded!

---

### 6. Train/Test Split
**File:** [rs-ass4.ipynb:6](rs-ass4.ipynb#cell-5) (Lines 23-25)

```python
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
```

#### What it does:

**Splits data into two sets:**
- **Training set (80%):** 3,219 patients - Used to teach the models
- **Test set (20%):** 805 patients - Used to evaluate performance

**Why split?**
- Can't evaluate on the same data we trained on (would be cheating!)
- Like studying with a textbook, then taking a different exam
- Test set simulates "unseen" future patients

#### Important Parameters:

**1. test_size=0.20**
- 20% of data reserved for testing
- 80% used for training
- Common ratios: 80/20 or 70/30

**2. random_state=42**
- Makes the split reproducible
- Same random split every time you run the code
- Any number works (42 is conventional)

**3. stratify=y** ← **Very Important!**

**What is stratification?**
- Maintains the **same class distribution** in train and test sets
- Critical for imbalanced datasets

**Example without stratification:**
```
Full dataset: 80% Alive, 20% Dead

Random split might give:
Train: 90% Alive, 10% Dead  ← Too few deaths to learn from!
Test:  70% Alive, 30% Dead  ← Test is harder than training data!

Model trained on mostly alive patients, tested on many dead → Poor performance!
```

**Example WITH stratification:**
```
Full dataset: 80% Alive, 20% Dead

Stratified split:
Train: 80% Alive, 20% Dead  ← Same distribution
Test:  80% Alive, 20% Dead  ← Same distribution

Fair comparison!
```

**In this dataset:**
- If 75% alive, 25% dead overall
- Both train and test will have 75% alive, 25% dead
- Ensures representative samples

---

### 7. Three Classification Models
**File:** [rs-ass4.ipynb:6](rs-ass4.ipynb#cell-5) (Lines 27-33)

```python
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=42)
}
```

---

## Model 1: Logistic Regression

```python
LogisticRegression(max_iter=1000, class_weight="balanced")
```

### What is Logistic Regression?

**Simple Definition:**
- A **linear model** for binary classification (yes/no, dead/alive)
- Despite the name "regression", it's used for **classification**
- Draws a line (or hyperplane) to separate two classes

### How it Works:

**Step 1: Calculate weighted sum**
```
z = w1×Age + w2×TumorSize + w3×Grade + ... + bias
```
- Each feature gets a weight (importance)
- Bias = constant adjustment

**Step 2: Apply sigmoid function**
```
Probability(Dead) = 1 / (1 + e^(-z))
```
- Sigmoid "squashes" z into probability between 0 and 1

**Step 3: Make prediction**
```
If P(Dead) > 0.5 → Predict "Dead" (class 1)
If P(Dead) ≤ 0.5 → Predict "Alive" (class 0)
```

### Visual Example:

**Simple 2D case (Age vs Tumor Size):**
```
Tumor Size
    ↑
100 |     x  x          (x = Dead patients)
 80 |   x  x  x
 60 | o  x  x           Decision boundary (line)
 40 | o o  x              ↓
 20 | o o o            (o = Alive patients)
    |________________→ Age
    20  40  60  80
```

Logistic Regression finds the line that best separates x's from o's.

### Parameters Explained:

**max_iter=1000**
- Maximum number of training iterations
- More iterations = more time for model to converge
- 1000 is usually enough for small datasets

**class_weight="balanced"** ← **Critical for Imbalanced Data**

**Problem:**
```
Dataset: 80% Alive, 20% Dead

Without balancing:
Model learns: "Just predict Alive for everyone!"
Accuracy: 80% (but useless - never catches deaths!)
```

**Solution:**
```
class_weight="balanced"

Gives higher weight to minority class (Dead)
Penalty for misclassifying Dead patients is 4× higher
Forces model to pay attention to rare class
```

**How it works:**
- Alive (majority): weight = 0.625
- Dead (minority): weight = 2.5
- Model cares 4× more about getting Dead predictions right!

### Pros and Cons:

**Pros:**
- Fast training and prediction
- Interpretable (can see feature importance from weights)
- Works well when data is linearly separable
- Provides probability scores (not just yes/no)
- Good baseline model

**Cons:**
- Assumes linear relationship (can't capture complex patterns)
- Struggles with non-linear boundaries
- Limited expressiveness compared to tree-based models

---

## Model 2: Random Forest Classifier

```python
RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
```

### What is Random Forest?

**Simple Definition:**
- An **ensemble** of many decision trees
- Each tree "votes" on the prediction
- Final prediction = **Majority vote** from all trees

**Analogy:**
- Like asking 300 doctors for their opinion
- Each doctor (tree) looks at different aspects of the patient
- Final diagnosis = What most doctors agree on
- **Wisdom of the crowd**

### How it Works:

**Step 1: Build 300 decision trees**
- Each tree trained on a **random subset** of patients (bootstrapping)
- Each tree uses a **random subset** of features
- This randomness makes trees different (diverse opinions)

**Step 2: Each tree makes a prediction**
```
Tree 1: Dead
Tree 2: Alive
Tree 3: Dead
...
Tree 300: Dead
```

**Step 3: Majority vote**
```
Results: 180 trees say "Dead", 120 say "Alive"
Final prediction: Dead (majority wins)
```

**Confidence:**
```
Probability(Dead) = 180/300 = 60%
```

### What is a Decision Tree?

**A tree of if/else decisions:**

```
                    Tumor Size > 50mm?
                    /               \
                  Yes                No
                  /                   \
        Grade 3 or 4?           Estrogen Positive?
        /          \              /              \
      Yes          No           Yes              No
      /            \             /                \
   DEAD          Age>70?      ALIVE            DEAD
               /      \
             Yes      No
             /        \
          DEAD      ALIVE
```

**Example prediction path:**
```
Patient: Tumor=65mm, Grade=3, Age=55

1. Tumor Size > 50mm? YES → Go left
2. Grade 3 or 4? YES → Go left
3. Prediction: DEAD
```

### Why 300 Trees?

**Single tree problems:**
- Overfits to training data (memorizes noise)
- Small changes in data → completely different tree
- Unstable

**Many trees (ensemble):**
- Average out errors and randomness
- More robust and stable
- Reduces overfitting

### Parameters Explained:

**n_estimators=300**
- Build 300 decision trees
- More trees = better performance (to a point)
- More trees = longer training time
- 300 is a good balance

**random_state=42**
- Controls randomness (reproducible results)
- Same trees every time you run the code

**class_weight="balanced"**
- Same as Logistic Regression
- Handles imbalanced classes
- Gives more importance to minority class (Dead)

### Bootstrapping and Feature Randomness:

**Bootstrapping (random samples):**
```
Original data: 3,219 patients

Tree 1 sees: Random 3,219 patients WITH replacement
  [Patient 5, Patient 2, Patient 5, Patient 100, ...]
  (Some patients appear multiple times, some not at all)

Tree 2 sees: Different random 3,219 patients
  [Patient 300, Patient 1, Patient 1, Patient 50, ...]
```

**Feature randomness:**
```
All features: Age, TumorSize, Grade, ... (15 features after encoding)

Tree 1 considers: Random subset (e.g., √15 ≈ 4 features at each split)
  [Age, Grade, TumorSize, EstrogenStatus]

Tree 2 considers: Different random subset
  [TumorSize, NStage, Race, ProgesteroneStatus]
```

This ensures trees are different (decorrelated) → better ensemble!

### Pros and Cons:

**Pros:**
- Handles non-linear relationships well
- No need for feature scaling (tree-based)
- Automatically captures feature interactions
- Resistant to overfitting (averaging many trees)
- Works with mixed data types (numerical + categorical)
- Feature importance scores

**Cons:**
- Slower than Logistic Regression
- Less interpretable (300 trees to explain!)
- Larger model size (memory)
- Can overfit on very noisy data

---

## Model 3: XGBoost Classifier

```python
XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
              random_state=42)
```

### What is XGBoost?

**Simple Definition:**
- **eXtreme Gradient Boosting**
- Builds trees **sequentially**, each fixing mistakes of previous trees
- One of the most powerful ML algorithms (wins many competitions!)

**Analogy:**
- Like a student learning from mistakes on practice exams
- Exam 1: Make predictions, see what you got wrong
- Exam 2: Focus on questions you missed in Exam 1
- Exam 3: Focus on remaining mistakes from Exams 1+2
- Final grade = Weighted combination of all attempts

### How it Works:

**Sequential Learning:**

**Tree 1:**
```
Makes initial predictions
Error: Wrongly classifies 30% of deaths
```

**Tree 2:**
```
Focuses on the 30% Tree 1 got wrong
Learns: "Oh, patients with Grade 4 + large tumors are high risk"
Adds corrections to Tree 1's predictions
Combined error: 18%
```

**Tree 3:**
```
Focuses on remaining 18% errors
Learns: "Patients with many positive nodes are also high risk"
Adds more corrections
Combined error: 10%
```

**... continues for 300 trees**

**Final prediction:**
```
Prediction = 0.05×Tree1 + 0.05×Tree2 + 0.05×Tree3 + ... + 0.05×Tree300

(0.05 = learning_rate, controls how much each tree contributes)
```

### Random Forest vs XGBoost:

| Feature | Random Forest | XGBoost |
|---------|---------------|---------|
| **Tree Building** | Parallel (all at once) | Sequential (one after another) |
| **Tree Independence** | Independent | Each tree depends on previous |
| **Learning Strategy** | Equal vote | Learn from mistakes |
| **Final Prediction** | Majority vote | Weighted sum |
| **Speed** | Faster training | Slower but optimized |
| **Performance** | Good | Usually better |
| **Overfitting Risk** | Lower | Higher (needs tuning) |

### Parameters Explained:

**n_estimators=300**
- Build 300 sequential trees
- More trees = better learning but slower
- Too many trees can overfit

**learning_rate=0.05** ← **Very Important!**
- Controls how much each tree contributes to final prediction
- Range: 0 to 1
  - **0.05 (5%):** Conservative, slow learning, more trees needed
  - **0.1 (10%):** Moderate (common default)
  - **0.3 (30%):** Aggressive, fast learning, fewer trees needed

**Lower learning rate = Better performance + Less overfitting BUT needs more trees**

**Example:**
```
learning_rate=0.3, n_estimators=100 ≈ learning_rate=0.05, n_estimators=600

Slower learning (0.05) is more careful and usually better!
```

**max_depth=4**
- Maximum depth of each tree (number of levels)
- Limits tree complexity

**Example:**
```
Depth 1:
  Root → [Left, Right]

Depth 2:
  Root → [Left, Right]
          ↓       ↓
       [L,R]   [L,R]

Depth 4:
  16 leaf nodes (2^4)
```

- **Shallow trees (4):** Simple, prevent overfitting, generalize better
- **Deep trees (10+):** Complex, might overfit, memorize training data

**max_depth=4 is a good conservative choice**

**subsample=0.8**
- Use 80% random sample of training data for each tree
- Like Random Forest's bootstrapping

**Why?**
- Reduces overfitting (each tree doesn't see all data)
- Adds randomness (trees are different)
- Faster training

**colsample_bytree=0.8**
- Use 80% random features for each tree
- Similar to Random Forest's feature randomness

**Why?**
- Decorrelates trees (makes them different)
- Reduces overfitting
- Some features might be noisy - random sampling helps

**eval_metric="logloss"**
- Evaluation metric during training
- **Log Loss (Cross-Entropy):** Measures prediction probability quality
  - Penalizes confident wrong predictions heavily
  - Good for classification

**random_state=42**
- Reproducible results

### Gradient Boosting Math (Simplified):

**Goal:** Minimize loss function

**Loss = Prediction Error**

**Step 1: Initial prediction**
```
pred_0 = average of all training labels
```

**Step 2: Calculate errors (residuals)**
```
errors_1 = true_labels - pred_0
```

**Step 3: Build tree to predict errors**
```
Tree_1 learns to predict errors_1
```

**Step 4: Update predictions**
```
pred_1 = pred_0 + learning_rate × Tree_1
```

**Step 5: Repeat**
```
errors_2 = true_labels - pred_1
Tree_2 learns to predict errors_2
pred_2 = pred_1 + learning_rate × Tree_2

... continues for 300 trees
```

**Each tree fixes the mistakes of the previous ensemble!**

### Pros and Cons:

**Pros:**
- **Best performance** (usually beats Random Forest)
- Handles complex patterns and interactions
- Built-in regularization (subsample, colsample, max_depth)
- Feature importance
- Highly optimized (fast inference)
- Industry standard for tabular data

**Cons:**
- Requires careful hyperparameter tuning
- Can overfit if not tuned properly
- Slower training than Random Forest
- Less interpretable
- More sensitive to noise

---

### Model Comparison Summary:

| Aspect | Logistic Regression | Random Forest | XGBoost |
|--------|---------------------|---------------|---------|
| **Type** | Linear model | Ensemble (parallel) | Ensemble (sequential) |
| **Complexity** | Simple | Moderate | Complex |
| **Training Speed** | Fast | Moderate | Slow |
| **Interpretability** | High | Low | Low |
| **Non-linearity** | No | Yes | Yes |
| **Feature Scaling** | Required | Not required | Not required |
| **Overfitting Risk** | Low | Low | Moderate |
| **Performance** | Baseline | Good | Best |
| **Best For** | Linear problems | Robust baseline | Maximum accuracy |

---

### 8. Training & Evaluation
**File:** [rs-ass4.ipynb:6](rs-ass4.ipynb#cell-5) (Lines 35-45)

```python
for name, model in models.items():
    pipe = Pipeline([("prep", pre), ("clf", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print(f"\n{name}  –  test metrics")
    print("Accuracy :", round(accuracy_score(y_test, preds), 3))
    print("Precision:", round(precision_score(y_test, preds), 3))
    print("Recall   :", round(recall_score(y_test, preds), 3))
    print("F1-score :", round(f1_score(y_test, preds), 3))
```

### What This Code Does:

**For each of the 3 models:**

**Step 1: Create Pipeline**
```python
pipe = Pipeline([("prep", pre), ("clf", model)])
```
- Links preprocessing with the model
- Data flows: Raw data → Preprocessing → Model → Predictions

**Step 2: Train Model**
```python
pipe.fit(X_train, y_train)
```
- Learns patterns from 3,219 training patients
- Preprocessing + model training in one step

**Step 3: Make Predictions**
```python
preds = pipe.predict(X_test)
```
- Predicts "Alive" (0) or "Dead" (1) for 805 test patients
- Tests on unseen data

**Step 4: Calculate Metrics**
- Compares predictions vs actual outcomes
- Reports 4 different metrics

---

## Evaluation Metrics - Detailed Explanation

### Understanding the Confusion Matrix

**First, let's understand what happens when we make predictions:**

```
                        PREDICTED
                    Alive  |  Dead
        ______________|_________|________
ACTUAL  Alive  |     TN    |    FP    |
        Dead   |     FN    |    TP    |
               |___________|__________|
```

**Four possible outcomes:**

1. **TP (True Positive):** Predicted Dead, Actually Dead ✓
   - Correctly identified death
   - Good outcome!

2. **TN (True Negative):** Predicted Alive, Actually Alive ✓
   - Correctly identified survival
   - Good outcome!

3. **FP (False Positive):** Predicted Dead, Actually Alive ✗
   - **Type I Error**
   - False alarm
   - Patient survives but we scared them unnecessarily

4. **FN (False Negative):** Predicted Alive, Actually Dead ✗
   - **Type II Error**
   - Missed death
   - Patient dies but we told them they'd be fine
   - **Most dangerous error in medical diagnosis!**

---

### Metric 1: Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct Predictions / Total Predictions
```

**What it means:**
- **What percentage of all predictions are correct?**
- Overall correctness rate

**Results:**
```
Logistic Regression: 0.806 (80.6%)
Random Forest:       0.896 (89.6%)
XGBoost:             0.916 (91.6%) ← Best!
```

**Interpretation:**
- XGBoost correctly predicts 91.6% of patients
- Gets 738 out of 805 predictions right

**Example Calculation (hypothetical XGBoost):**
```
Test set: 805 patients
Correctly predicted: 738
Incorrectly predicted: 67

Accuracy = 738/805 = 0.916 (91.6%)
```

### The Accuracy Paradox:

**Why accuracy alone is misleading:**

**Scenario:**
```
Dataset: 90% Alive, 10% Dead

Dumb Model: Always predict "Alive"
Results:
- Correctly predicts all Alive patients (90%)
- Misses all Dead patients (10%)
- Accuracy = 90% ← Looks great!
- But completely useless (never catches deaths!)
```

**This is why we need other metrics...**

---

### Metric 2: Precision

**Formula:**
```
Precision = TP / (TP + FP)
          = True Positives / All Positive Predictions
```

**What it means:**
- **Of all patients we predicted would die, how many actually died?**
- How trustworthy are our positive (Dead) predictions?
- Measures **false alarm rate**

**Results:**
```
Logistic Regression: 0.421 (42.1%)
Random Forest:       0.810 (81.0%)
XGBoost:             0.867 (86.7%) ← Best!
```

**Interpretation:**

**Logistic Regression:**
```
Predicts 100 patients will die
Reality: Only 42 actually died, 58 survived (false alarms!)
Precision = 42/100 = 42.1%

Problem: Too many false alarms!
```

**XGBoost:**
```
Predicts 100 patients will die
Reality: 87 actually died, 13 survived
Precision = 87/100 = 86.7%

Much better: When it says "high risk", it's usually right!
```

**Real-World Impact:**
- **High Precision:** Few false alarms
  - Don't scare healthy patients unnecessarily
  - Avoid expensive/painful treatments for low-risk patients
  - Trustworthy predictions
- **Low Precision:** Many false alarms
  - Psychological harm (anxiety, stress)
  - Waste resources on false positives
  - Loss of trust in the model

---

### Metric 3: Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
       = True Positives / All Actual Positives
```

**What it means:**
- **Of all patients who actually died, how many did we catch?**
- How good are we at finding positive cases?
- Measures **miss rate**

**Results:**
```
Logistic Regression: 0.715 (71.5%) ← Best!
Random Forest:       0.415 (41.5%)
XGBoost:             0.528 (52.8%)
```

**Interpretation:**

**Logistic Regression:**
```
Reality: 100 patients died
Model caught: 72 of them
Missed: 28 (told them "low risk" but died)
Recall = 72/100 = 71.5%

Catches most deaths!
```

**Random Forest:**
```
Reality: 100 patients died
Model caught: Only 42 of them
Missed: 58 patients died unexpectedly!
Recall = 42/100 = 41.5%

Problem: Misses more than half of deaths!
```

**Real-World Impact:**
- **High Recall:** Catch most actual cases
  - Few missed deaths (low false negatives)
  - Early intervention for high-risk patients
  - Lives saved!
- **Low Recall:** Miss many cases
  - Patients sent home who should be treated
  - Deaths that could have been prevented
  - **Very dangerous in medical diagnosis!**

---

### Precision vs Recall Tradeoff

**The Fundamental Dilemma:**

You can't maximize both simultaneously - there's always a tradeoff!

**Scenario 1: Conservative Model (High Precision)**
```
Only predicts "Dead" when very confident
Result:
- Few false alarms (high precision)
- But misses many deaths (low recall)

Example: XGBoost
Precision: 86.7% (trustworthy predictions)
Recall: 52.8% (misses 47% of deaths!)
```

**Scenario 2: Aggressive Model (High Recall)**
```
Predicts "Dead" for anyone remotely suspicious
Result:
- Catches most deaths (high recall)
- But many false alarms (low precision)

Example: Logistic Regression
Recall: 71.5% (catches most deaths)
Precision: 42.1% (many false alarms)
```

**Threshold Adjustment:**

Models output probabilities:
```
Patient A: 0.3 probability of death
Patient B: 0.7 probability of death
```

**Default threshold: 0.5**
```
If P(Dead) > 0.5 → Predict Dead
If P(Dead) ≤ 0.5 → Predict Alive
```

**Lower threshold (e.g., 0.3) → Higher Recall**
```
More patients classified as "Dead"
Catch more actual deaths ✓
But more false alarms ✗
```

**Higher threshold (e.g., 0.7) → Higher Precision**
```
Fewer patients classified as "Dead"
Fewer false alarms ✓
But miss more deaths ✗
```

**Medical Decision:**
- Screening tests: Use low threshold (high recall - don't miss anyone)
- Treatment decisions: Use high threshold (high precision - don't treat unnecessarily)

---

### Metric 4: F1-Score

**Formula:**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
         = Harmonic Mean of Precision and Recall
```

**What it means:**
- **Balanced metric** combining precision and recall
- Single number to compare models
- Penalizes extreme imbalance

**Results:**
```
Logistic Regression: 0.530
Random Forest:       0.548
XGBoost:             0.657 ← Best balance!
```

**Why Harmonic Mean?**

**Arithmetic mean (regular average):**
```
Precision = 90%, Recall = 10%
Average = (90 + 10) / 2 = 50% ← Misleading!
```

**Harmonic mean (F1):**
```
Precision = 90%, Recall = 10%
F1 = 2 × (90 × 10) / (90 + 10) = 18% ← Penalizes imbalance!
```

**F1 forces balance:**
```
Model A: Precision=90%, Recall=20% → F1=0.32
Model B: Precision=50%, Recall=90% → F1=0.64
Model C: Precision=70%, Recall=70% → F1=0.70 ← Best!
```

**Real Examples:**

**Logistic Regression:**
```
Precision: 42.1%
Recall: 71.5%
F1 = 2 × (42.1 × 71.5) / (42.1 + 71.5) = 53.0%

Good recall but poor precision drags down F1
```

**XGBoost:**
```
Precision: 86.7%
Recall: 52.8%
F1 = 2 × (86.7 × 52.8) / (86.7 + 52.8) = 65.7%

Better balance → Higher F1 despite lower recall
```

---

### Complete Results Analysis

```
LogisticRegression:
Accuracy : 0.806   (80.6% correct overall)
Precision: 0.421   (42% of death predictions are correct)
Recall   : 0.715   (Catches 71.5% of actual deaths)
F1-score : 0.530

Analysis:
+ Catches most deaths (good recall)
- Too many false alarms (low precision)
- Use for: Initial screening (don't miss anyone)

RandomForest:
Accuracy : 0.896   (89.6% correct overall)
Precision: 0.810   (81% of death predictions are correct)
Recall   : 0.415   (Only catches 41.5% of deaths!)
F1-score : 0.548

Analysis:
+ High accuracy, trustworthy predictions
- Misses more than half of deaths! (low recall)
- Dangerous for medical use (too many false negatives)

XGBoost:
Accuracy : 0.916   (91.6% correct overall) ← Best
Precision: 0.867   (87% of death predictions are correct) ← Best
Recall   : 0.528   (Catches 52.8% of deaths)
F1-score : 0.657   ← Best balance

Analysis:
+ Highest accuracy and precision
+ Best F1-score (balanced performance)
- Moderate recall (misses 47% of deaths)
- Use for: Treatment decisions (trustworthy predictions)
```

---

### Which Model is Best?

**It depends on the use case:**

**Scenario 1: Initial Screening**
- **Goal:** Don't miss anyone who might die
- **Priority:** High Recall
- **Best Model:** Logistic Regression (71.5% recall)
- **Tradeoff:** Accept false alarms (will investigate further)

**Scenario 2: Treatment Decisions**
- **Goal:** Trustworthy predictions for aggressive treatment
- **Priority:** High Precision
- **Best Model:** XGBoost (86.7% precision)
- **Tradeoff:** Some deaths will be missed, but predictions are reliable

**Scenario 3: Overall Performance**
- **Goal:** Best balance of precision and recall
- **Priority:** High F1-Score
- **Best Model:** XGBoost (0.657 F1)
- **Tradeoff:** Balanced, no extreme strengths or weaknesses

**Scenario 4: Resource Constraints**
- **Goal:** Fast, simple, interpretable model
- **Priority:** Speed + Interpretability
- **Best Model:** Logistic Regression
- **Tradeoff:** Lower accuracy, but explainable and fast

---

## Comparison with Previous Assignments

| Aspect | Assignment 1 | Assignment 2 | Assignment 4 |
|--------|--------------|--------------|--------------|
| **Domain** | Agriculture | Entertainment | Healthcare |
| **Problem Type** | Regression | Recommendation | **Classification** |
| **Target** | Continuous (yield) | Rankings (movies) | **Binary (Dead/Alive)** |
| **Target Values** | 0-100,000+ | Preference scores | **0 or 1** |
| **Algorithms** | RF, XGBoost Regressor | Content, Collab, Hybrid | **LR, RF, XGB Classifier** |
| **Preprocessing** | OneHotEncoder | TF-IDF, SVD | **OneHotEncoder + StandardScaler** |
| **Evaluation** | R², RMSE | User satisfaction | **Accuracy, Precision, Recall, F1** |
| **Class Imbalance** | N/A | N/A | **Yes (balanced with class_weight)** |
| **Feature Scaling** | No (tree-based) | No (text-based) | **Yes (StandardScaler for numerical)** |
| **Medical Stakes** | Low (crop loss) | None (entertainment) | **High (life/death)** |

---

## Key Concepts Learned

### 1. Classification vs Regression
- **Regression:** Predict continuous values (Assignment 1)
- **Classification:** Predict discrete categories (Assignment 4)

### 2. Imbalanced Data
- **Problem:** More alive than dead patients
- **Solution:** `class_weight="balanced"` gives more weight to minority class

### 3. Feature Preprocessing
- **Numerical:** StandardScaler (normalize ranges)
- **Categorical:** OneHotEncoder (text → binary columns)
- **ColumnTransformer:** Apply different preprocessing to different column types

### 4. Pipeline
- Chain preprocessing + model training
- Ensures preprocessing applied consistently to train and test data
- Cleaner code, less error-prone

### 5. Ensemble Methods
- **Random Forest:** Parallel trees, majority vote
- **XGBoost:** Sequential trees, learn from mistakes
- Both outperform single models

### 6. Evaluation Metrics for Classification
- **Accuracy:** Overall correctness (can be misleading)
- **Precision:** Trustworthiness of positive predictions
- **Recall:** Completeness of catching positive cases
- **F1-Score:** Balanced metric

### 7. Precision-Recall Tradeoff
- Can't maximize both simultaneously
- Choose based on use case (screening vs treatment)

### 8. Medical ML Considerations
- **False negatives are dangerous** (missed deaths)
- **False positives have costs** (unnecessary treatment, anxiety)
- **Interpretability matters** (doctors need to trust the model)
- **Regulatory requirements** (FDA approval for medical devices)

---

## Summary in Simple Terms

**Goal:** Predict if breast cancer patients will survive or die

**Process:**
1. **Load data** - 4,024 patients with age, tumor info, hormone status
2. **Clean data** - Remove junk, convert "Alive"/"Dead" to 0/1
3. **Preprocess:**
   - Scale numerical features (age, tumor size)
   - Encode categorical features (race, stage, grade)
4. **Split data** - 80% train, 20% test (stratified to maintain balance)
5. **Train 3 models:**
   - **Logistic Regression** - Simple linear model
   - **Random Forest** - 300 trees voting
   - **XGBoost** - 300 trees learning from mistakes
6. **Evaluate with 4 metrics:**
   - **Accuracy** - Overall correctness (91.6% for XGBoost)
   - **Precision** - Trustworthiness (86.7% for XGBoost)
   - **Recall** - Completeness (71.5% for Logistic Regression)
   - **F1** - Balance (0.657 for XGBoost)

**Winner:** XGBoost (best accuracy, precision, and F1-score)

**Real-world use:**
- Doctors can identify high-risk patients
- Early intervention and personalized treatment
- Better resource allocation
- Potentially save lives!

---

## Advanced Topics (Optional)

### ROC Curve and AUC
- **ROC:** Receiver Operating Characteristic
- Plots True Positive Rate vs False Positive Rate at different thresholds
- **AUC:** Area Under Curve (0.5 = random, 1.0 = perfect)
- Better than single-threshold metrics

### Calibration
- Do predicted probabilities match reality?
- If model says 70% chance of death, do 70% actually die?
- Important for medical decisions

### Feature Importance
- Which features matter most?
- XGBoost and Random Forest provide importance scores
- Example: Tumor size, grade, lymph node status likely most important

### Cross-Validation
- Instead of single train/test split, use 5-fold or 10-fold
- More robust performance estimates
- Reduces variance from random split

### Hyperparameter Tuning
- Use GridSearchCV or RandomSearchCV
- Find optimal parameters automatically
- Example: Try learning_rate=[0.01, 0.05, 0.1], max_depth=[3, 4, 5, 6]

### SHAP Values
- Explainable AI technique
- Shows why model made specific predictions
- Critical for medical applications (doctors need explanations)

### Handling More Classes
- This is binary (Dead/Alive)
- Could extend to multi-class (Remission, Stable, Deteriorating, Dead)
- Metrics change: macro/micro averaging, confusion matrix becomes NxN
