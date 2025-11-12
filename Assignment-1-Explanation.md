# Assignment 1: Crop Yield Prediction - Complete Explanation

## Overview
This code predicts **crop yield** (how much crops will grow) using machine learning. It uses environmental and agricultural data to predict harvest amounts.

---

## Step-by-Step Breakdown

### 1. Loading Data
**File:** [rs-ass1.ipynb:1](rs-ass1.ipynb#cell-0)

```python
import pandas as pd
df = pd.read_csv("/kaggle/input/crop-yield-prediction-dataset/yield_df.csv")
df.head()
```

**What it does:**
- Loads a dataset about crop yields from different countries
- Shows data like:
  - **Area** - Country name (e.g., Albania)
  - **Item** - Crop type (e.g., Maize, Potatoes, Rice)
  - **Year** - Year of harvest
  - **hg/ha_yield** - Crop yield in hectograms per hectare (TARGET to predict)
  - **average_rain_fall_mm_per_year** - Annual rainfall
  - **pesticides_tonnes** - Amount of pesticides used
  - **avg_temp** - Average temperature

**Sample Output:**
```
   Area         Item  Year  hg/ha_yield  average_rain_fall_mm_per_year  pesticides_tonnes  avg_temp
0  Albania      Maize  1990       36613                         1485.0              121.0     16.37
1  Albania   Potatoes  1990       66667                         1485.0              121.0     16.37
```

---

### 2. Data Cleaning
**File:** [rs-ass1.ipynb:3](rs-ass1.ipynb#cell-2)

```python
df.drop(columns=["Unnamed: 0"], inplace=True)
```

**What it does:**
- Removes unnecessary column "Unnamed: 0" (just an index column)
- Checks data types to ensure numbers are numbers and text is text

---

### 3. Building Prediction Models
**File:** [rs-ass1.ipynb:5](rs-ass1.ipynb#cell-4)

#### 3.1 Preparing the Data

```python
X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

cat_features = ["Area", "Item"]
num_features = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
```

**What it does:**
- **X (Features)** = Things that affect yield → rainfall, pesticides, temperature, year, country, crop type
- **y (Target)** = The yield we want to predict
- Separates categorical (text) from numerical (numbers) features

#### 3.2 Preprocessing

```python
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
], remainder='passthrough')
```

**What it does:**
- **One-Hot Encoding** - Converts text data into numbers
  - Example: "Albania" → [1, 0, 0, 0...], "Argentina" → [0, 1, 0, 0...]
- Machine learning models can only work with numbers, not text
- `remainder='passthrough'` keeps numerical features as-is

#### 3.3 Splitting Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**What it does:**
- **80% training data** - Used to teach the model
- **20% test data** - Used to evaluate accuracy (unseen data)
- `random_state=42` ensures reproducible results

---

### 4. Model 1: Random Forest

```python
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
```

#### What is a Pipeline?
Think of it as an **assembly line**:
1. **Step 1 (Preprocessor):** Clean & convert data (text → numbers)
2. **Step 2 (Regressor):** Make predictions

#### What is Random Forest?
**Concept:** Like asking 200 experts and taking the average of their opinions

**How it works:**
- `n_estimators=200` → Creates **200 decision trees**
- Each tree looks at random subsets of data and features
- Each tree makes its own prediction
- **Final prediction = Average of all 200 predictions**
- This reduces errors and overfitting

**Real-world analogy:**
- One expert might be biased or wrong
- 200 experts together give more reliable answers (wisdom of the crowd)

**Training & Prediction:**
- `rf_model.fit(X_train, y_train)` → Learns patterns from training data
- `rf_preds = rf_model.predict(X_test)` → Makes predictions on new test data

---

### 5. Model 2: XGBoost

```python
xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42))
])

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
```

#### What is XGBoost?
**Concept:** Like learning from mistakes, one step at a time

**How it works:**
- `n_estimators=300` → Creates **300 decision trees sequentially**
- **Tree 1:** Makes initial predictions
- **Tree 2:** Focuses on fixing Tree 1's mistakes
- **Tree 3:** Fixes what Trees 1+2 got wrong
- **Tree 300:** Final corrections
- **Final prediction = Weighted combination of all trees**

**Key Parameters:**
- `learning_rate=0.1` → How fast it learns (10% per step)
  - Lower = slower but more careful learning
  - Higher = faster but might miss details
  - 0.1 is a good balance

#### Random Forest vs XGBoost

| Feature | Random Forest | XGBoost |
|---------|---------------|---------|
| **Tree Building** | Independent (parallel) | Sequential (one after another) |
| **Learning Strategy** | Each tree votes equally | Trees learn from previous mistakes |
| **Analogy** | Democracy (everyone votes) | Iterative improvement (learn from errors) |
| **Speed** | Faster training | Slower but more powerful |
| **Overfitting** | Less prone | Needs careful tuning |

---

### 6. Evaluation - Measuring Performance

```python
def evaluate(y_true, y_pred, model_name):
    print(f"\n{model_name} Results:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")

evaluate(y_test, rf_preds, "Random Forest")
evaluate(y_test, xgb_preds, "XGBoost")
```

#### What is Evaluation?
After training models, we ask: **"How good are the predictions?"**

**Parameters:**
- `y_true` = **Actual real values** from test data
- `y_pred` = **Predicted values** from the model
- `model_name` = Name for display

---

#### Metric 1: R² Score (R-Squared)

**What it means:**
- Tells you **how well your model explains the data**
- Range: **0 to 1** (higher is better)
  - **1.0 = Perfect** (100% accurate predictions)
  - **0.8 = Good** (explains 80% of variance)
  - **0.5 = Mediocre** (only explains 50%)
  - **0.0 = Bad** (model is useless)

**In your results:**
- Random Forest: **R² = 0.9877** → 98.77% accurate! Excellent!
- XGBoost: **R² = 0.9741** → 97.41% accurate! Very good!

**Real-world analogy:**
If you're predicting exam scores:
- R² = 0.98 means you can predict 98 out of 100 points correctly

---

#### Metric 2: RMSE (Root Mean Squared Error)

**What it means:**
- Shows **average error** in predictions (same units as target)
- **Lower is better** (0 = perfect, no errors)

**How it's calculated:**
1. For each prediction: `error = (actual - predicted)²`
2. Average all squared errors
3. Take the square root

**In your results:**
- Random Forest: **RMSE = 9,454**
- XGBoost: **RMSE = 13,715**

**What does 9,454 mean?**
- Yield is measured in **hectograms per hectare (hg/ha)**
- If actual yield = 36,613, model might predict 46,067 or 27,159 (±9,454 error)
- About **25% average error**, which is reasonable for agricultural predictions

---

#### Why Use Both Metrics?

| Metric | What it tells you | Example |
|--------|-------------------|---------|
| **R² Score** | Overall fit - "How much variance explained?" | 0.9877 = explains 98.77% of patterns |
| **RMSE** | Practical error - "How far off in real numbers?" | 9,454 = average mistake size |

**Example Comparison:**
- Model A: R² = 0.99, RMSE = 50,000
- Model B: R² = 0.98, RMSE = 5,000

**Winner:** Model B! Lower RMSE means more precise real-world predictions.

---

### 7. Results & Conclusion

**Output:**
```
Random Forest Results:
R² Score: 0.9877
RMSE: 9453.84

XGBoost Results:
R² Score: 0.9741
RMSE: 13714.82
```

**Winner: Random Forest**
- Higher R² (0.9877 > 0.9741)
- Lower RMSE (9,454 < 13,715)

**Why did Random Forest win?**
- XGBoost needs more hyperparameter tuning
- This dataset works well with Random Forest's ensemble approach
- Random Forest is less prone to overfitting on this data

---

## Summary in Simple Terms

**Goal:** Predict how much crops will grow based on weather and farming conditions

**Process:**
1. Load crop data (country, crop type, rainfall, temperature, pesticides)
2. Clean and prepare data (remove junk, convert text to numbers)
3. Split data (80% training, 20% testing)
4. Train two models:
   - **Random Forest** - 200 trees voting together
   - **XGBoost** - 300 trees learning from mistakes
5. Evaluate both models (R² and RMSE)
6. **Winner: Random Forest** with 98.77% accuracy!

**Real-world use:**
Farmers and governments can use this to predict harvests and plan food supply!

---

## Key Concepts Learned

1. **Supervised Learning** - Learning from labeled data (we know the yields)
2. **Regression** - Predicting continuous numbers (yield amounts)
3. **Ensemble Methods** - Combining multiple models (trees) for better accuracy
4. **Train/Test Split** - Testing on unseen data to measure real performance
5. **Preprocessing** - Converting text to numbers for machine learning
6. **Model Evaluation** - Using R² and RMSE to measure accuracy
