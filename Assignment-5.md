Overview: Amazon Product Recommendation System
This code builds a recommendation system for Amazon products using two advanced approaches:
Hybrid SVD - Matrix factorization with bias terms (Surprise library)
Neural Collaborative Filtering - Deep learning approach (TensorFlow)
Dataset: 42,000 Amazon products with ratings, reviews, prices, and metadata
Step-by-Step Breakdown:
1. Initial Data Exploration (rs-ass5.ipynb:1-4)
import pandas as pd
import numpy as np
products = pd.read_csv('/kaggle/input/amazon-products-sales-dataset-42k-items-2025/amazon_products_sales_data_cleaned.csv')
What's in the Dataset?
17 columns with product information: Product Identity:
product_title - Product name (e.g., "Apple AirPods Pro 2")
product_category - Phones, Laptops, etc.
Ratings & Reviews: 3. product_rating - Average rating (1.0 to 5.0 stars) 4. total_reviews - Number of reviews (e.g., 35,882) 5. purchased_last_month - How many bought recently (e.g., 10,000) Pricing: 6. discounted_price - Current price ($89.68) 7. original_price - Original price ($159.00) 8. discount_percentage - Discount % (43.60%) Product Flags: 9. is_best_seller - "Best Seller" or "No Badge" 10. is_sponsored - "Sponsored" or "Organic" 11. has_coupon - "Save 15% with coupon" or "No Coupon" 12. buy_box_availability - "Add to cart" or NaN 13. sustainability_tags - "Carbon impact" or NaN Other: 14. delivery_date - Estimated delivery (2025-09-01) 15. product_image_url - Image link 16. product_page_url - Product page link 17. data_collected_at - When data was scraped (2025-08-21) Total products: ~42,000
2. Missing Values Analysis (rs-ass5.ipynb:2)
products.isna().sum()
Missing data counts:
product_rating           1,024  (2.4% missing)
total_reviews            1,024  (2.4% missing)
purchased_last_month    10,511  (25% missing)
discounted_price         2,062  (5% missing)
original_price           2,062  (5% missing)
buy_box_availability    14,653  (35% missing)
delivery_date           11,983  (28% missing)
sustainability_tags     39,267  (93% missing - rarely used)
Why missing?
New products (no reviews yet)
Out of stock (no delivery date)
Not eligible for sustainability badges
Data collection issues
Part 1: Hybrid SVD Recommendation
3. Data Preparation (rs-ass5.ipynb:5, lines 1-9)
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

products = pd.read_csv('...')
Step 1: Handle Missing Values
products['product_rating'] = products['product_rating'].fillna(products['product_rating'].mean())
products['total_reviews']  = products['total_reviews'].fillna(0)
products['discounted_price'] = products['discounted_price'].fillna(products['discounted_price'].median())
Why these strategies? product_rating → Mean imputation:
Missing ratings filled with average (e.g., 4.2 stars)
Neutral assumption (not too high, not too low)
total_reviews → Fill with 0:
No reviews = 0 reviews (logical)
New products start with 0
discounted_price → Median imputation:
Median is robust to outliers (extreme prices)
Example: If prices are [$10, $15, $20, $500], median = $17.50 (better than mean = $136)
Step 2: Create User IDs (rs-ass5.ipynb:5, lines 11-15)
products = products.reset_index(drop=True)
products['product_id'] = products.index.astype(str)
N_USERS = 50_000
np.random.seed(42)
products['user_id'] = np.random.randint(0, N_USERS, size=len(products))
What's happening? Problem: Dataset has products but no user data! Solution: Simulate users product_id:
Use row index as product ID
Product 0, Product 1, Product 2, ...
user_id:
Randomly assign 50,000 fake users to products
Each product gets a random user ID (0 to 49,999)
This simulates "User X viewed/bought Product Y"
Example:
Row 0: Product=AirPods → user_id=12345 (User 12345 interacted with AirPods)
Row 1: Product=iPhone → user_id=789 (User 789 interacted with iPhone)
Row 2: Product=Laptop → user_id=12345 (User 12345 also interacted with Laptop)
Why random?
Real user data not available
Simulates diverse user interactions
seed(42) makes it reproducible
Step 3: Create Ratings (rs-ass5.ipynb:5, lines 17-19)
products['rating'] = (products['product_rating']
                        .add(np.random.normal(0, 0.15, size=len(products)))
                        .clip(1, 5).round().astype(int))
What's happening? Goal: Convert product ratings to individual user ratings Step-by-step:
Start with product_rating (e.g., 4.6 stars)
Add random noise: normal(0, 0.15)
Mean = 0 (centered)
Std = 0.15 (small variation)
Example: 4.6 + 0.08 = 4.68 or 4.6 - 0.12 = 4.48
Clip to valid range: clip(1, 5)
Ensures ratings stay between 1 and 5 stars
Round to integer: round()
4.68 → 5, 4.48 → 4
Convert to int: Final ratings are 1, 2, 3, 4, or 5
Why add noise?
Product rating = 4.6 (average)
Individual users might rate it 4, 5, or sometimes 3
Noise simulates individual user preferences
Makes data more realistic
Example:
Product: AirPods Pro, product_rating=4.6

Without noise:
All users rate it exactly 4.6 (unrealistic!)

With noise:
User 1: 4.6 + 0.12 = 4.72 → 5 stars
User 2: 4.6 - 0.08 = 4.52 → 5 stars  
User 3: 4.6 - 0.25 = 4.35 → 4 stars
User 4: 4.6 + 0.03 = 4.63 → 5 stars

More realistic distribution!
Step 4: Feature Engineering (rs-ass5.ipynb:5, lines 21-25)
products['log_reviews'] = np.log1p(products['total_reviews'])
products['price_bucket'] = pd.qcut(products['discounted_price'], 5, labels=False, duplicates='drop')
products['is_bestSeller'] = (products['is_best_seller'] == 'Best Seller').astype(int)
products['hasCoupon']     = (products['has_coupon'] == 'Yes').astype(int)
Creating side features for better predictions: 1. log_reviews - Log transformation:
log_reviews = np.log1p(total_reviews)
Why log transform? Reviews are heavily skewed:
Most products: 10-100 reviews
Popular products: 1,000-10,000 reviews
Viral products: 35,000+ reviews
Without log:
Product A: 50 reviews
Product B: 100 reviews  
Difference: 50 (seems small)

Product C: 1,000 reviews
Product D: 1,050 reviews
Difference: 50 (seems insignificant for popular products)
With log:
Product A: log(50) = 3.91
Product B: log(100) = 4.61
Difference: 0.70

Product C: log(1000) = 6.91
Product D: log(1050) = 6.96
Difference: 0.05

Compresses large values, spreads small values!
np.log1p() = log(1 + x) handles zeros safely (log(0) is undefined) 2. price_bucket - Quantile binning:
price_bucket = pd.qcut(discounted_price, 5, labels=False, duplicates='drop')
What is qcut?
Divides prices into 5 equal-sized groups (quantiles)
Each bucket has ~20% of products
Example:
All prices: [$5, $10, $15, $20, $25, $30, $50, $100, $200, $500]

5 buckets:
Bucket 0 (cheapest 20%): $5, $10  
Bucket 1: $15, $20
Bucket 2: $25, $30
Bucket 3: $50, $100
Bucket 4 (most expensive 20%): $200, $500
Why bucketize?
Simplifies continuous prices into categories
Captures price range effects (budget vs premium)
Reduces noise (small price differences don't matter)
3. is_bestSeller - Binary flag:
is_bestSeller = (products['is_best_seller'] == 'Best Seller').astype(int)
Converts:
"Best Seller" → 1
"No Badge" → 0
Why important?
Best sellers tend to have higher ratings
Social proof (people buy what others buy)
Quality signal
4. hasCoupon - Binary flag:
hasCoupon = (products['has_coupon'] == 'Yes').astype(int)
Converts:
"Save 15% with coupon" → 1
"No Coupon" → 0
Why important?
Coupons influence purchase decisions
Price-sensitive users prefer coupons
Can bias ratings (bargain hunters rate higher)
4. Surprise Dataset Preparation (rs-ass5.ipynb:5, lines 27-33)
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

df_core = products[['user_id', 'product_id', 'rating']].copy()
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_core, reader)
train, test = train_test_split(data, test_size=.2, random_state=42)
What is Surprise?
Scikit-learn for recommendation systems
Specialized library for collaborative filtering
Built-in algorithms: SVD, KNN, NMF, etc.
Preparing data: 1. Extract core columns:
df_core = products[['user_id', 'product_id', 'rating']].copy()
Only need user-item-rating triplets
Other features will be added separately
2. Define rating scale:
reader = Reader(rating_scale=(1, 5))
Tells Surprise ratings are 1-5 stars
Important for normalization
3. Load into Surprise format:
data = Dataset.load_from_df(df_core, reader)
Converts pandas DataFrame to Surprise Dataset object
Internal format optimized for collaborative filtering
4. Train/test split:
train, test = train_test_split(data, test_size=.2, random_state=42)
80% training (~33,600 interactions)
20% testing (~8,400 interactions)
Tests on unseen user-item pairs
5. Hybrid SVD Model (rs-ass5.ipynb:5, lines 35-40)
svd = SVD(n_factors=50, reg_all=.02, n_epochs=20)
svd.fit(train)
preds = svd.test(test)
print('Hybrid-SVD RMSE:', accuracy.rmse(preds))
What is SVD?
Singular Value Decomposition (same as Assignment 2!)
Matrix factorization for collaborative filtering
Learns hidden patterns in user-item interactions
How it works: User-Item Matrix:
            Product1  Product2  Product3  ...
User1          5        ?         4
User2          ?        3         ?
User3          4        ?         5
...
SVD decomposes into:
User matrix (50,000 × 50) × Item matrix (50 × 42,000)
Each user/item gets 50 "latent factors":
Factor 1: "Tech enthusiast" score
Factor 2: "Budget conscious" score
Factor 3: "Premium quality preference"
... 50 hidden dimensions
Prediction:
Rating = User factors · Item factors + User bias + Item bias + Global bias
Parameters: n_factors=50:
Use 50 latent dimensions
More factors = capture more nuance but slower
50 is standard for medium-sized datasets
reg_all=0.02:
Regularization strength
Prevents overfitting
Penalizes overly large weights
0.02 is light regularization
n_epochs=20:
Training iterations
Each epoch goes through all training data once
More epochs = better fit but risk overfitting
Training & Evaluation:
svd.fit(train)  # Learn user/item factors from training data
preds = svd.test(test)  # Predict ratings for test set
accuracy.rmse(preds)  # Calculate error
Result:
Hybrid-SVD RMSE: 0.5638
What does RMSE = 0.5638 mean?
Predictions are off by ~0.56 stars on average
If actual rating = 5, might predict 4.4 or 5.6
Pretty good! (random guessing would be ~1.5 RMSE)
Part 2: Neural Collaborative Filtering
6. Data Preparation (Neural) (rs-ass5.ipynb:7, lines 1-33)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
New imports:
tensorflow - Deep learning framework
LabelEncoder - Convert categories to integers (0, 1, 2, ...)
Same data cleaning as before, but with extra steps:
# Encode categorical features
cat_cols = ['price_bucket', 'is_bestSeller', 'hasCoupon']
for c in cat_cols:
    products[c] = LabelEncoder().fit_transform(products[c].astype(str))
Why LabelEncoder?
Neural networks need integers, not strings
Converts: [0, 1, 2, 3, 4] (already numbers, but ensures consistency)
Prepare arrays:
X = products[['user_id', 'product_id', 'log_reviews'] + cat_cols].values
y = products['rating'].values.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X features (6 total):
user_id
product_id
log_reviews (continuous)
price_bucket (categorical: 0-4)
is_bestSeller (binary: 0 or 1)
hasCoupon (binary: 0 or 1)
y target:
rating (1-5 stars)
7. TensorFlow Dataset Pipeline (rs-ass5.ipynb:7, lines 35-54)
def split_features(x, y):
    return {
        'user_id': tf.cast(x[:, 0], tf.int32),
        'product_id': tf.cast(x[:, 1], tf.int32),
        'log_reviews': x[:, 2],
        'price_bucket': tf.cast(x[:, 3], tf.int32),
        'is_bestSeller': tf.cast(x[:, 4], tf.int32),
        'hasCoupon': tf.cast(x[:, 5], tf.int32)
    }, y

train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(10000)
            .batch(1024)
            .map(split_features)
            .prefetch(tf.data.AUTOTUNE))
What's happening? TensorFlow Dataset advantages:
Efficient data loading (batching, shuffling)
Parallel processing
Prefetching (loads next batch while training)
Pipeline steps: 1. Create dataset:
tf.data.Dataset.from_tensor_slices((X_train, y_train))
Wraps numpy arrays
2. Shuffle:
.shuffle(10000)
Randomizes order (prevents learning spurious patterns)
Buffer size = 10,000 (shuffles within this window)
3. Batch:
.batch(1024)
Groups 1,024 examples per batch
Trains on 1,024 samples at a time (mini-batch gradient descent)
Faster than one-by-one
4. Split features:
.map(split_features)
Separates combined array into named inputs
Column 0 → 'user_id', Column 1 → 'product_id', etc.
Prepares for model inputs
5. Prefetch:
.prefetch(tf.data.AUTOTUNE)
While GPU trains on batch N, CPU prepares batch N+1
Reduces idle time
8. Neural Network Architecture (rs-ass5.ipynb:7, lines 56-83)
This is the most complex part - a deep learning model for recommendations!
# Model parameters
n_users = int(products['user_id'].max() + 1)  # 50,000
n_items = int(products['product_id'].astype(int).max() + 1)  # 42,000
k = 50  # Embedding dimension
cat_dims = [products[c].nunique() for c in cat_cols]  # [5, 2, 2]
cat_embed = 4  # Categorical embedding size
Parameters:
50,000 unique users
42,000 unique products
50-dimensional embeddings (like SVD's 50 factors)
3 categorical features with 5, 2, 2 unique values
4-dimensional embeddings for categories
Model Inputs:
user_in = tf.keras.layers.Input(shape=(), dtype='int32', name='user_id')
item_in = tf.keras.layers.Input(shape=(), dtype='int32', name='product_id')
log_r_in = tf.keras.layers.Input(shape=(), dtype='float32', name='log_reviews')
cat_ins = [tf.keras.layers.Input(shape=(), dtype='int32', name=c) for c in cat_cols]
6 inputs:
user_id (integer)
product_id (integer)
log_reviews (float)
price_bucket (integer)
is_bestSeller (integer)
hasCoupon (integer)
Embedding Layers:
user_emb = tf.keras.layers.Embedding(n_users, k)(user_in)
item_emb = tf.keras.layers.Embedding(n_items, k)(item_in)
user_bias = tf.keras.layers.Embedding(n_users, 1)(user_in)
item_bias = tf.keras.layers.Embedding(n_items, 1)(item_in)
What are embeddings?
Convert IDs to dense vectors
Learn representations during training
User embedding:
User 12345 → [0.2, -0.5, 0.8, ..., 0.3]  (50 numbers)
            (represents user's preferences in 50 dimensions)
Item embedding:
Product 567 → [0.1, 0.7, -0.2, ..., 0.5]  (50 numbers)
             (represents product's characteristics)
Biases:
user_bias: Some users rate everything high/low
item_bias: Some products are universally loved/hated
Example:
User 999 bias: +0.5 (rates 0.5 stars higher than average)
Product 123 bias: -0.3 (rated 0.3 stars lower than average)
Categorical embeddings:
cat_embs = [tf.keras.layers.Embedding(d, cat_embed)(inp) for d, inp in zip(cat_dims, cat_ins)]
cat_flat = tf.keras.layers.Concatenate()(cat_embs)
Converts categories to 4D vectors:
price_bucket=2 → [0.3, -0.1, 0.5, 0.2]
is_bestSeller=1 → [0.8, 0.6, -0.2, 0.1]
hasCoupon=0 → [-0.3, 0.4, 0.1, -0.5]

Concatenated: [0.3, -0.1, 0.5, 0.2, 0.8, 0.6, -0.2, 0.1, -0.3, 0.4, 0.1, -0.5]
             (12 numbers total: 3 categories × 4 dims)
Dense Layers:
log_r_dense = tf.keras.layers.Reshape((1,))(log_r_in)
log_r_dense = tf.keras.layers.Dense(8, activation='relu')(log_r_dense)
Processes log_reviews:
Reshapes scalar to 1D vector
Dense layer with 8 neurons (learns non-linear transformations)
ReLU activation (max(0, x) - introduces non-linearity)
Concatenate All Features:
x = tf.keras.layers.Concatenate()([user_emb, item_emb, log_r_dense, cat_flat])
Combines everything:
user_emb: 50 numbers
item_emb: 50 numbers
log_r_dense: 8 numbers
cat_flat: 12 numbers
------------------------
Total: 120 numbers
Deep Neural Network:
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)
Architecture:
Input (120 features)
    ↓
Dense 128 neurons (ReLU)
    ↓
Dropout 20% (randomly drop neurons to prevent overfitting)
    ↓
Dense 64 neurons (ReLU)
    ↓
Output 1 neuron (predicted rating)
Why deep layers?
Learns complex non-linear patterns
Captures interactions between features
Example: "Tech enthusiast" × "Premium product" → Higher rating
Dropout:
Randomly deactivates 20% of neurons during training
Prevents overfitting (model doesn't memorize training data)
Forces redundancy (multiple neurons learn similar patterns)
Add Biases:
user_bias_flat = tf.keras.layers.Flatten()(user_bias)
item_bias_flat = tf.keras.layers.Flatten()(item_bias)
output = tf.keras.layers.Add()([output, user_bias_flat, item_bias_flat])
Final prediction:
Rating = Neural Network output + User bias + Item bias
Example:
NN output: 4.2
User bias: +0.3 (generous rater)
Item bias: -0.1 (slightly below average product)
Final: 4.2 + 0.3 - 0.1 = 4.4 stars
9. Model Compilation & Training (rs-ass5.ipynb:7, lines 85-93)
model = tf.keras.Model(inputs=[user_in, item_in, log_r_in] + cat_ins, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

early = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_ds, validation_data=test_ds, epochs=15, callbacks=[early], verbose=2)
Compilation: Optimizer: Adam
Adaptive learning rate
Efficient gradient descent variant
Industry standard
Loss: MSE (Mean Squared Error)
MSE = average of (predicted - actual)²
Penalizes large errors heavily
Suitable for regression (predicting ratings)
Metric: RMSE
RMSE = sqrt(MSE)
Same units as ratings (stars)
Easier to interpret
Training: Early Stopping:
EarlyStopping(patience=2, restore_best_weights=True)
Stops training if validation RMSE doesn't improve for 2 epochs
Prevents overfitting
Restores weights from best epoch
Epochs: 15
Maximum 15 passes through training data
Early stopping might stop earlier
Training progress:
Epoch 1: Train loss=5.90, Val RMSE=1.06 (poor start)
Epoch 2: Train loss=0.63, Val RMSE=0.59 (huge improvement!)
Epoch 3: Train loss=0.34, Val RMSE=0.60 (slight degradation)
Epoch 4: Train loss=0.21, Val RMSE=0.61 (worse - early stop triggered!)
Early stopping activates after Epoch 4 (2 epochs without improvement).
10. Final Evaluation (rs-ass5.ipynb:7, lines 95-97)
test_rmse = model.evaluate(test_ds, return_dict=True)['root_mean_squared_error']
print(f"Neural-LF (TensorFlow) RMSE: {test_rmse:.4f}")
Result:
Neural-LF (TensorFlow) RMSE: 0.5915
Comparison:
Hybrid SVD:  RMSE = 0.5638 ← Better!
Neural CF:   RMSE = 0.5915
Observations:
SVD slightly outperforms neural network
Neural network has more parameters (risk of overfitting)
SVD is simpler and faster
Neural network has potential with more tuning
Summary
Goal: Recommend Amazon products to users Two Approaches: 1. Hybrid SVD (Surprise):
Matrix factorization (50 latent factors)
Learns user/item embeddings + biases
Fast, simple, effective
RMSE: 0.5638 ← Winner!
2. Neural Collaborative Filtering (TensorFlow):
Deep learning with embeddings
128 → 64 dense layers
Dropout regularization
More complex, captures non-linear patterns
RMSE: 0.5915
Key Techniques:
Simulated user interactions (50,000 fake users)
Feature engineering (log_reviews, price_bucket, flags)
Embeddings (convert IDs to dense vectors)
Bias terms (user/item-specific adjustments)
Early stopping (prevent overfitting)
Real-world use:
Amazon product recommendations
"Customers who bought this also bought..."
Personalized homepages
Email recommendations