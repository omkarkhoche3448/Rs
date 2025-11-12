# Assignment 5 — Amazon Product Recommendation System

This assignment builds two complementary recommenders for Amazon products using the same cleaned dataset but different modeling strategies: a matrix-factorization baseline implemented with `Surprise`, and a deep-learning model implemented with TensorFlow.

## Problem Statement
- Generate personalized product suggestions for Amazon shoppers.
- Compare a latent-factor baseline against a neural collaborative filtering model.
- Quantify accuracy using Root Mean Squared Error (RMSE) on a held-out test split.

## Dataset Overview
- Source: `amazon_products_sales_data_cleaned.csv` (~42k products, 17 columns).
- Key fields:
  - **Identity**: `product_title`, `product_category`, `product_page_url`.
  - **Engagement**: `product_rating`, `total_reviews`, `purchased_last_month`.
  - **Pricing**: `discounted_price`, `original_price`, `discount_percentage`.
  - **Badges**: `is_best_seller`, `is_sponsored`, `has_coupon`, `buy_box_availability`, `sustainability_tags`.
  - **Metadata**: `delivery_date`, `product_image_url`, `data_collected_at`.
- Challenge: No user identifiers were provided, so synthetic interactions were generated.

## Data Preparation Workflow
1. **Exploration & Missing Values**
   - Identified notable gaps (e.g., `buy_box_availability` missing for 35% of products, `sustainability_tags` missing for 93%).
   - Justified gaps: new listings, out-of-stock products, rare sustainability badges.

2. **Imputation Strategy**
   - `product_rating`: filled with column mean to keep rating scale intact.
   - `total_reviews`: set to zero for unrated products.
   - `discounted_price`: filled with median price to limit sensitivity to outliers.

3. **Synthetic Interaction Matrix**
   - `product_id`: re-indexed rows converted to strings.
   - `user_id`: 50,000 pseudo-users sampled uniformly to simulate interactions while guaranteeing reproducibility (`np.random.seed(42)`).
   - `rating`: `product_rating` plus Gaussian noise (μ=0, σ=0.15), clipped to `[1, 5]` and rounded to integers; imitates individual rating variance.

4. **Side-Feature Engineering**
   - `log_reviews = log1p(total_reviews)` to dampen skew from viral products.
   - `price_bucket = qcut(discounted_price, 5)` to capture relative price bands.
   - `is_bestSeller` / `hasCoupon`: boolean flags converted to integers for downstream models.

## Model 1: Hybrid SVD (Surprise)
- **Objective**: Learn latent user/item factors with explicit regularization and bias terms.
- **Pipeline**
  1. Prepare `df_core = products[['user_id', 'product_id', 'rating']]`.
  2. Load through `Reader(rating_scale=(1, 5))` and split 80/20 for train/test.
  3. Train `SVD(n_factors=50, reg_all=0.02, n_epochs=20)`.
  4. Evaluate with `accuracy.rmse(preds)` on the held-out set.
- **Result**: `Hybrid-SVD RMSE ≈ 0.5638`.
- **Interpretation**: Predictions deviate by ~0.56 stars on average — a strong collaborative filtering baseline that benefits from lightweight side-feature engineering and bias compensation.

## Model 2: Neural Collaborative Filtering (TensorFlow)
- **Objective**: Capture richer non-linear interactions by combining embeddings with dense layers.
- **Pre-processing**
  - Label-encode `price_bucket`, `is_bestSeller`, `hasCoupon` for embedding compatibility.
  - Construct feature matrix `X` (`user_id`, `product_id`, `log_reviews`, categorical flags) and target vector `y`.
  - Split into train/test (80/20) and build `tf.data.Dataset` pipelines with shuffling, batching (`1024`), mapping, and prefetching for efficient training.
- **Architecture Highlights**
  - Learned embeddings: 50-d user and item embeddings plus bias terms to mirror matrix factorization capacity.
  - Auxiliary embeddings: 4-d vectors for each categorical side feature.
  - Dense tower: `Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dense(1)` with additive user/item biases for final prediction.
- **Training Setup**
  - Optimizer: `Adam`.
  - Loss: Mean Squared Error, metric: RMSE.
  - Regularization: `EarlyStopping(patience=2, restore_best_weights=True)`, max 15 epochs (early stop at epoch 4).
- **Result**: `Neural-CF RMSE ≈ 0.5915`.
- **Interpretation**: Slightly worse than SVD; deep model is more expressive but more prone to overfitting without additional tuning or implicit feedback signals.

## Performance Comparison

| Model                        | RMSE (↓ better) | Notes                                                                    |
|-----------------------------|-----------------|--------------------------------------------------------------------------|
| Hybrid SVD (Surprise)       | **0.5638**       | Fast training, strong baseline; benefits from bias terms and latent factors. |
| Neural Collaborative Filter | 0.5915          | More flexible but higher variance; requires deeper tuning and regularization. |

## Key Takeaways
- Matrix factorization delivers the best RMSE with minimal complexity and faster iterations.
- Neural Collaborative Filtering opens the door to richer side-information modeling, but needs careful hyper-parameter tuning or additional regularization (e.g., dropout scheduling, embedding constraints).
- Synthetic user generation is a workable workaround in absence of real interactions; however, production systems should use real user-item histories.

## Real-World Application Ideas
- Populate “Customers also bought” carousels with SVD predictions as the default baseline.
- Reserve neural models for scenarios where abundant side information and compute are available (e.g., personalized homepages with re-ranking).
- Combine approaches: use SVD for candidate generation, neural model for downstream re-ranking once more features (e.g., textual embeddings, session signals) are integrated.