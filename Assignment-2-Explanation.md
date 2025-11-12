# Assignment 2: Movie Recommendation System - Complete Explanation

## Overview
This code builds a **Movie Recommendation System** that suggests movies users will enjoy. It uses **3 different approaches**:
1. **Content-Based Filtering** - Recommends similar movies based on genres/tags
2. **Collaborative Filtering** - Recommends based on what similar users liked
3. **Hybrid** - Combines both methods (50-50 mix)

---

## Step-by-Step Breakdown

### 1. Installation
**File:** [rs-ass2-and-ass3.ipynb:1](rs-ass2-and-ass3.ipynb#cell-0)

```python
!pip install faiss-cpu
!pip install surprise
```

**Installing special libraries:**
- **faiss-cpu** - Facebook's library for fast similarity search
  - Efficiently finds similar movies among millions
  - Used for content-based recommendations
- **surprise** - Scikit-learn for recommendation systems
  - Specialized tools for collaborative filtering
  - Built-in algorithms like SVD, KNN, etc.

---

### 2. Loading & Preparing Data
**File:** [rs-ass2-and-ass3.ipynb:2](rs-ass2-and-ass3.ipynb#cell-1) (Lines 13-26)

```python
ratings = pd.read_csv("/kaggle/working/Filtered_Ratings.csv")
movies  = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")
tags    = pd.read_csv("/kaggle/input/movielens-20m-dataset/tag.csv")
```

#### What's Loaded:

**1. Ratings Dataset**
```
userId | movieId | rating
  1    |   231   |  4.5
  1    |   532   |  5.0
```
- Users rating movies from 0.5 to 5.0 stars

**2. Movies Dataset**
```
movieId | title                    | genres
  231   | Dumb & Dumber (1994)     | Comedy
  532   | The Matrix (1999)        | Action|Sci-Fi
```

**3. Tags Dataset**
```
movieId | tag
  532   | mind-bending
  532   | dystopian
```
- User-created descriptive tags

---

#### Data Cleaning & Merging

```python
# Collapse all tags for each movie into one text
tags_agg = (tags.groupby("movieId")["tag"]
              .apply(lambda x: " ".join(x.fillna("").astype(str)))
              .reset_index(name="tags"))

# Convert genres from "Action|Thriller" → "Action Thriller"
movies["genres"] = movies["genres"].str.replace("|", " ")

# Merge tags with movies
movies = movies.merge(tags_agg, on="movieId", how="left").fillna("")

# Create combined content column
movies["content"] = movies["genres"] + " " + movies["tags"]
```

**What it does:**
- Combines all tags for each movie: `["mind-bending", "dystopian"]` → `"mind-bending dystopian"`
- Converts pipe-separated genres to spaces: `"Action|Sci-Fi"` → `"Action Sci-Fi"`
- Creates **content** column = genres + tags
  - Example: `"Action Sci-Fi mind-bending dystopian"`
- This text is used to find similar movies

---

### 3. Content-Based Filtering
**File:** [rs-ass2-and-ass3.ipynb:2](rs-ass2-and-ass3.ipynb#cell-1) (Lines 28-41)

**Core Idea:** *"If you liked Movie A, you'll like movies similar to it"*

---

#### Step 1: TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(max_features=5_000, stop_words='english')
tfidf_sparse = vectorizer.fit_transform(movies["content"])
tfidf_mat = np.asarray(tfidf_sparse.todense(), dtype=np.float32)
```

**What is TF-IDF?**
- **TF-IDF** = Term Frequency - Inverse Document Frequency
- Converts text into numbers that represent importance

**How it works:**
- **Term Frequency (TF):** How often a word appears in a movie's description
  - "Sci-Fi" appears 3 times in The Matrix description → Higher TF
- **Inverse Document Frequency (IDF):** How rare a word is across all movies
  - "Sci-Fi" appears in 5% of movies → Rare → High IDF
  - "movie" appears in 90% of movies → Common → Low IDF
- **TF-IDF = TF × IDF**
  - Gives high scores to important, distinctive words
  - Ignores common, meaningless words

**Parameters:**
- `max_features=5_000` → Use top 5,000 most important words
- `stop_words='english'` → Ignore common words like "the", "and", "is"

**Result:**
Each movie becomes a vector of 5,000 numbers representing its content
```
The Matrix → [0.0, 0.0, 0.8, 0.0, 0.6, ..., 0.0]
             (position 3 = "sci-fi", position 5 = "action", etc.)
```

---

#### Step 2: FAISS Index for Fast Search

```python
index = faiss.IndexFlatIP(tfidf_mat.shape[1])
faiss.normalize_L2(tfidf_mat)
index.add(tfidf_mat)
```

**What is FAISS?**
- **FAISS** = Facebook AI Similarity Search
- Ultra-fast library for finding similar vectors
- Can search millions of movies in milliseconds

**How it works:**
- `IndexFlatIP` → Inner Product index (measures similarity)
  - Inner product of normalized vectors = cosine similarity
- `normalize_L2` → Normalizes vectors to unit length
  - Makes all vectors same "magnitude" for fair comparison
- `index.add()` → Stores all movie vectors for instant lookup

**Analogy:**
Like a library index that lets you instantly find books similar to one you're holding

---

#### Step 3: Content Recommendation Function

```python
def content_recommend(movie_id, k=5):
    idx = movies.index[movies["movieId"] == movie_id][0]
    query = tfidf_mat[idx].reshape(1, -1)
    faiss.normalize_L2(query)
    _, I = index.search(query, k+1)
    return movies.iloc[I[0][1:]][["movieId", "title"]]
```

**What it does:**
1. **Find the movie's position** in the dataset (`idx`)
2. **Get its TF-IDF vector** (`query`)
3. **Normalize it** for comparison
4. **Search FAISS index** for k+1 most similar movies
5. **Return top k** (skip the movie itself - it's always most similar to itself!)

**Example:**
```
User liked: "Village of the Damned (1960)" (Horror)

Content-based recommends:
- Children of the Damned (1963)    [Similar title + genre]
- He Knows You're Alone (1979)     [Horror]
- Push (2009)                       [Thriller/Horror]
```

---

### 4. Collaborative Filtering
**File:** [rs-ass2-and-ass3.ipynb:2](rs-ass2-and-ass3.ipynb#cell-1) (Lines 43-58)

**Core Idea:** *"Users similar to you also liked these movies"*

---

#### Building the Collaborative Model

```python
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset = data.build_full_trainset()
svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)
```

**What is SVD?**
- **SVD** = Singular Value Decomposition
- **Matrix factorization** technique
- Finds hidden patterns in user-movie interactions

**How it works:**

**Original Rating Matrix:**
```
        Movie1  Movie2  Movie3  Movie4
User1     5      ?       4       ?
User2     ?      3       ?       5
User3     4      ?       ?       3
```
(? = not rated)

**SVD decomposes this into:**
1. **User features** (50 hidden preferences)
   - User1: [0.8 action, 0.2 romance, 0.6 sci-fi, ...]
2. **Movie features** (50 hidden characteristics)
   - Movie1: [0.9 action, 0.1 romance, 0.7 sci-fi, ...]

**Prediction:**
```
Predicted rating = User features · Movie features
User1 rating Movie2 = [0.8, 0.2, 0.6, ...] · [Movie2 features]
```

**Parameters:**
- `n_factors=50` → Use 50 hidden features/dimensions
  - More factors = captures more nuance but slower
  - 50 is a good balance
- `rating_scale=(0.5, 5.0)` → Valid rating range

**Learns patterns like:**
- "Users who liked The Matrix also liked Inception"
- "User123 tends to rate action movies 4+ stars"
- "Movie456 is generally rated high by sci-fi fans"

---

#### Collaborative Recommendation Function

```python
def collab_recommend(user_id, k=5):
    seen = set(ratings[ratings["userId"]==user_id]["movieId"])
    preds = [svd.predict(user_id, iid) for iid in movies["movieId"] if iid not in seen]
    preds.sort(key=lambda x: x.est, reverse=True)
    top = preds[:k]
    return pd.DataFrame([(int(p.iid), movies[movies.movieId==p.iid].title.iloc[0]) for p in top],
                        columns=["movieId", "title"])
```

**What it does:**
1. **Get movies user already watched** (`seen`)
   - Don't recommend movies they've already seen
2. **Predict ratings for all unwatched movies**
   - `svd.predict(user_id, movie_id)` → Predicted rating
3. **Sort by predicted rating** (highest first)
4. **Return top k recommendations**

**Example:**
```
User 77549 hasn't watched these classics:

Collaborative filtering recommends:
- Schindler's List (1993)           [Predicted: 4.8 stars]
- Shawshank Redemption, The (1994)  [Predicted: 4.7 stars]
- Usual Suspects, The (1995)        [Predicted: 4.6 stars]
```

**Why these?**
- Other users with similar taste to User 77549 rated these highly
- SVD learned User 77549 likes highly-rated dramas

---

### 5. Hybrid Recommendation
**File:** [rs-ass2-and-ass3.ipynb:2](rs-ass2-and-ass3.ipynb#cell-1) (Lines 60-79)

**Core Idea:** *"Combine content similarity + collaborative predictions for best results"*

---

#### The Hybrid Function

```python
def hybrid_recommend(user_id, k=5):
    seen = set(ratings[ratings["userId"]==user_id]["movieId"])
    cands = [iid for iid in movies["movieId"] if iid not in seen]

    # 1. Get collaborative scores
    collab_scores = {iid: svd.predict(user_id, iid).est for iid in cands}

    # 2. Get content scores based on last 10 liked movies
    liked = list(ratings[(ratings.userId==user_id)&(ratings.rating>=4)].movieId.tail(10))
    content_scores = defaultdict(float)
    for lid in liked:
        sims = content_recommend(lid, k=50)
        for _, row in sims.iterrows():
            content_scores[row.movieId] += 1.0   # Count frequency

    # 3. Normalize both scores to 0-1 range
    def norm(d):
        m, M = min(d.values()), max(d.values())
        return {k: (v-m)/(M-m+1e-8) for k,v in d.items()}

    cs, cls = norm(content_scores), norm(collab_scores)

    # 4. 50-50 blend
    hybrid = {iid: 0.5*cs.get(iid,0) + 0.5*cls.get(iid,0) for iid in cands}

    # 5. Return top k
    top = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:k]
    return pd.DataFrame([(int(iid), movies[movies.movieId==iid].title.iloc[0]) for iid,_ in top],
                        columns=["movieId", "title"])
```

---

#### Step-by-Step Process

**Step 1: Collaborative Scores**
```python
collab_scores = {iid: svd.predict(user_id, iid).est for iid in cands}
```
- Predict rating for each unwatched movie using SVD
- Example: `{movieId: predicted_rating}`
  - Movie 318: 4.5
  - Movie 527: 4.3

**Step 2: Content Scores**
```python
liked = ratings[(ratings.userId==user_id)&(ratings.rating>=4)].movieId.tail(10)
content_scores = defaultdict(float)
for lid in liked:
    sims = content_recommend(lid, k=50)
    for _, row in sims.iterrows():
        content_scores[row.movieId] += 1.0
```
- Get user's **last 10 highly-rated movies** (rating >= 4)
- For each liked movie, find 50 similar movies
- **Count how often each movie appears** in similarity lists
- Example: If Movie 318 appears similar to 7 out of 10 liked movies:
  - content_scores[318] = 7.0

**Step 3: Normalization**
```python
def norm(d):
    m, M = min(d.values()), max(d.values())
    return {k: (v-m)/(M-m+1e-8) for k,v in d.items()}
```
- **Why normalize?**
  - Collaborative scores: 0.5 to 5.0 range
  - Content scores: 0 to 10 range (counts)
  - Can't compare apples to oranges!
- **Min-Max normalization** scales both to 0-1 range
- Formula: `(value - min) / (max - min)`

**Example:**
```
Before normalization:
collab_scores = {318: 4.5, 527: 3.2, 50: 4.8}
content_scores = {318: 7, 527: 2, 50: 9}

After normalization (0-1 range):
collab_norm = {318: 0.81, 527: 0.0, 50: 1.0}
content_norm = {318: 0.71, 527: 0.0, 50: 1.0}
```

**Step 4: 50-50 Blend**
```python
hybrid = {iid: 0.5*cs.get(iid,0) + 0.5*cls.get(iid,0) for iid in cands}
```
- Combine both scores with equal weight
- **Hybrid score = 50% content + 50% collaborative**

**Example:**
```
Movie 318:
  Content score (normalized): 0.71
  Collaborative score (normalized): 0.81
  Hybrid score = 0.5 × 0.71 + 0.5 × 0.81 = 0.76

Movie 527:
  Content: 0.0, Collaborative: 0.0
  Hybrid = 0.0
```

**Step 5: Return Top k**
- Sort by hybrid score (highest first)
- Return top 5 recommendations

---

#### Why Hybrid is Better

| Method | Strength | Weakness | Example |
|--------|----------|----------|---------|
| **Content-Based** | Finds similar movies | Ignores user preferences | Recommends all horror if you watched one horror movie |
| **Collaborative** | Knows what you like | Can't explain why | Recommends based on similar users but might not match your taste |
| **Hybrid** | Best of both worlds! | More complex | Recommends similar movies that popular users also loved |

**Real Example:**
- User likes "The Matrix"
- **Content-based:** Recommends sci-fi action movies (genre match)
- **Collaborative:** Recommends critically acclaimed films (what similar users liked)
- **Hybrid:** Recommends "Inception" (sci-fi thriller + highly rated by similar users) ✓

---

### 6. Demo - Putting It All Together
**File:** [rs-ass2-and-ass3.ipynb:2](rs-ass2-and-ass3.ipynb#cell-1) (Lines 81-91)

```python
if __name__ == "__main__":
    user = random.choice(ratings.userId.unique())
    print(f"--- Recommendations for user {user} ---")

    print("Content-based (likes similar to last high-rated):")
    print(content_recommend(ratings[(ratings.userId==user)&(ratings.rating>=4)].movieId.iloc[-1]))

    print("\nCollaborative (SVD):")
    print(collab_recommend(user))

    print("\nHybrid (50-50 blend):")
    print(hybrid_recommend(user))
```

**What it does:**
1. Picks a random user from the dataset
2. Shows recommendations from all 3 methods for comparison

**Sample Output:**
```
--- Recommendations for user 77549 ---

Content-based (likes similar to last high-rated):
  movieId                              title
     2554      Children of the Damned (1963)
     2553       Village of the Damned (1960)
     5183  He Knows You're Alone (1979)

Collaborative (SVD):
  movieId                              title
      527          Schindler's List (1993)
      318   Shawshank Redemption, The (1994)
       50      Usual Suspects, The (1995)

Hybrid (50-50 blend):
  movieId                              title
      527          Schindler's List (1993)
      318   Shawshank Redemption, The (1994)
       50      Usual Suspects, The (1995)
```

**Observations:**
- **Content-based:** Focuses on genre similarity (all horror/thriller)
- **Collaborative:** Focuses on highly-rated classics
- **Hybrid:** Often similar to collaborative but with content influence

---

## Comparison: Assignment 1 vs Assignment 2

| Aspect | Assignment 1 (Crop Yield) | Assignment 2 (Movie Recommendations) |
|--------|---------------------------|--------------------------------------|
| **Problem Type** | Regression (predict numbers) | Ranking/Recommendation (order items) |
| **Target** | Continuous values (yield amount) | User preferences (ratings) |
| **Algorithms** | Random Forest, XGBoost | Content-based, Collaborative (SVD), Hybrid |
| **Input Features** | Numerical + categorical | Text (genres/tags) + user-item interactions |
| **Evaluation** | R², RMSE | User satisfaction (no explicit metric shown) |
| **Data Structure** | Tabular (rows = crops, cols = features) | User-item matrix + text descriptions |
| **Techniques** | Ensemble learning, decision trees | TF-IDF, matrix factorization, similarity search |

---

## Key Concepts Learned

### 1. Recommendation Systems
- **Content-Based:** Recommend based on item similarity
- **Collaborative Filtering:** Recommend based on user similarity
- **Hybrid:** Combine both approaches

### 2. Text Processing
- **TF-IDF:** Convert text to meaningful numbers
- **Vectorization:** Represent documents as vectors

### 3. Similarity Search
- **FAISS:** Fast similarity search at scale
- **Cosine Similarity:** Measure angle between vectors
- **Inner Product:** Efficient similarity metric

### 4. Matrix Factorization
- **SVD:** Decompose user-item matrix into latent factors
- **Latent Features:** Hidden patterns (e.g., "action lover", "drama fan")
- **Prediction:** Reconstruct missing ratings

### 5. Ensemble Strategies
- **Normalization:** Scale different scores to comparable ranges
- **Weighted Combination:** Blend multiple signals
- **50-50 blend:** Equal weight to content and collaborative

---

## Summary in Simple Terms

**Goal:** Recommend movies users will enjoy

**3 Approaches:**

1. **Content-Based Filtering**
   - "You liked sci-fi? Here's more sci-fi!"
   - Uses: TF-IDF + FAISS for similarity search

2. **Collaborative Filtering**
   - "People like you loved these classics!"
   - Uses: SVD matrix factorization

3. **Hybrid**
   - "Best of both: similar movies that popular users also loved!"
   - Uses: 50-50 blend of content + collaborative scores

**Technologies:**
- **TF-IDF:** Converts movie descriptions to numbers
- **FAISS:** Lightning-fast similarity search
- **SVD:** Finds hidden user-movie patterns
- **Hybrid:** Combines multiple recommendation strategies

**Real-world use:**
This is how Netflix, Amazon, Spotify recommend content to you!

---

## Advanced Topics (Optional Deep Dive)

### Cold Start Problem
- **New users:** No rating history → Collaborative fails → Use content-based
- **New movies:** No ratings → Collaborative fails → Use content-based
- **Solution:** Hybrid systems handle this better

### Scalability
- **FAISS:** Can handle millions of movies efficiently
- **SVD:** Needs retraining when new data arrives
- **Alternatives:** Online learning algorithms (incremental updates)

### Evaluation Metrics (Not shown in code)
- **Precision@K:** What % of top K recommendations are relevant?
- **Recall@K:** What % of relevant items appear in top K?
- **NDCG:** Normalized Discounted Cumulative Gain
- **MAE/RMSE:** Mean Absolute Error on rating predictions

### Improvements
- Try different blend ratios (not just 50-50)
- Add more features (director, actors, release year)
- Use deep learning (neural collaborative filtering)
- Incorporate implicit feedback (views, clicks, watch time)
