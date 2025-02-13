import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure processed_data directory exists
os.makedirs("processed_data", exist_ok=True)

# Load datasets
course_df = pd.read_csv("cleaned_udemy_course_data.csv")
interaction_df = pd.read_csv("user_course_interactions.csv")

# Normalize numerical features (excluding 'rating')
numeric_cols = ["num_subscribers", "num_reviews", "price"]
scaler = MinMaxScaler()
if all(col in course_df.columns for col in numeric_cols):
    course_df[numeric_cols] = scaler.fit_transform(course_df[numeric_cols])
    
    # Log transform to reduce skewness
    course_df["num_subscribers"] = np.log1p(course_df["num_subscribers"])
    course_df["num_reviews"] = np.log1p(course_df["num_reviews"])
else:
    print("Warning: Missing numeric columns in course data.")

# Create engagement score
if "time_spent" in interaction_df.columns and "course_duration" in interaction_df.columns:
    interaction_df["engagement_score"] = interaction_df["time_spent"] / interaction_df["course_duration"]
    interaction_df["engagement_score"].fillna(0, inplace=True)

# Remove duplicate (user_id, course_id) pairs by averaging ratings
interaction_df = interaction_df.groupby(["user_id", "course_id"], as_index=False).agg({"rating": "mean"})

# Filter users with at least 5 interactions
interaction_df = interaction_df.groupby("user_id").filter(lambda x: len(x) >= 5)

# Train-test split
train_data, test_data = train_test_split(interaction_df, test_size=0.2, random_state=42)

# Create User-Item Interaction Matrix
interaction_matrix = train_data.pivot(index='user_id', columns='course_id', values='rating').fillna(0)
interaction_sparse = csr_matrix(interaction_matrix)

# Train KNN Model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(interaction_sparse)

# Collaborative Filtering Recommendation
def recommend_courses(user_id, num_recommendations=5):
    if user_id not in interaction_matrix.index:
        return []
    
    user_index = interaction_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(interaction_sparse[user_index], n_neighbors=min(num_recommendations+1, len(interaction_matrix)))
    
    # Ensure index does not go out of bounds
    recommended_indices = [i for i in indices[0][1:] if i < len(interaction_matrix.columns)]
    
    recommended_course_ids = [int(interaction_matrix.columns[i]) for i in recommended_indices]
    return recommended_course_ids


# Content-Based Filtering (TF-IDF + Cosine Similarity using course_title)
if "course_title" in course_df.columns:
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = tfidf.fit_transform(course_df['course_title'].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
else:
    print("Warning: 'course_title' column missing in course data.")
    cosine_sim = None

def recommend_similar_courses(course_id, num_recommendations=5):
    if cosine_sim is None or course_id not in course_df['course_id'].values:
        return []
    
    course_index = course_df.index[course_df['course_id'] == course_id].tolist()
    if not course_index:
        return []
    
    course_index = course_index[0]
    sim_scores = list(enumerate(cosine_sim[course_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    return [int(course_df.iloc[i[0]]['course_id']) for i in sim_scores]

# Hybrid Recommendation
def hybrid_recommendation(user_id, num_recommendations=5):
    collab_recommendations = recommend_courses(user_id, num_recommendations)
    content_recommendations = []
    
    for course_id in collab_recommendations:
        content_recommendations.extend(recommend_similar_courses(course_id, 2))
    
    return list(map(int, set(collab_recommendations + content_recommendations)))[:num_recommendations]

# Save processed data
train_data.to_csv("processed_data/train_interactions.csv", index=False)
test_data.to_csv("processed_data/test_interactions.csv", index=False)

print("Training complete. Models are ready!")
