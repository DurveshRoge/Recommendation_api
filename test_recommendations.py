import pandas as pd
from train_model import recommend_courses, recommend_similar_courses, hybrid_recommendation

# Load processed training data
train_data = pd.read_csv("processed_data/train_interactions.csv")

# Check if data is loaded correctly
print("First 5 rows of training data:\n", train_data.head())

# Get a sample user_id and course_id from training data
sample_user_id = train_data["user_id"].iloc[0]
sample_course_id = train_data["course_id"].iloc[0]

# Test Collaborative Filtering (KNN)
print(f"\nRecommended courses for user {sample_user_id}:")
print(recommend_courses(sample_user_id, num_recommendations=5))

# Test Content-Based Filtering (TF-IDF)
print(f"\nSimilar courses for course {sample_course_id}:")
print(recommend_similar_courses(sample_course_id, num_recommendations=5))

# Test Hybrid Recommendation
print(f"\nHybrid recommendations for user {sample_user_id}:")
print(hybrid_recommendation(sample_user_id, num_recommendations=5))
