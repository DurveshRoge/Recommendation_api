from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load preprocessed datasets
course_df = pd.read_csv("cleaned_udemy_course_data.csv")
interaction_df = pd.read_csv("processed_data/train_interactions.csv")

# Ensure required columns exist
if 'user_id' not in interaction_df.columns or 'course_id' not in interaction_df.columns:
    raise ValueError("Missing required columns in interaction data.")
if 'course_id' not in course_df.columns or 'course_title' not in course_df.columns:
    raise ValueError("Missing required columns in course data.")

# Create user-item interaction matrix
interaction_matrix = interaction_df.pivot(index='user_id', columns='course_id', values='rating').fillna(0)
interaction_sparse = csr_matrix(interaction_matrix)

# Train collaborative filtering model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(interaction_sparse)

# Train content-based filtering model
if "course_title" in course_df.columns:
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(course_df['course_title'].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
else:
    cosine_sim = None

# Collaborative Filtering Recommendation
def recommend_courses(user_id, num_recommendations=5):
    if user_id not in interaction_matrix.index:
        return []
    
    user_index = interaction_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(interaction_sparse[user_index], n_neighbors=num_recommendations+1)
    recommended_courses = interaction_matrix.iloc[indices[0][1:]].index.tolist()
    return [int(course_id) for course_id in recommended_courses]

# Content-Based Filtering Recommendation
def recommend_similar_courses(course_id, num_recommendations=5):
    if cosine_sim is None or course_id not in course_df['course_id'].values:
        return []
    
    course_index = course_df[course_df['course_id'] == course_id].index.tolist()
    if not course_index:
        return []
    
    course_index = course_index[0]
    sim_scores = list(enumerate(cosine_sim[course_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    recommended_courses = [int(course_df.iloc[i[0]]['course_id']) for i in sim_scores]
    return recommended_courses

# Hybrid Recommendation
def hybrid_recommendation(user_id, num_recommendations=5):
    collab_recommendations = recommend_courses(user_id, num_recommendations)
    content_recommendations = []
    
    for course_id in collab_recommendations:
        content_recommendations.extend(recommend_similar_courses(course_id, 2))
    
    final_recommendations = list(set(collab_recommendations + content_recommendations))[:num_recommendations]
    return [int(course_id) for course_id in final_recommendations]

# API Endpoints
@app.route('/')
def home():
    return jsonify({"message": "Recommendation API is running!"}), 200

@app.route('/recommend/collaborative', methods=['GET'])
def collaborative_recommend():
    try:
        user_id = int(request.args.get('user_id'))
        recommendations = recommend_courses(user_id)
        return jsonify({'user_id': user_id, 'recommended_courses': recommendations})
    except ValueError:
        return jsonify({'error': 'Invalid user_id. Must be an integer.'}), 400

@app.route('/recommend/content', methods=['GET'])
def content_recommend():
    try:
        course_id = int(request.args.get('course_id'))
        recommendations = recommend_similar_courses(course_id)
        return jsonify({'course_id': course_id, 'recommended_courses': recommendations})
    except ValueError:
        return jsonify({'error': 'Invalid course_id. Must be an integer.'}), 400

@app.route('/recommend/hybrid', methods=['GET'])
def hybrid_recommend():
    try:
        user_id = int(request.args.get('user_id'))
        recommendations = hybrid_recommendation(user_id)
        return jsonify({'user_id': user_id, 'recommended_courses': recommendations})
    except ValueError:
        return jsonify({'error': 'Invalid user_id. Must be an integer.'}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides a dynamic port
    app.run(host='0.0.0.0', port=port, debug=True)
