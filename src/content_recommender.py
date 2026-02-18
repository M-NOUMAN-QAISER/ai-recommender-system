import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class ContentRecommender:
    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.vectorizer = None
    
    def load_data(self, filepath='data/raw/courses.csv'):
        """Load courses data"""
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} courses")
        return self.df
    
    def prepare_features(self):
        """Combine title + description + category + tags into one text field"""
        if self.df is None:
            raise ValueError("Load data first!")
        
        # Combine all text fields
        self.df['features'] = (
            self.df['title'].fillna('') + ' ' +
            self.df['description'].fillna('') + ' ' +
            self.df['category'].fillna('') + ' ' +
            self.df['tags'].fillna('')
        )
        print("Features prepared")
    
    def build_model(self):
        """Create TF-IDF matrix and cosine similarity"""
        if 'features' not in self.df.columns:
            self.prepare_features()
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['features'])
        
        # Cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        
        print("Model built (TF-IDF + cosine similarity)")
    
    def get_recommendations(self, item_id, top_n=5):
        """Get top N recommendations for given item_id"""
        if self.cosine_sim is None:
            self.build_model()
        
        # Find item index
        item_idx = self.df[self.df['id'] == item_id].index
        if len(item_idx) == 0:
            return {"error": f"Item {item_id} not found"}
        
        item_idx = item_idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[item_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (exclude itself)
        sim_scores = sim_scores[1:top_n+1]
        
        rec_indices = [i[0] for i in sim_scores]
        rec_scores = [i[1] for i in sim_scores]
        
        recommendations = self.df.iloc[rec_indices][['id', 'title', 'category']].copy()
        recommendations['similarity_score'] = rec_scores
        
        return recommendations.to_dict('records')


# Global recommender instance (for API)
recommender = ContentRecommender()
