from flask import Flask, jsonify, request
from content_recommender import recommender
import os

app = Flask(__name__)

@app.route("/")
def index():
    return {
        "message": "AI Content-Based Recommender System 🚀",
        "endpoints": {
            "recommend": "/recommend/<item_id>?top_n=5",
            "health": "/health"
        },
        "example": "GET /recommend/1"
    }

@app.route("/health")
def health():
    return {"status": "healthy", "model_loaded": recommender.df is not None}

@app.route("/recommend/<int:item_id>")
def get_recommendations(item_id):
    try:
        top_n = request.args.get('top_n', default=5, type=int)
        recommendations = recommender.get_recommendations(item_id, top_n)
        
        return jsonify({
            "item_id": item_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("Loading recommender model...")
    recommender.load_data()
    recommender.build_model()
    print("Model ready! Starting API...")
    app.run(debug=True, host='127.0.0.1', port=5000)
