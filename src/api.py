from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return "AI Recommender System is running! 🚀"


@app.route("/recommend/<item_id>")
def get_recommendations(item_id):
    # TODO: later replace this with your real recommender logic
    recommendations = [
        {"item_id": "101", "title": "Intro to Python", "score": 0.95},
        {"item_id": "102", "title": "Data Science Basics", "score": 0.87},
        {"item_id": "103", "title": "Machine Learning 101", "score": 0.83},
    ]
    return jsonify({
        "item_id": item_id,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)
