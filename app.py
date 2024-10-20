from flask import Flask, request, jsonify, render_template
import urllib3
from main import agent_executor  # Import the agent_executor from app.py

# Initialize Flask app
app = Flask(__name__)

# Define Flask routes
@app.route('/')
def home():
    return render_template('index.html')  # Serves the HTML file

@app.route('/ask', methods=['POST'])
def ask():
    """Endpoint to process the user's question."""
    try:
        data = request.get_json()
        query = data.get('query', '')

        # Using the ADGPT agent to process the input query
        response = agent_executor.run(query)

        # Simulating steps (in real scenarios, you might have multiple steps)
        steps = [
            {"title": "Step 1", "content": "Processing the query."},
            {"title": "Step 2", "content": "Fetching relevant information."},
        ]

        # Final response from the agent
        final_answer = response

        return jsonify({
            "steps": steps,
            "finalAnswer": final_answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    urllib3.disable_warnings()
    app.run(host='0.0.0.0', port=5000, debug=True)
