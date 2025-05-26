from flask import Flask, request, jsonify
from services.auth_service import AuthService
from utils.database import get_cassandra_session
from services.llm_service import GroqService

app = Flask(__name__)

# Initialize services
session = get_cassandra_session()
auth_service = AuthService(session)

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify(error="Invalid request format"), 400
            
        return auth_service.register_user(data['email'], data['password'])
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify(error="Invalid request format"), 400
            
        return auth_service.login_user(data['email'], data['password'])
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    # Verify JWT
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    payload = auth_service.verify_jwt(token)  # <-- Use auth_service instance
    if not payload:
        return jsonify(error="Invalid token"), 401

    # Get request data
    data = request.get_json()
    prompt = data.get('prompt', '')
    model = data.get('model', 'llama3-70b-8192')

    # Process prompt
    llm = GroqService()
    response = llm.generate_response(prompt, model)

    return jsonify(response=response)

@app.errorhandler(404)
def handle_404(e):
    return jsonify(error="Endpoint not found"), 404

@app.errorhandler(500)
def handle_500(e):
    return jsonify(error="Internal server error"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

