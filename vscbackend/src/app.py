from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS 
from services.auth_service import AuthService
from utils.database import get_cassandra_session
from services.llm_service import GroqService
from werkzeug.serving import run_simple
import os
import logging
from services.llm_middleware import LLMMiddleware
from services.llm_middleware_v2 import LLMMiddlewareV2

middleware_v2 = LLMMiddlewareV2()
middleware = LLMMiddleware()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enhanced CORS configuration for Cloudflare
CORS(app, resources={
    r"/rw/*": {
        "origins": [
            "https://wolfx0.com",          # Your domain
            "vscode-webview://*",          # VSCode WebView
            "https://*.cloudflare.com"     # Cloudflare proxies
        ],
        "allow_headers": ["Authorization", "Content-Type"],
        "methods": ["GET", "POST", "OPTIONS"],
        "supports_credentials": True
    }
})

# Initialize services with enhanced security
session = get_cassandra_session()
auth_service = AuthService(session)

# Create blueprint for /rw endpoints
rw_bp = Blueprint('rw', __name__, url_prefix='/rw')

@rw_bp.route('/register', methods=['POST'], strict_slashes=False)
def register():
    """Handle user registration with input validation"""
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify(error="Invalid request format"), 400
            
        # Additional validation
        if len(data['password']) < 10:
            return jsonify(error="Password must be at least 10 characters"), 400
            
        return auth_service.register_user(data['email'], data['password'])
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify(error="Internal server error"), 500

@rw_bp.route('/login', methods=['POST'], strict_slashes=False)
def login():
    """Handle user login with security logging"""
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify(error="Invalid request format"), 400
            
        result, status = auth_service.login_user(data['email'], data['password'])
        if status == 200:
            logger.info(f"Successful login for {data['email']}")
        else:
            logger.warning(f"Failed login attempt for {data['email']}")
            
        return result, status
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify(error="Internal server error"), 500

# app.py - Updated handle_prompt() function
@rw_bp.route('/prompt', methods=['POST'])
def handle_prompt():
    """Process LLM prompts with conversation context"""
    try:
        # JWT Verification
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            logger.warning("Missing or invalid Authorization header")
            return jsonify(error="Unauthorized"), 401

        token = auth_header.split(' ')[1]
        payload = auth_service.verify_jwt(token)
        if not payload:
            logger.warning(f"Invalid JWT token: {token[:15]}...")
            return jsonify(error="Invalid token"), 401

        # Input validation
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify(error="Prompt required"), 400

        if len(data['prompt']) > 2000:
            return jsonify(error="Prompt too long"), 413

        # Extract thread parameters
        thread_id = data.get('thread_id')
        new_thread = data.get('new_thread', False)  

        # Process request with middleware
        #middleware = LLMMiddleware()
        response, new_thread_id = middleware.generate_response(
            prompt=data['prompt'],
            model=data.get('model', 'llama3-70b-8192'),
            thread_id=thread_id,
            new_thread=new_thread
        )

        # Get thread/session parameters
        #session_id = data.get('thread_id')  # Rename to match middleware
        #new_thread = data.get('new_thread', False)

        # Reset session if new_thread requested
        #if new_thread:
        #    session_id = None

        # Process with middleware (single instance)
        #response, new_thread_id = middleware_v2.generate_response(
        #    prompt=data['prompt'],
        #    session_id=session_id
        #)

        return jsonify({
            "response": response,
            "thread_id": new_thread_id,
            "new_thread": new_thread
        })

    except Exception as e:
        logger.error(f"Prompt processing error: {str(e)}")
        return jsonify(error="Internal server error"), 500

# Register the blueprint
app.register_blueprint(rw_bp)

@app.errorhandler(404)
def handle_404(e):
    return jsonify(error="Endpoint not found"), 404

@app.errorhandler(500)
def handle_500(e):
    logger.error("Internal server error", exc_info=True)
    return jsonify(error="Internal server error"), 500

if __name__ == '__main__':
    socket_path = '/tmp/woflx0.sock'
    
    # Cleanup existing socket
    if os.path.exists(socket_path):
        os.remove(socket_path)
    
    # Production-grade server configuration
    run_simple(
        'unix://' + socket_path,
        0,
        app,
        use_reloader=False,
        use_debugger=False,
        threaded=True,
        ssl_context=(
            '/etc/ssl/cloudflare/cert.pem', 
            '/etc/ssl/cloudflare/key.pem'
        ) if os.path.exists('/etc/ssl/cloudflare/cert.pem') else None
    )

