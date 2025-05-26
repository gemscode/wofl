# backend/src/services/auth_service.py
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
import bcrypt
import re
import os
import jwt
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class AuthService:
    def __init__(self, session):
        self.session = session
        self.JWT_SECRET = os.getenv("JWT_SECRET")
        self.JWT_ALGORITHM = "HS256"
        self.JWT_EXPIRATION = timedelta(days=90)

        # Initialize Cassandra schema
        self._initialize_schema()

    def _initialize_schema(self):
        """Create required Cassandra tables if they don't exist"""
        self.session.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "email text PRIMARY KEY, "
            "password text, "
            "created_at timestamp)"
        )

    def validate_email(self, email: str) -> bool:
        """Validate email format using regex"""
        return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email) is not None

    def validate_password(self, password: str) -> bool:
        """Validate password meets complexity requirements"""
        return (len(password) >= 8 and
                re.search(r"[A-Z]", password) and
                re.search(r"[a-z]", password) and
                re.search(r"\d", password) and
                re.search(r"[!@#$%^&*(),.?\":{}|<>]", password))

    def register_user(self, email: str, password: str) -> Tuple[Dict, int]:
        """Register a new user with email and password"""
        try:
            # Validate inputs
            if not self.validate_email(email):
                return {"error": "Invalid email format"}, 400
                
            if not self.validate_password(password):
                return {"error": "Password must be 8+ characters with uppercase, lowercase, number, and symbol"}, 400

            # Check if user exists
            query = "SELECT email FROM users WHERE email = %s"
            existing = self.session.execute(query, [email]).one()
            if existing:
                return {"error": "Email already registered"}, 409

            # Hash password
            hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

            # Store user in Cassandra
            self.session.execute(
                "INSERT INTO users (email, password, created_at) VALUES (%s, %s, %s)",
                [email, hashed_pw, datetime.now()]
            )

            # Generate JWT token
            token = self._generate_jwt(email)
            return {"token": token}, 201

        except Exception as e:
            return {"error": str(e)}, 500

    def login_user(self, email: str, password: str) -> Tuple[Dict, int]:
        """Authenticate existing user"""
        try:
            # Get user from database
            query = "SELECT * FROM users WHERE email = %s"
            user = self.session.execute(query, [email]).one()

            if not user:
                return {"error": "Invalid credentials"}, 401

            # Verify password
            if not bcrypt.checkpw(password.encode(), user.password.encode()):
                return {"error": "Invalid credentials"}, 401

            # Generate JWT token
            token = self._generate_jwt(email)
            return {"token": token}, 200

        except Exception as e:
            return {"error": str(e)}, 500

    def _generate_jwt(self, email: str) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "sub": email,
            "exp": datetime.utcnow() + self.JWT_EXPIRATION
        }
        return jwt.encode(payload, self.JWT_SECRET, algorithm=self.JWT_ALGORITHM)

    def verify_jwt(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload if valid"""
        try:
            payload = jwt.decode(
                token,
                self.JWT_SECRET,
                algorithms=[self.JWT_ALGORITHM],
                options={"verify_aud": False}  # Optional: Disable audience check
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

