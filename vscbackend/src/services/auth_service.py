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
        # RSA Configuration
        self.JWT_ALGORITHM = "RS256"
        self.JWT_EXPIRATION = timedelta(days=90)
        self.JWT_PRIVATE_KEY_PATH = "/etc/ssl/jwt/private.pem"  # Updated secure path
        self.JWT_PUBLIC_KEY_PATH = "/etc/ssl/jwt/public.pem"    # Updated secure path

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
        """Generate JWT token using RSA private key"""
        try:
            with open(self.JWT_PRIVATE_KEY_PATH, "r") as f:
                private_key = f.read()
            
            payload = {
                "sub": email,
                "exp": datetime.utcnow() + self.JWT_EXPIRATION
            }
            return jwt.encode(payload, private_key, algorithm=self.JWT_ALGORITHM)
        
        except FileNotFoundError:
            raise RuntimeError("JWT private key not found at specified path")
        except IOError as e:
            raise RuntimeError(f"Error reading private key: {str(e)}")

    def verify_jwt(self, token: str) -> Optional[Dict]:
        """Verify JWT token using RSA public key"""
        try:
            with open(self.JWT_PUBLIC_KEY_PATH, "r") as f:
                public_key = f.read()
            
            return jwt.decode(
                token,
                public_key,
                algorithms=[self.JWT_ALGORITHM],
                options={"verify_aud": False}
            )
        
        except FileNotFoundError:
            raise RuntimeError("JWT public key not found at specified path")
        except IOError as e:
            raise RuntimeError(f"Error reading public key: {str(e)}")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

