# utils/security.py
import bcrypt
import re
import jwt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict

# RSA Configuration
DEFAULT_PRIVATE_KEY_PATH = "/etc/ssl/jwt/private.pem"
DEFAULT_PUBLIC_KEY_PATH = "/etc/ssl/jwt/public.pem"

def validate_email(email: str) -> bool:
    """Validate email format using RFC 5322 compliant regex"""
    return re.match(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$",
        email
    ) is not None

def validate_password(password: str) -> bool:
    """Validate password meets complexity requirements"""
    return (
        len(password) >= 10 and
        re.search(r"[A-Z]", password) and
        re.search(r"[a-z]", password) and
        re.search(r"\d", password) and
        re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password)
    )

def hash_password(password: str) -> str:
    """Generate bcrypt hash for password"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash"""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_jwt_token(email: str, private_key_path: str = DEFAULT_PRIVATE_KEY_PATH) -> str:
    """
    Create RS256 signed JWT token
    Args:
        email: User's email address
        private_key_path: Path to RSA private key
    Returns:
        Encoded JWT token
    Raises:
        FileNotFoundError: If private key is missing
        ValueError: For invalid key format
    """
    try:
        with open(private_key_path, "r") as f:
            private_key = f.read()
            
        return jwt.encode(
            {
                "sub": email,
                "exp": datetime.utcnow() + timedelta(days=90),
                "iat": datetime.utcnow()
            },
            private_key,
            algorithm="RS256"
        )
    
    except FileNotFoundError:
        raise RuntimeError(f"Private key not found at {private_key_path}")
    except IOError as e:
        raise RuntimeError(f"Key read error: {str(e)}")

def verify_jwt_token(token: str, public_key_path: str = DEFAULT_PUBLIC_KEY_PATH) -> Optional[Dict]:
    """
    Verify RS256 signed JWT token
    Args:
        token: JWT token to verify
        public_key_path: Path to RSA public key
    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        with open(public_key_path, "r") as f:
            public_key = f.read()
            
        return jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
    
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
    except FileNotFoundError:
        raise RuntimeError(f"Public key not found at {public_key_path}")
    except IOError as e:
        raise RuntimeError(f"Key read error: {str(e)}")

