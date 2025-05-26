import bcrypt
import re
from datetime import datetime, timedelta
import jwt

def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def validate_password(password):
    return (len(password) >= 8 and
            re.search(r"[A-Z]", password) and
            re.search(r"[a-z]", password) and
            re.search(r"\d", password) and
            re.search(r"[!@#$%^&*(),.?\":{}|<>]", password))

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_jwt_token(email, secret):
    return jwt.encode({
        'email': email,
        'exp': datetime.utcnow() + timedelta(days=90)
    }, secret, algorithm='HS256')

def verify_jwt_token(token, secret):
    try:
        return jwt.decode(token, secret, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

