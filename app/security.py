import bcrypt




def hash_password(password: str) -> str:
    # Convertir en bytes et hasher
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')  # Stocker comme string

def verify_password(password: str, hashed: str) -> bool:
    pwd_bytes = password.encode('utf-8')
    hashed_bytes = hashed.encode('utf-8')
    return bcrypt.checkpw(pwd_bytes, hashed_bytes)