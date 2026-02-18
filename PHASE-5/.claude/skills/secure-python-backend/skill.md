# Secure Python Backend Development Skill

This skill scaffolds a secure FastAPI (or Flask) backend in Python with best‑practice security components.

## Included features
- **JWT token validation** using `python‑jose`
- **Password hashing** with `passlib` (bcrypt)
- **Rate limiting** via `slowapi`
- **CORS configuration** (configurable origins)
- **Request validation** using Pydantic models
- **Secure HTTP headers** (Helmet‑style via `secure` library)
- **Environment variable management** with `python‑dotenv`
- **Cryptography best practices** (key generation, secret handling)
- **API endpoint protection patterns** (dependency injection, scopes)

## Usage
Run the skill and describe the resources you need, e.g.:
```
Create a `User` model with JWT‑protected CRUD endpoints, rate‑limited to 100 req/min, CORS allowed for https://example.com, and environment‑based secret keys.
```
The skill will generate a directory structure:
```
secure-backend/
├─ app/
│  ├─ main.py                # FastAPI app, middleware, CORS, security headers
│  ├─ core/
│  │  ├─ config.py           # Loads env vars via python‑dotenv
│  │  ├─ security.py        # JWT encode/decode, password hashing, token revocation
│  │  └─ rate_limit.py      # SlowAPI limiter configuration
│  ├─ models/
│  │  ├─ tables.py          # SQLModel/SQLAlchemy tables
│  │  └─ schemas.py         # Pydantic request/response models
│  ├─ api/
│  │  ├─ routers/
│  │  │  └─ users.py        # Endpoints with Depends(security.get_current_user)
│  │  └─ dependencies.py    # DB session, security dependencies
│  └─ utils/
│     └─ crypto.py          # Helper for key generation, encryption
└─ requirements.txt
```

## Generated code highlights
### `app/core/config.py`
```python
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    SECRET_KEY: str = os.getenv('SECRET_KEY')
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
    CORS_ORIGINS: list[str] = os.getenv('CORS_ORIGINS', '*').split(',')
```
### `app/core/security.py`
```python
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from .config import Settings

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=Settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Settings.SECRET_KEY, algorithm=Settings.ALGORITHM)
    return encoded_jwt
```
### `app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from secure import SecureHeaders
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from app.api.routers import users
from app.core.config import Settings

app = FastAPI(title="Secure Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Secure headers
secure_headers = SecureHeaders()
app.middleware('http')(secure_headers.middleware)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

app.include_router(users.router)
```

## How to run
```bash
pip install fastapi uvicorn python-jose passlib[bcrypt] slowapi secure python-dotenv sqlmodel
uvicorn app.main:app --reload
```

---
**Note**: Adjust `requirements.txt` and environment variables (`.env`) as needed for your deployment.
