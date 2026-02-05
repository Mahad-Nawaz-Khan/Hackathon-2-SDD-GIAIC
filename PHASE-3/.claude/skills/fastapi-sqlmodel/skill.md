# FastAPI + SQLModel Backend Scaffold Skill

This skill generates a starter FastAPI project that uses **SQLModel** for the ORM. It includes:

**Features**
- FastAPI route organization with APIRouter modules
- Dependency injection for DB sessions and common services
- SQLModel table definitions with relationships
- Separate Pydantic models for request validation and response schemas
- Fully async using `async def` endpoints and `async_engine`
- Middleware configuration (CORS, GZip, custom error handling)
- Global exception handlers that return standard JSON error responses
- Response models typed with Pydantic for OpenAPI docs
- Background tasks example (e.g., sending email after a DB write)
- API documentation best‑practice settings (tags, summary, description)

**Usage**
Run the skill with a description of the resources you need, for example:
```
Create a `User` table with `id`, `email`, `hashed_password`, and a one‑to‑many relationship to `Post`. Add routes for creating, reading, updating, and deleting users and posts.
```
The skill will output a directory structure and starter files:
```
my_fastapi_app/
├─ app/
│  ├─ api/
│  │  ├─ routers/
│  │  │  ├─ users.py
│  │  │  └─ posts.py
│  │  └─ dependencies.py
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ database.py
│  │  └─ security.py
│  ├─ models/
│  │  ├─ tables.py          # SQLModel table definitions
│  │  └─ schemas.py         # Pydantic request/response models
│  ├─ main.py               # FastAPI app instance, middleware, include routers
│  └─ background.py         # Example background task functions
└─ requirements.txt
```

**Generated Code Highlights**

*`app/models/tables.py`*
```python
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, nullable=False, unique=True)
    hashed_password: str
    posts: List["Post"] = Relationship(back_populates="owner")

class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    owner_id: int = Field(foreign_key="user.id")
    owner: User = Relationship(back_populates="posts")
```

*`app/models/schemas.py`*
```python
from pydantic import BaseModel, EmailStr
from typing import List

class PostRead(BaseModel):
    id: int
    title: str
    content: str
    class Config:
        orm_mode = True

class UserRead(BaseModel):
    id: int
    email: EmailStr
    posts: List[PostRead] = []
    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str
```

*`app/core/database.py`*
```python
from sqlmodel import SQLModel, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(DATABASE_URL, echo=True)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
```

*`app/api/routers/users.py`*
```python
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_session
from app.models.tables import User
from app.models.schemas import UserCreate, UserRead

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    db_user = User(email=payload.email, hashed_password=hash_password(payload.password))
    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    background_tasks.add_task(send_welcome_email, db_user.email)
    return db_user
```

**Middleware Example (`app/main.py`)**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import users, posts
from app.core.database import init_db

app = FastAPI(title="FastAPI + SQLModel Starter", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    await init_db()

app.include_router(users.router)
app.include_router(posts.router)
```

---
**Customisation**
- Adjust `DATABASE_URL` for PostgreSQL, MySQL, etc.
- Add additional middleware (authentication, logging) in `app/main.py`.
- Extend `background.py` for other async background jobs.

**Running the Project**
```bash
pip install fastapi[all] sqlmodel uvicorn
uvicorn app.main:app --reload
```

The skill is now ready to scaffold a complete FastAPI + SQLModel backend based on the description you provide.
