# Quickstart Guide: AI Task Management Chatbot

## Overview
This guide provides instructions for setting up and running the AI Task Management Chatbot locally.

## Prerequisites
- Python 3.11+
- Node.js 18+ and npm/yarn
- PostgreSQL (or access to Neon Serverless PostgreSQL)
- OpenAI API key
- (Optional) Google API key for Gemini access

## Environment Setup

### Backend (Python)
1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://user:password@host:port/database_name
NEON_DATABASE_URL=your_neon_database_url
SECRET_KEY=your_jwt_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
GEMINI_API_KEY=your_gemini_api_key_optional
```

### Frontend (Next.js)
1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Set up environment variables by creating a `.env.local` file:
```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api
NEXT_PUBLIC_JWT_SECRET=your_jwt_secret_key_same_as_backend
```

## Database Setup

1. Run database migrations:
```bash
cd backend
# Activate virtual environment if not already done
alembic upgrade head
```

## Running the Application

### Backend
1. Start the FastAPI server:
```bash
cd backend
# Activate virtual environment
uvicorn app.main:app --reload --port 8000
```

2. In a separate terminal, start the MCP server:
```bash
cd backend/mcp_server
python server.py
```

### Frontend
1. Start the Next.js development server:
```bash
cd frontend
npm run dev
# or
yarn dev
```

2. Open your browser to `http://localhost:3000`

## API Endpoints

### Backend API
- `POST /api/message` - Send a message to the chatbot
- `GET /api/history` - Get conversation history
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration

### MCP Server
- Exposes task management tools as defined in the MCP specification

## Development Commands

### Backend
- Run tests: `pytest`
- Format code: `black .`
- Lint: `flake8`

### Frontend
- Run tests: `npm test` or `yarn test`
- Build: `npm run build` or `yarn build`
- Lint: `npm run lint` or `yarn lint`

## Troubleshooting

### Common Issues
1. **Database Connection**: Ensure PostgreSQL is running and credentials are correct
2. **API Keys**: Verify all required API keys are set in environment variables
3. **Port Conflicts**: Check that ports 8000 (backend) and 3000 (frontend) are available

### Verification Steps
1. Confirm backend is running: `curl http://localhost:8000/health`
2. Confirm MCP server is running: Check logs for successful startup
3. Confirm frontend is running: Visit `http://localhost:3000` in browser