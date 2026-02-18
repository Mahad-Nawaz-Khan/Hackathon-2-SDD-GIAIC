# Quickstart Guide: TODO Application (Full-Stack Web) with Authentication

## Prerequisites

- Node.js 18+ (for frontend)
- Python 3.11+ (for backend)
- PostgreSQL client tools
- Clerk account and API keys
- Git

## Setup Instructions

### 1. Clone and Initialize Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Clerk and database configuration
   ```

5. Run database migrations:
   ```bash
   python -m src.main init_db
   ```

6. Start the backend server:
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```

### 3. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your Clerk and backend API configuration
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

### 4. Clerk Configuration

1. Create a Clerk account at [clerk.com](https://clerk.com)
2. Create a new application
3. Configure your frontend and backend origins:
   - Frontend: `http://localhost:3000`
   - Backend: `http://localhost:8000`
4. Add the API keys to your `.env` files

## Running the Application

1. Start the backend server (port 8000)
2. Start the frontend server (port 3000)
3. Access the application at `http://localhost:3000`
4. Sign up or sign in using Clerk's authentication flow
5. Create and manage your personal tasks

## Key Endpoints

### Backend API (http://localhost:8000)
- `/api/v1/tasks` - Task management endpoints
- `/api/v1/auth/me` - Current user information
- `/api/v1/tags` - Tag management endpoints

### Frontend Pages
- `/` - Task dashboard
- `/tasks/[id]` - Individual task view
- `/settings` - User settings

## Development Commands

### Backend
- Run tests: `pytest`
- Format code: `black . && isort .`
- Lint code: `flake8`

### Frontend
- Run tests: `npm test`
- Format code: `npm run format`
- Lint code: `npm run lint`
- Build for production: `npm run build`

## Troubleshooting

### Common Issues
1. **Clerk authentication not working**: Verify your Clerk API keys and allowed origins
2. **Database connection errors**: Check your database URL in the environment variables
3. **CORS errors**: Ensure your frontend URL is added to the backend's CORS configuration