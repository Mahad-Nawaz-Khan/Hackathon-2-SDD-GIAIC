# TODO Application with Clerk Authentication

A full-stack TODO application with Next.js frontend, FastAPI backend, and Clerk authentication.

## Features

- User authentication and authorization with Clerk
- Create, read, update, and delete tasks
- Task prioritization (Low, Medium, High)
- Due dates and recurrence rules
- Tagging system for tasks
- Responsive design

## Tech Stack

- Frontend: Next.js (App Router), React, Tailwind CSS
- Backend: FastAPI, SQLModel, PostgreSQL
- Authentication: Clerk
- Database: PostgreSQL (Neon)

## Prerequisites

- Node.js (v18 or higher)
- Python (v3.9 or higher)
- PostgreSQL database (or Neon account)

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the backend root directory with the following content:
   ```env
   DATABASE_URL=postgresql://username:password@localhost:5432/todo_db
   CLERK_SECRET_KEY=your_clerk_secret_key
   CLERK_JWT_KEY=your_clerk_jwt_key
   ```

6. Run the backend server:
   ```bash
   python -m uvicorn src.main:app --reload
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env.local` file in the frontend root directory with the following content:
   ```env
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
   NEXT_PUBLIC_CLERK_FRONTEND_API_URL=http://localhost:3000
   NEXT_PUBLIC_BACKEND_API_URL=http://localhost:8000
   ```

4. Run the development server:
   ```bash
   npm run dev
   ```

## Environment Variables

### Backend (.env)
- `DATABASE_URL`: PostgreSQL database connection string
- `CLERK_SECRET_KEY`: Clerk secret key for backend verification
- `CLERK_JWT_KEY`: Clerk JWT key for token verification

### Frontend (.env.local)
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`: Clerk publishable key for frontend
- `NEXT_PUBLIC_BACKEND_API_URL`: URL of the backend API

## API Endpoints

### Authentication
- `GET /api/v1/auth/me` - Get current user info
- `POST /api/v1/auth/login` - Login user
- `POST /api/v1/auth/logout` - Logout user

### Tasks
- `GET /api/v1/tasks` - Get all tasks for authenticated user
- `POST /api/v1/tasks` - Create a new task
- `GET /api/v1/tasks/{id}` - Get a specific task
- `PUT /api/v1/tasks/{id}` - Update a specific task
- `DELETE /api/v1/tasks/{id}` - Delete a specific task
- `PATCH /api/v1/tasks/{id}/toggle-completion` - Toggle task completion

### Tags
- `GET /api/v1/tags` - Get all tags for authenticated user
- `POST /api/v1/tags` - Create a new tag
- `GET /api/v1/tags/{id}` - Get a specific tag
- `PUT /api/v1/tags/{id}` - Update a specific tag
- `DELETE /api/v1/tags/{id}` - Delete a specific tag

## Running Tests

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Deployment

### Backend
The backend can be deployed to platforms like Heroku, Railway, or AWS. Make sure to set the environment variables in the deployment environment.

### Frontend
The frontend can be deployed to Vercel, Netlify, or similar platforms. Configure the environment variables during deployment.

## Development

### Backend Development
Run the backend with auto-reload:
```bash
uvicorn src.main:app --reload
```

### Frontend Development
Run the frontend with hot reload:
```bash
npm run dev
```

## Security Features

- JWT token verification with Clerk
- User data isolation (users can only access their own data)
- Rate limiting on API endpoints
- Input validation using Pydantic models
- SQL injection prevention with SQLModel
- Proper error handling without exposing sensitive information

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License.