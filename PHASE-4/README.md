# AI-Powered Task Management Chatbot

A full-stack AI-powered task management application with Next.js frontend, FastAPI backend, OpenAI Agents SDK integration, MCP server, Clerk authentication, and comprehensive rate limiting for security.

## Features

- **AI-Powered Chat Interface**: Natural language task management via conversational AI
- **OpenAI Agents SDK Integration**: Advanced AI processing with streaming responses
- **MCP Server**: Model Context Protocol server for standardized tool access
- **Server-Sent Events (SSE)**: Real-time streaming responses for better UX
- User authentication and authorization with Clerk (JWT-based)
- Create, read, update, and delete tasks via natural language
- Task prioritization (HIGH, MEDIUM, LOW)
- Due dates and task management
- Tagging system for tasks with many-to-many relationships
- Advanced filtering, sorting, and search capabilities
- Rate limiting on all API endpoints to prevent abuse
- Responsive dark-themed design
- Optimistic UI updates for instant interactions

## Tech Stack

- **Frontend**: Next.js (App Router), React 19, Tailwind CSS v4
- **Backend**: FastAPI, SQLModel, PostgreSQL (Neon)
- **AI**: OpenAI Agents SDK (Python), GPT-4o-mini model
- **MCP**: Official MCP Python SDK with FastMCP
- **Authentication**: Clerk (JWT-based)
- **Rate Limiting**: slowapi
- **Streaming**: Server-Sent Events (SSE)
- **Database**: PostgreSQL (Neon Serverless)

## Architecture

```
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│   Next.js       │──────│   FastAPI Backend   │──────│  PostgreSQL     │
│   Frontend      │ SSE  │                     │      │  (Neon)         │
└─────────────────┘      │  ┌───────────────┐  │      └─────────────────┘
                          │  │ OpenAI Agents│  │
                          │  │ SDK Service  │  │
                          │  └───────────────┘  │
                          │  ┌───────────────┐  │
                          │  │ MCP Server    │  │
                          │  │ (FastMCP)     │  │
                          │  └───────────────┘  │
                          └─────────────────────┘
```

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
     .venv\Scripts\activate
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
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_MODEL=gemini-2.0-flash-exp
   ```

   **Note:** Get your free Gemini API key from https://aistudio.google.com/app/apikey

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
- `GEMINI_API_KEY`: Gemini API key for AI agent functionality (get free at https://aistudio.google.com/app/apikey)
- `GEMINI_MODEL`: Gemini model to use (default: gemini-2.0-flash-exp, options: gemini-2.0-flash-exp, gemini-2.5-flash-lite, gemini-2.5-pro)
- `SQL_ECHO`: Set to "true" to enable SQL query logging (optional)

### Frontend (.env.local)
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`: Clerk publishable key for frontend
- `NEXT_PUBLIC_API_URL`: URL of the backend API (e.g., http://localhost:8000)

## API Endpoints

### Authentication
- `GET /api/v1/auth/me` - Get current user info

### Tasks
- `GET /api/v1/tasks` - Get all tasks for authenticated user (with filtering, sorting, and search capabilities)
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

### Chat (AI-Powered)
- `POST /api/v1/chat/message` - Send a message to the AI chatbot
- `POST /api/v1/chat/message/stream` - Send a message and receive streamed response (SSE)
- `GET /api/v1/chat/history` - Get chat history for a session
- `DELETE /api/v1/chat/history` - Clear chat history

### MCP Server (Internal)
The application includes an MCP server using the official MCP Python SDK with FastMCP:

**Tools Available:**
- `create_task` - Create a new task
- `update_task` - Update an existing task
- `toggle_task_completion` - Toggle task completion status
- `delete_task` - Delete a task
- `search_tasks` - Search for tasks
- `list_today_tasks` - List tasks due today
- `get_task` - Get a specific task

**Resources:**
- `tasks://pending` - Get pending tasks as JSON
- `tasks://summary` - Get task summary statistics

**Prompts:**
- `task_review` - Generate a task review prompt
- `daily_plan` - Generate a daily planning prompt

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
- Rate limiting on API endpoints (GET: 100/min, POST: 20/min, PUT/PATCH: 30/min, DELETE: 30/min)
- Input validation using Pydantic models
- SQL injection prevention with SQLModel
- Proper error handling without exposing sensitive information
- Database indexing for performance and security
- Cache headers configured to prevent sensitive data caching

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License.