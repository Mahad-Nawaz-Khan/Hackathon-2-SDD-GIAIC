# Quickstart Guide: AI Chatbot System

## Overview
This guide provides the essential steps to get the AI Chatbot system up and running with the existing TODO application.

## Prerequisites
- Python 3.11+ installed
- Node.js 18+ installed
- Access to OpenAI API (for OpenAI Agents SDK)
- Access to Google Generative AI API (for Gemini model)
- PostgreSQL database (already set up for TODO app)
- Completed TODO application setup

## Environment Configuration
Create a `.env` file in the backend directory with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google Generative AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Database Configuration (should already exist from TODO app)
DATABASE_URL=postgresql://username:password@localhost:5432/todo_db

# Clerk Configuration (should already exist from TODO app)
CLERK_SECRET_KEY=your_clerk_secret_key
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key

# MCP Server Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8001
```

## Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   # Make sure to add OpenAI, google-generativeai, and any MCP-related packages
   pip install openai google-generativeai
   ```

3. Set up the database (migrations should extend existing TODO app structure):
   ```bash
   # Run existing TODO app migrations first
   # Then run chatbot-specific migrations if any
   ```

4. Start the MCP server:
   ```bash
   python -m src.mcp_server.main
   ```

5. Start the main API server:
   ```bash
   uvicorn src.main:app --reload
   ```

## Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install JavaScript dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## Running the Chatbot
1. Ensure both backend and frontend servers are running
2. Open the TODO application in your browser
3. Look for the chatbot interface (typically a floating button or dedicated section)
4. Authenticate with Clerk as usual
5. Start interacting with the chatbot using natural language

## API Usage Example
Send a message to the chatbot:
```bash
curl -X POST http://localhost:8000/api/chat/messages \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a high priority task to finish report by tomorrow"
  }'
```

## Key Components
- **OpenAI Agents SDK**: Handles complex reasoning and multi-step operations
- **Gemini API**: Handles read-only operations and cost-efficient queries
- **MCP Server**: Mediates all data operations ensuring security and authorization
- **Chat Service**: Orchestrates the interaction between AI models and MCP server
- **Frontend Component**: Provides the user interface for chat interactions

## Troubleshooting
- If chatbot responses are delayed, check your API key quotas for OpenAI and Google
- If authentication fails, ensure Clerk tokens are properly configured
- If data operations fail, verify MCP server is running and properly configured
- Check server logs for detailed error messages