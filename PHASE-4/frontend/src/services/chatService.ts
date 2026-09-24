/**
 * Chat Service - Client for AI Chatbot API
 */

export interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

export interface ChatResponse {
  message: {
    id: string;
    content: string;
    sender_type: 'USER' | 'AI';
    intent?: string;
    created_at: string;
  };
  operation_performed?: {
    type: string;
    result?: any;
    count?: number;
    task_id?: number;
  };
  model_used?: string;
}

export interface ChatHistoryResponse {
  messages: ChatMessage[];
  total_count: number;
  session_id: string;
}

type TokenGetter = () => Promise<string | null>;

interface StreamCallbacks {
  onContent: (delta: string) => void;
  onToolCall?: (tool: string, args: any) => void;
  onToolOutput?: (output: any) => void;
  onDone: (response: ChatResponse) => void;
  onError: (error: string) => void;
}

interface SseProcessState {
  fullResponse: string;
  shouldStop: boolean;
}

function parseFinalResponse(fullResponse: string): ChatResponse {
  try {
    return JSON.parse(fullResponse);
  } catch (e) {
    console.warn("Could not parse final stream response as JSON, falling back:", e);
    return {
      message: {
        id: Date.now().toString(),
        content: fullResponse || 'Response completed',
        sender_type: 'AI',
        created_at: new Date().toISOString(),
      },
    };
  }
}

function processSseLine(line: string, callbacks: StreamCallbacks, state: SseProcessState): void {
  if (!line.startsWith('data: ')) return;
  const data = line.slice(6);
  if (data === '[DONE]') return;

  try {
    const parsed = JSON.parse(data);
    switch (parsed.type) {
      case 'content_delta':
        callbacks.onContent(parsed.content || '');
        state.fullResponse += parsed.content || '';
        break;
      case 'tool_call':
        callbacks.onToolCall?.(parsed.tool, parsed.args);
        break;
      case 'tool_output':
        callbacks.onToolOutput?.(parsed.output);
        break;
      case 'final':
        state.fullResponse = parsed.content || state.fullResponse;
        callbacks.onDone({
          message: {
            id: Date.now().toString(),
            content: parsed.content || '',
            sender_type: 'AI',
            created_at: new Date().toISOString(),
          },
          operation_performed: parsed.operation_performed,
          model_used: parsed.model_used,
        });
        state.shouldStop = true;
        break;
      case 'error':
        callbacks.onError(parsed.content || 'Unknown error');
        state.shouldStop = true;
        break;
      default:
        break;
    }
  } catch (e) {
    console.warn("Failed to parse SSE data chunk:", e);
  }
}

async function consumeStream(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  callbacks: StreamCallbacks,
  controller: AbortController
): Promise<void> {
  const decoder = new TextDecoder();
  let buffer = '';
  const state: SseProcessState = { fullResponse: '', shouldStop: false };

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      callbacks.onDone(parseFinalResponse(state.fullResponse));
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      processSseLine(line, callbacks, state);
      if (state.shouldStop) {
        controller.abort();
        return;
      }
    }
  }
}

class ChatService {
  private readonly baseUrl: string;
  private sessionId: string;
  private tokenGetter: TokenGetter | null = null;

  constructor() {
    // Use environment variable or fallback to localhost
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    // Generate or retrieve session ID
    if (typeof window !== 'undefined') {
      this.sessionId = localStorage.getItem('chat_session_id') || this.generateSessionId();
      localStorage.setItem('chat_session_id', this.sessionId);
    } else {
      this.sessionId = 'session_server';
    }
  }

  /**
   * Set the token getter function from Clerk hook
   */
  setTokenGetter(getter: TokenGetter) {
    this.tokenGetter = getter;
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  private async getTokenFromGetter(): Promise<string | null> {
    if (!this.tokenGetter) return null;
    try {
      return (await this.tokenGetter()) || null;
    } catch (error) {
      console.warn("Failed to get token from getter:", error);
      return null;
    }
  }

  private async getTokenFromWindowClerk(): Promise<string | null> {
    if (typeof window === 'undefined' || !(window as any).Clerk) return null;
    try {
      const clerk = new (window as any).Clerk(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);
      await clerk.load();
      if (clerk.session) {
        return (await clerk.session.getToken()) || null;
      }
    } catch (error) {
      console.warn("Failed to get token from window.Clerk:", error);
    }
    return null;
  }

  private getTokenFromStorage(): string | null {
    if (typeof window === 'undefined') return null;
    const keys = ['__clerk_client_jwt', '__session'];
    for (const key of keys) {
      const token = localStorage.getItem(key);
      if (token) return token;
    }
    return null;
  }

  /**
   * Get the auth token from Clerk
   */
  private async getAuthToken(): Promise<string> {
    const fromGetter = await this.getTokenFromGetter();
    if (fromGetter) return fromGetter;

    const fromClerk = await this.getTokenFromWindowClerk();
    if (fromClerk) return fromClerk;

    const fromStorage = this.getTokenFromStorage();
    if (fromStorage) return fromStorage;

    throw new Error('No authentication token available. Please sign in.');
  }

  /**
   * Send a message to the chatbot and get a response
   */
  async sendMessage(content: string): Promise<ChatResponse> {
    try {
      const token = await this.getAuthToken();

      const response = await fetch(`${this.baseUrl}/api/v1/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          content,
          session_id: this.sessionId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Failed to send message');
    }
  }

  /**
   * Send a message with streaming response
   */
  sendMessageStream(
    content: string,
    callbacks: StreamCallbacks
  ): AbortController {
    const controller = new AbortController();

    (async () => {
      try {
        const token = await this.getAuthToken();
        const url = new URL(`${this.baseUrl}/api/v1/chat/stream`);
        url.searchParams.set('content', content);
        url.searchParams.set('session_id', this.sessionId);

        const response = await fetch(url.toString(), {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }

        await consumeStream(reader, callbacks, controller);
      } catch (error: any) {
        if (error.name === 'AbortError') {
          return;
        }
        callbacks.onError(error.message || 'Stream error');
      }
    })();

    return controller;
  }

  /**
   * Get chat history
   */
  async getHistory(): Promise<ChatHistoryResponse> {
    try {
      const token = await this.getAuthToken();

      const response = await fetch(
        `${this.baseUrl}/api/v1/chat/history?session_id=${this.sessionId}&limit=100`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch history: ${response.status}`);
      }

      const data = await response.json();

      const messages: ChatMessage[] = (data.messages || []).map((msg: any) => ({
        id: msg.id.toString(),
        text: msg.content,
        sender: msg.sender_type === 'USER' ? 'user' : 'ai',
        timestamp: new Date(msg.created_at),
      }));

      return {
        messages,
        total_count: data.total_count,
        session_id: data.session_id,
      };
    } catch (error) {
      console.warn("Failed to get chat history from server, returning empty:", error);
      // Return empty history on error
      return {
        messages: [],
        total_count: 0,
        session_id: this.sessionId,
      };
    }
  }

  /**
   * Clear chat history
   */
  async clearHistory(): Promise<void> {
    try {
      const token = await this.getAuthToken();

      await fetch(`${this.baseUrl}/api/v1/chat/history`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      // Generate new session ID
      this.sessionId = this.generateSessionId();
      if (typeof window !== 'undefined') {
        localStorage.setItem('chat_session_id', this.sessionId);
      }
    } catch (error) {
      console.warn("Failed to clear chat history:", error);
      throw error;
    }
  }

  /**
   * Get current session ID
   */
  getSessionId(): string {
    return this.sessionId;
  }

  /**
   * Save welcome message to database (so it's included in conversation history)
   */
  async saveWelcomeMessage(content: string): Promise<void> {
    const token = await this.getAuthToken();

    await fetch(`${this.baseUrl}/api/v1/chat/message/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        content,
        session_id: this.sessionId,
        is_welcome: true,
      }),
    });
  }

  /**
   * Cancel any ongoing request
   */
  cancelRequest(): void {
    // This is handled by the AbortController returned by sendMessageStream
  }

  /**
   * Format message response with HTML line breaks
   */
  formatResponse(response: string): string {
    return response.replace(/\n/g, '<br>');
  }

  /**
   * Extract operation type from response
   */
  getOperationType(response: ChatResponse): string | undefined {
    return response.operation_performed?.type;
  }

  /**
   * Check if operation was performed successfully
   */
  isOperationSuccessful(response: ChatResponse): boolean {
    return Boolean(response.operation_performed);
  }
}

// Singleton instance
const chatService = new ChatService();

export default chatService;
