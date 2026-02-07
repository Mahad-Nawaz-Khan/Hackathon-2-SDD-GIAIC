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

class ChatService {
  private baseUrl: string;
  private sessionId: string;

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

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Send a message to the chatbot and get a response
   */
  async sendMessage(content: string): Promise<ChatResponse> {
    try {
      // Get the auth token from Clerk
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
      console.error('Error sending message:', error);
      throw error;
    }
  }

  /**
   * Send a message and receive a streamed response using Server-Sent Events
   *
   * @param content - The message content to send
   * @param onContent - Callback for each content delta (streaming text)
   * @param onToolCall - Callback when a tool is called
   * @param onToolOutput - Callback when a tool returns output
   * @param onDone - Callback when the stream is complete with final response
   * @param onError - Callback when an error occurs
   * @returns AbortController to cancel the stream if needed
   */
  sendMessageStream(
    content: string,
    callbacks: {
      onContent?: (delta: string) => void;
      onToolCall?: (tool: string, args: any) => void;
      onToolOutput?: (output: any) => void;
      onDone?: (response: ChatResponse) => void;
      onError?: (error: string) => void;
    }
  ): AbortController {
    const controller = new AbortController();
    const signal = controller.signal;

    this.getAuthToken().then((token) => {
      // Construct URL with session_id as query param for SSE
      const url = new URL(`${this.baseUrl}/api/v1/chat/message/stream`);
      url.searchParams.set('content', content);
      url.searchParams.set('session_id', this.sessionId);

      fetch(url.toString(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        signal,
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('No response body');
          }

          const decoder = new TextDecoder();
          let buffer = '';
          let fullContent = '';

          const readStream = (): Promise<void> =>
            reader.read().then(({ done, value }) => {
              if (done) {
                return Promise.resolve();
              }

              buffer += decoder.decode(value, { stream: true });

              // Process complete SSE events
              const lines = buffer.split('\n');
              buffer = lines.pop() || ''; // Keep incomplete line in buffer

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  const data = line.slice(6);
                  try {
                    const event = JSON.parse(data);

                    switch (event.type) {
                      case 'content':
                      case 'content_delta':
                        fullContent += event.content;
                        callbacks.onContent?.(event.content);
                        break;
                      case 'tool_call':
                        callbacks.onToolCall?.(event.tool, event.args);
                        break;
                      case 'tool_output':
                        callbacks.onToolOutput?.(event.output);
                        break;
                      case 'done':
                        callbacks.onDone?.({
                          message: {
                            id: event.message_id || '',
                            content: event.content || fullContent,
                            sender_type: 'AI' as const,
                            created_at: new Date().toISOString(),
                          },
                          operation_performed: event.operation_performed,
                          model_used: event.model_used,
                        });
                        break;
                      case 'error':
                        callbacks.onError?.(event.error || 'Unknown error');
                        break;
                    }
                  } catch (e) {
                    // Ignore invalid JSON
                  }
                }
              }

              return readStream();
            });

          return readStream();
        })
        .catch((error) => {
          if (error.name !== 'AbortError') {
            callbacks.onError?.(error.message);
          }
        });
    }).catch((error) => {
      callbacks.onError?.(error.message);
    });

    return controller;
  }

  /**
   * Get chat history for the current session
   */
  async getHistory(limit: number = 50): Promise<ChatHistoryResponse> {
    try {
      const token = await this.getAuthToken();

      const response = await fetch(
        `${this.baseUrl}/api/v1/chat/history?limit=${limit}&session_id=${this.sessionId}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Convert backend messages to frontend format
      const messages = data.messages.map((msg: any) => ({
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
      console.error('Error getting chat history:', error);
      throw error;
    }
  }

  /**
   * Clear chat history for the current session
   */
  async clearHistory(): Promise<void> {
    try {
      const token = await this.getAuthToken();

      const response = await fetch(
        `${this.baseUrl}/api/v1/chat/history?session_id=${this.sessionId}`,
        {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      // Generate new session ID after clearing
      this.sessionId = this.generateSessionId();
      if (typeof window !== 'undefined') {
        localStorage.setItem('chat_session_id', this.sessionId);
      }
    } catch (error) {
      console.error('Error clearing chat history:', error);
      throw error;
    }
  }

  /**
   * Get the current session ID
   */
  getSessionId(): string {
    return this.sessionId;
  }

  /**
   * Set a new session ID
   */
  setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
    if (typeof window !== 'undefined') {
      localStorage.setItem('chat_session_id', this.sessionId);
    }
  }

  /**
   * Get the auth token from Clerk
   */
  private async getAuthToken(): Promise<string> {
    // If we have access to Clerk, get the token
    if (typeof window !== 'undefined' && (window as any).clerk) {
      try {
        const token = await (window as any).clerk.session.getToken();
        if (token) {
          return token;
        }
      } catch (error) {
        console.error('Error getting Clerk token:', error);
      }
    }

    // Fallback: try to get from localStorage
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('__clerk_client_jwt');
      if (token) {
        return token;
      }
    }

    throw new Error('No authentication token available');
  }

  /**
   * Format AI response text for display
   */
  static formatResponse(text: string): string {
    // Convert newlines to line breaks
    return text.replace(/\n/g, '<br>');
  }

  /**
   * Extract operation type from chat response
   */
  static getOperationType(response: ChatResponse): string | null {
    return response.operation_performed?.type || null;
  }

  /**
   * Check if response indicates a successful operation
   */
  static isOperationSuccessful(response: ChatResponse): boolean {
    if (!response.operation_performed) {
      return false;
    }

    const { type, result } = response.operation_performed;

    // Check if result indicates success
    if (result && typeof result === 'object' && 'success' in result) {
      return result.success === true;
    }

    // Check operation type
    return [
      'create_task',
      'toggle_task',
      'delete_task',
      'search_tasks',
      'list_tasks',
      'list_today_tasks',
    ].includes(type);
  }
}

// Create singleton instance
const chatService = new ChatService();

export default chatService;