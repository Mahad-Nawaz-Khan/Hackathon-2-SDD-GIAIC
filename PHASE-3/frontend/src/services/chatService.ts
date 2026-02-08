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

type TokenGetter = () => Promise<string>;

class ChatService {
  private baseUrl: string;
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
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get the auth token from Clerk
   */
  private async getAuthToken(): Promise<string> {
    console.log('[chatService] Getting auth token...');

    // First try: Use the token getter set by the React component
    if (this.tokenGetter) {
      try {
        console.log('[chatService] Using token getter from Clerk hook...');
        const token = await this.tokenGetter();
        console.log('[chatService] Token from getter:', token ? 'FOUND' : 'NULL');
        if (token) {
          return token;
        }
      } catch (error) {
        console.error('[chatService] Error getting Clerk token from getter:', error);
      }
    } else {
      console.log('[chatService] No token getter available!');
    }

    // Second try: Use the clerk-js loaded via script tag (if available)
    if (typeof window !== 'undefined' && (window as any).Clerk) {
      try {
        console.log('[chatService] Trying clerk-js fallback...');
        const clerk = new (window as any).Clerk(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);
        await clerk.load();
        if (clerk.session) {
          const token = await clerk.session.getToken();
          if (token) {
            console.log('[chatService] Token from clerk-js: FOUND');
            return token;
          }
        }
      } catch (error) {
        console.error('[chatService] Error getting Clerk token from clerk-js:', error);
      }
    }

    // Third try: Check for token in localStorage (backup)
    if (typeof window !== 'undefined') {
      console.log('[chatService] Checking localStorage for token...');
      // Check various Clerk storage keys
      const keys = ['__clerk_client_jwt', '__session'];
      for (const key of keys) {
        const token = localStorage.getItem(key);
        if (token) {
          console.log(`[chatService] Token from localStorage (${key}): FOUND`);
          return token;
        }
      }
    }

    console.error('[chatService] No authentication token found!');
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
    callbacks: {
      onContent: (delta: string) => void;
      onToolCall?: (tool: string, args: any) => void;
      onToolOutput?: (output: any) => void;
      onDone: (response: ChatResponse) => void;
      onError: (error: string) => void;
    }
  ): AbortController {
    const controller = new AbortController();

    this.getAuthToken()
      .then((token) => {
        const url = new URL(`${this.baseUrl}/api/v1/chat/stream`);
        url.searchParams.set('content', content);
        url.searchParams.set('session_id', this.sessionId);

        fetch(url.toString(), {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
          signal: controller.signal,
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
            let fullResponse = '';

            const readStream = (): Promise<void> => {
              return reader.read().then(({ done, value }) => {
                if (done) {
                  // Parse final response
                  try {
                    const finalResponse: ChatResponse = JSON.parse(fullResponse);
                    callbacks.onDone(finalResponse);
                  } catch (e) {
                    // If we can't parse as JSON, create a minimal response
                    callbacks.onDone({
                      message: {
                        id: Date.now().toString(),
                        content: fullResponse || 'Response completed',
                        sender_type: 'AI',
                        created_at: new Date().toISOString(),
                      },
                    });
                  }
                  return Promise.resolve();
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                  if (line.startsWith('data: ')) {
                    const data = line.slice(6);

                    if (data === '[DONE]') {
                      continue;
                    }

                    try {
                      const parsed = JSON.parse(data);

                      if (parsed.type === 'content_delta') {
                        callbacks.onContent(parsed.content || '');
                        fullResponse += parsed.content || '';
                      } else if (parsed.type === 'tool_call') {
                        callbacks.onToolCall?.(parsed.tool, parsed.args);
                      } else if (parsed.type === 'tool_output') {
                        callbacks.onToolOutput?.(parsed.output);
                      } else if (parsed.type === 'final') {
                        fullResponse = parsed.content || fullResponse;
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
                        controller.abort();
                        return Promise.resolve();
                      } else if (parsed.type === 'error') {
                        callbacks.onError(parsed.content || 'Unknown error');
                        controller.abort();
                        return Promise.resolve();
                      }
                    } catch (e) {
                      console.error('Failed to parse SSE data:', data, e);
                    }
                  }
                }

                return readStream();
              });
            };

            return readStream();
          })
          .catch((error) => {
            if (error.name === 'AbortError') {
              return;
            }
            console.error('Stream error:', error);
            callbacks.onError(error.message || 'Stream error');
          });
      })
      .catch((error) => {
        console.error('Token error:', error);
        callbacks.onError(error.message || 'Failed to get authentication token');
      });

    return controller;
  }

  /**
   * Get chat history
   */
  async getHistory(): Promise<ChatHistoryResponse> {
    try {
      const token = await this.getAuthToken();

      const url = new URL(`${this.baseUrl}/api/v1/chat/history`);
      url.searchParams.set('session_id', this.sessionId);
      url.searchParams.set('limit', '50');

      const response = await fetch(url.toString(), {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Transform the response to match our interface
      const messages: ChatMessage[] = data.messages.map((msg: any) => ({
        id: msg.id,
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
      console.error('Failed to get history:', error);
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
      console.error('Failed to clear history:', error);
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
   * Cancel any ongoing request
   */
  cancelRequest(): void {
    // This is handled by the AbortController returned by sendMessageStream
  }
}

// Singleton instance
const chatService = new ChatService();

export default chatService;
