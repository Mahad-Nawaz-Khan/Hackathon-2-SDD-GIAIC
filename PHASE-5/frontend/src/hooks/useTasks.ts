/**
 * useTasks Hook
 *
 * Custom hook for fetching and managing tasks with filtering and sorting.
 * Provides loading state, error state, and refetch functionality.
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { Task, TaskFilters } from '../types/task';

interface UseTasksResult {
  tasks: Task[];
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

const DEFAULT_FILTERS: TaskFilters = {
  sort_by: 'created_at',
  order: 'desc',
  limit: 10,
  offset: 0
};

/**
 * Fetch tasks from the API with optional filters
 */
async function fetchTasks(filters: TaskFilters): Promise<Task[]> {
  const params = new URLSearchParams();

  if (filters.completed !== undefined) {
    params.set('completed', String(filters.completed));
  }
  if (filters.priority) {
    params.set('priority', filters.priority);
  }
  if (filters.tags && filters.tags.length > 0) {
    params.set('tags', filters.tags.join(','));
  }
  if (filters.search) {
    params.set('search', filters.search);
  }
  if (filters.sort_by) {
    params.set('sort_by', filters.sort_by);
  }
  if (filters.order) {
    params.set('order', filters.order);
  }
  if (filters.limit !== undefined) {
    params.set('limit', String(filters.limit));
  }
  if (filters.offset !== undefined) {
    params.set('offset', String(filters.offset));
  }

  const queryString = params.toString();
  const queryPart = queryString ? `?${queryString}` : '';
  const url = `/api/v1/tasks${queryPart}`;

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch tasks: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Custom hook for managing tasks with filters
 */
export function useTasks(filters: TaskFilters = {}): UseTasksResult {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Merge user filters with defaults
  const mergedFilters = { ...DEFAULT_FILTERS, ...filters };

  const loadTasks = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await fetchTasks(mergedFilters);
      setTasks(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error fetching tasks:', err);
    } finally {
      setIsLoading(false);
    }
  }, [mergedFilters]);

  useEffect(() => {
    loadTasks();
  }, [loadTasks]);

  return {
    tasks,
    isLoading,
    error,
    refetch: loadTasks
  };
}

/**
 * Hook for creating a new task
 */
export function useCreateTask() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const createTask = useCallback(async (taskData: {
    title: string;
    description?: string;
    priority?: string;
    due_date?: string;
    recurrence_rule?: string;
    reminder_time?: string;
    tag_ids?: number[];
  }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/tasks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskData),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || 'Failed to create task');
      }

      return await response.json();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { createTask, isLoading, error };
}

/**
 * Hook for updating a task
 */
export function useUpdateTask() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const updateTask = useCallback(async (taskId: number, updates: {
    title?: string;
    description?: string;
    completed?: boolean;
    priority?: string;
    due_date?: string;
    recurrence_rule?: string;
    reminder_time?: string;
    tag_ids?: number[];
  }) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/v1/tasks/${taskId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || 'Failed to update task');
      }

      return await response.json();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { updateTask, isLoading, error };
}

/**
 * Hook for toggling task completion
 */
export function useToggleTaskCompletion() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const toggleCompletion = useCallback(async (taskId: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/v1/tasks/${taskId}/toggle-completion`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to toggle task completion');
      }

      return await response.json();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { toggleCompletion, isLoading, error };
}

/**
 * Hook for deleting a task
 */
export function useDeleteTask() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const deleteTask = useCallback(async (taskId: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/v1/tasks/${taskId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete task');
      }

      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { deleteTask, isLoading, error };
}
