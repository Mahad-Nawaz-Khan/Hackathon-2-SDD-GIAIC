"use client";

import { useState, useEffect, useRef } from 'react';
import { TaskItem } from './TaskItem';
import { useAuth } from '@clerk/nextjs';

export const TaskList = ({ createdTask }) => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [filters, setFilters] = useState({
    completed: null,
    priority: '',
    search: ''
  });
  const [searchInput, setSearchInput] = useState('');
  const [sortConfig, setSortConfig] = useState({
    sortBy: 'created_at',
    order: 'desc'
  });
  const { getToken } = useAuth();
  const abortControllerRef = useRef(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    fetchTasksFromAPI();
  }, [filters, sortConfig]);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    const normalizedSearch = searchInput.trim();
    const timeout = setTimeout(() => {
      setFilters((prev) => {
        if (prev.search === normalizedSearch) {
          return prev;
        }
        return {
          ...prev,
          search: normalizedSearch,
        };
      });
    }, 300);

    return () => clearTimeout(timeout);
  }, [searchInput]);

  useEffect(() => {
    if (!createdTask || !createdTask.id) {
      return;
    }

    setTasks((prev) => {
      if (prev.some((task) => task.id === createdTask.id)) {
        return prev;
      }

      if (filters.completed !== null && createdTask.completed !== filters.completed) {
        return prev;
      }
      if (filters.priority && createdTask.priority !== filters.priority) {
        return prev;
      }
      if (filters.search) {
        const haystack = `${createdTask.title ?? ''} ${createdTask.description ?? ''}`.toLowerCase();
        if (!haystack.includes(filters.search.toLowerCase())) {
          return prev;
        }
      }

      return [createdTask, ...prev];
    });
  }, [createdTask, filters.completed, filters.priority, filters.search]);

  const fetchTasksFromAPI = async () => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    try {
      setError(null);
      setLoading(true);
      const token = await getToken();

      // Build query parameters
      const params = new URLSearchParams();
      if (filters.completed !== null) {
        params.append('completed', filters.completed.toString());
      }
      if (filters.priority) {
        params.append('priority', filters.priority);
      }
      if (filters.search) {
        params.append('search', filters.search);
      }
      params.append('sort_by', sortConfig.sortBy);
      params.append('order', sortConfig.order);

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/tasks?${params.toString()}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch tasks: ${response.status}`);
      }

      const tasksData = await response.json();
      if (requestIdRef.current !== requestId) {
        return;
      }
      setTasks(tasksData);
    } catch (err) {
      if (requestIdRef.current !== requestId) {
        return;
      }
      if (err?.name === 'AbortError') {
        return;
      }
      setError(err?.message || 'Failed to fetch tasks');
    } finally {
      if (requestIdRef.current === requestId) {
        setLoading(false);
      }
    }
  };

  const handleTaskUpdate = (updatedTask) => {
    setTasks((prev) => prev.map(task => task.id === updatedTask.id ? updatedTask : task));
  };

  const handleTaskDelete = (deletedTaskId) => {
    setTasks((prev) => prev.filter(task => task.id !== deletedTaskId));
  };

  const handleFilterChange = (filterName, value) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: value
    }));
  };

  const handleSortChange = (sortBy) => {
    setSortConfig(prev => ({
      sortBy,
      order: prev.sortBy === sortBy && prev.order === 'asc' ? 'desc' : 'asc'
    }));
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-lg">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Your Tasks</h2>
          <p className="mt-1 text-sm text-white/70">Filter, sort, and search across your tasks.</p>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          Error: {error}
        </div>
      )}

      {loading && (
        <div className="mt-4 text-sm text-white/70">Loading tasks...</div>
      )}

      <div className="mt-5 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium text-white/80">Status</label>
          <select
            value={filters.completed === null ? '' : filters.completed.toString()}
            onChange={(e) => handleFilterChange('completed', e.target.value === '' ? null : e.target.value === 'true')}
            className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-400/40"
          >
            <option value="">All</option>
            <option value="false">Active</option>
            <option value="true">Completed</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-white/80">Priority</label>
          <select
            value={filters.priority}
            onChange={(e) => handleFilterChange('priority', e.target.value)}
            className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-400/40"
          >
            <option value="">All</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-white/80">Sort By</label>
          <select
            value={sortConfig.sortBy}
            onChange={(e) => handleSortChange(e.target.value)}
            className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-400/40"
          >
            <option value="created_at">Created Date</option>
            <option value="updated_at">Updated Date</option>
            <option value="due_date">Due Date</option>
            <option value="priority">Priority</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-white/80">Search</label>
          <input
            type="text"
            placeholder="Search tasks..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-blue-400/40"
          />
        </div>
      </div>

      {tasks.length === 0 ? (
        <div className="mt-6 rounded-xl border border-white/10 bg-black/20 p-6">
          <div className="font-medium text-white">{loading ? 'Loading tasks...' : 'No tasks found'}</div>
          {!loading && (
            <div className="mt-1 text-sm text-white/70">Create your first task using the form on the right.</div>
          )}
        </div>
      ) : (
        <ul className="mt-6 space-y-4">
          {tasks.map(task => (
            <TaskItem
              key={task.id}
              task={task}
              onUpdate={handleTaskUpdate}
              onDelete={handleTaskDelete}
            />
          ))}
        </ul>
      )}
    </div>
  );
};