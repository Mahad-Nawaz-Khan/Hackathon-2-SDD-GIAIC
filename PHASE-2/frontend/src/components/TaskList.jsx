"use client";

import { useState, useEffect } from 'react';
import { TaskItem } from './TaskItem';
import { useAuth, useUser } from '@clerk/nextjs';

export const TaskList = () => {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    completed: null,
    priority: '',
    search: ''
  });
  const [sortConfig, setSortConfig] = useState({
    sortBy: 'created_at',
    order: 'desc'
  });
  const { getToken } = useAuth();

  useEffect(() => {
    loadTasks();
  }, [filters, sortConfig]);

  const loadTasks = async () => {
    try {
      setLoading(true);
      await fetchTasksFromAPI();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchTasksFromAPI = async () => {
    try {
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

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/tasks?${params.toString()}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch tasks: ${response.status}`);
      }

      const tasksData = await response.json();
      setTasks(tasksData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTaskUpdate = (updatedTask) => {
    setTasks(tasks.map(task => task.id === updatedTask.id ? updatedTask : task));
  };

  const handleTaskDelete = (deletedTaskId) => {
    setTasks(tasks.filter(task => task.id !== deletedTaskId));
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

  if (loading) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-lg">
        <div className="text-sm text-white/70">Loading tasks...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-5 text-sm text-red-200">
        Error: {error}
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-lg">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Your Tasks</h2>
          <p className="mt-1 text-sm text-white/70">Filter, sort, and search across your tasks.</p>
        </div>
      </div>


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
              value={filters.search}
              onChange={(e) => handleFilterChange('search', e.target.value)}
              className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-blue-400/40"
            />
          </div>
      </div>

      {tasks.length === 0 ? (
        <div className="mt-6 rounded-xl border border-white/10 bg-black/20 p-6">
          <div className="font-medium text-white">No tasks found</div>
          <div className="mt-1 text-sm text-white/70">Create your first task using the form on the right.</div>
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