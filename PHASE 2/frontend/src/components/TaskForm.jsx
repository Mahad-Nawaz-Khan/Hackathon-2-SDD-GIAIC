"use client";

import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import TagSelector from './TagSelector';

export const TaskForm = ({ onTaskCreated }) => {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [priority, setPriority] = useState('MEDIUM');
  const [dueDate, setDueDate] = useState('');
  const [recurrenceRule, setRecurrenceRule] = useState('');
  const [selectedTags, setSelectedTags] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { getToken } = useAuth();
  

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!title.trim()) {
      setError('Title is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const token = await getToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/tasks`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: title.trim(),
          description: description.trim(),
          priority,
          due_date: dueDate || null,
          recurrence_rule: recurrenceRule || null,
          tag_ids: selectedTags
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to create task: ${response.status}`);
      }

      const newTask = await response.json();

      // Reset form
      setTitle('');
      setDescription('');
      setPriority('MEDIUM');
      setDueDate('');
      setRecurrenceRule('');
      setSelectedTags([]);

      // Notify parent component
      if (onTaskCreated) {
        onTaskCreated(newTask);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-lg">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-white">Create New Task</h3>
          <p className="mt-1 text-sm text-white/70">Add a task with priority, due date, and optional tags.</p>
        </div>
      </div>
      {error && (
        <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          Error: {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="mt-5 space-y-4">
        <div>
          <label htmlFor="title" className="block text-sm font-medium text-white/80">Title *</label>
          <input
            type="text"
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-blue-400/40"
            required
          />
        </div>

        <div>
          <label htmlFor="description" className="block text-sm font-medium text-white/80">Description</label>
          <textarea
            id="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-blue-400/40"
            rows="3"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="priority" className="block text-sm font-medium text-white/80">Priority</label>
            <select
              id="priority"
              value={priority}
              onChange={(e) => setPriority(e.target.value)}
              className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-400/40"
            >
              <option value="LOW">Low</option>
              <option value="MEDIUM">Medium</option>
              <option value="HIGH">High</option>
            </select>
          </div>

          <div>
            <label htmlFor="dueDate" className="block text-sm font-medium text-white/80">Due Date</label>
            <input
              type="date"
              id="dueDate"
              value={dueDate}
              onChange={(e) => setDueDate(e.target.value)}
              className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-400/40"
            />
          </div>
        </div>

        <div>
          <label htmlFor="recurrenceRule" className="block text-sm font-medium text-white/80">Recurrence</label>
          <select
            id="recurrenceRule"
            value={recurrenceRule}
            onChange={(e) => setRecurrenceRule(e.target.value)}
            className="mt-1 w-full rounded-lg border border-white/10 bg-white/10 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-400/40"
          >
            <option value="">No recurrence</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
            <option value="yearly">Yearly</option>
            <option value="every 3 days">Every 3 days</option>
            <option value="every 7 days">Every week</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-white/80">Tags</label>
          <TagSelector
            selectedTags={selectedTags}
            onTagsChange={setSelectedTags}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-blue-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Creating...' : 'Create Task'}
        </button>
      </form>
    </div>
  );
};