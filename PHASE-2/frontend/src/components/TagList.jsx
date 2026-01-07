"use client";

import { useState, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';

const TagList = () => {
  const [tags, setTags] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newTagName, setNewTagName] = useState('');
  const [editingTagId, setEditingTagId] = useState(null);
  const [editingTagName, setEditingTagName] = useState('');
  const { getToken } = useAuth();

  useEffect(() => {
    fetchTags();
  }, []);

  const fetchTags = async () => {
    try {
      setLoading(true);
      setError(null);
      const token = await getToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/tags`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch tags: ${response.status}`);
      }

      const tagsData = await response.json();
      setTags(tagsData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const createTag = async (e) => {
    e.preventDefault();
    if (!newTagName.trim()) return;

    try {
      setLoading(true);
      setError(null);
      const token = await getToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/tags`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newTagName.trim(),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to create tag: ${response.status}`);
      }

      const createdTag = await response.json();
      setTags([...tags, createdTag]);
      setNewTagName('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const updateTag = async (tagId) => {
    try {
      setLoading(true);
      setError(null);
      const token = await getToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/tags/${tagId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: editingTagName.trim(),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to update tag: ${response.status}`);
      }

      const updatedTag = await response.json();
      setTags(tags.map(tag => tag.id === tagId ? updatedTag : tag));
      setEditingTagId(null);
      setEditingTagName('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const deleteTag = async (tagId) => {
    if (!window.confirm('Are you sure you want to delete this tag?')) {
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const token = await getToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/tags/${tagId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to delete tag: ${response.status}`);
      }

      setTags(tags.filter(tag => tag.id !== tagId));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading && tags.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="text-sm text-slate-500">Loading tags...</div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div>
        <h3 className="text-lg font-semibold text-slate-900">Manage Tags</h3>
        <p className="mt-1 text-sm text-slate-500">Create tags once and reuse them across tasks.</p>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-900">
          Error: {error}
        </div>
      )}

      <form onSubmit={createTag} className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-4">
        <h4 className="font-medium text-slate-900">Create new tag</h4>
        <div className="mt-3 flex gap-2">
          <input
            type="text"
            value={newTagName}
            onChange={(e) => setNewTagName(e.target.value)}
            placeholder="Tag name"
            className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            Create
          </button>
        </div>
      </form>

      {/* Tags list */}
      <div className="mt-6">
        <h4 className="font-medium text-slate-900">Your tags</h4>
        {tags.length === 0 ? (
          <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
            No tags created yet.
          </div>
        ) : (
          <ul className="mt-3 space-y-2">
            {tags.map(tag => (
              <li key={tag.id} className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-3">
                {editingTagId === tag.id ? (
                  <div className="flex-1 flex gap-2">
                    <input
                      type="text"
                      value={editingTagName}
                      onChange={(e) => setEditingTagName(e.target.value)}
                      className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                      autoFocus
                    />
                    <button
                      onClick={() => updateTag(tag.id)}
                      disabled={loading}
                      className="rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditingTagId(null)}
                      className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center">
                    <span className="mr-2 font-medium text-slate-900">{tag.name}</span>
                    <span className="text-xs text-slate-500">ID: {tag.id}</span>
                  </div>
                )}
                <div className="flex gap-1">
                  {editingTagId !== tag.id && (
                    <button
                      onClick={() => {
                        setEditingTagId(tag.id);
                        setEditingTagName(tag.name);
                      }}
                      disabled={loading}
                      className="text-sm font-medium text-blue-700 hover:text-blue-800"
                    >
                      Edit
                    </button>
                  )}
                  <button
                    onClick={() => deleteTag(tag.id)}
                    disabled={loading}
                    className="text-sm font-medium text-red-600 hover:text-red-700"
                  >
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default TagList;