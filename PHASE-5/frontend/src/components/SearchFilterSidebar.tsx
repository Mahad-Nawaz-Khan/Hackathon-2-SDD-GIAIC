/**
 * SearchFilterSidebar Component
 *
 * Provides search, filter by priority, and filter by tags functionality.
 * Allows users to quickly find and filter their tasks.
 */

'use client';

import React, { useState } from 'react';
import { Priority, TaskFilters, Tag } from '../types/task';

interface SearchFilterSidebarProps {
  filters: TaskFilters;
  onFiltersChange: (filters: TaskFilters) => void;
  tags: Tag[];
  onClearFilters: () => void;
}

export const SearchFilterSidebar: React.FC<SearchFilterSidebarProps> = ({
  filters,
  onFiltersChange,
  tags,
  onClearFilters
}) => {
  const [searchInput, setSearchInput] = useState(filters.search || '');
  const [selectedTags, setSelectedTags] = useState<number[]>(filters.tags || []);

  const handleSearchChange = (value: string) => {
    setSearchInput(value);
    onFiltersChange({ ...filters, search: value || undefined });
  };

  const handlePriorityChange = (value: string) => {
    const priority = value as Priority | undefined;
    onFiltersChange({ ...filters, priority });
  };

  const handleTagToggle = (tagId: number) => {
    const newSelectedTags = selectedTags.includes(tagId)
      ? selectedTags.filter(id => id !== tagId)
      : [...selectedTags, tagId];

    setSelectedTags(newSelectedTags);
    onFiltersChange({ ...filters, tags: newSelectedTags.length > 0 ? newSelectedTags : undefined });
  };

  const hasActiveFilters = filters.search || filters.priority || (filters.tags && filters.tags.length > 0);

  return (
    <div className="w-full bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      {/* Search */}
      <div className="mb-4">
        <label htmlFor="task-search" className="block text-sm font-medium text-gray-700 mb-1">
          Search
        </label>
        <input
          id="task-search"
          type="text"
          placeholder="Search tasks..."
          value={searchInput}
          onChange={(e) => handleSearchChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
        />
      </div>

      {/* Priority Filter */}
      <div className="mb-4">
        <label htmlFor="filter-priority" className="block text-sm font-medium text-gray-700 mb-1">
          Priority
        </label>
        <select
          id="filter-priority"
          value={filters.priority || ''}
          onChange={(e) => handlePriorityChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
        >
          <option value="">All Priorities</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
        </select>
      </div>

      {/* Tags Filter */}
      {tags.length > 0 && (
        <div className="mb-4">
          <span className="block text-sm font-medium text-gray-700 mb-1">
            Tags
          </span>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => {
              const isSelected = selectedTags.includes(tag.id);
              return (
                <button
                  key={tag.id}
                  type="button"
                  onClick={() => handleTagToggle(tag.id)}
                  className={`
                    inline-flex items-center px-3 py-1 rounded-full text-xs font-medium
                    transition-colors duration-200
                    ${isSelected
                      ? 'bg-blue-100 text-blue-800 border border-blue-300'
                      : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                    }
                  `}
                  style={tag.color && !isSelected ? { backgroundColor: tag.color + '20', color: tag.color } : {}}
                >
                  {tag.name}
                  {isSelected && (
                    <span className="ml-1">×</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Clear Filters */}
      {hasActiveFilters && (
        <button
          type="button"
          onClick={() => {
            setSearchInput('');
            setSelectedTags([]);
            onClearFilters();
          }}
          className="w-full px-3 py-2 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors duration-200"
        >
          Clear All Filters
        </button>
      )}
    </div>
  );
};

export default SearchFilterSidebar;
