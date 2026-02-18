/**
 * SortControls Component
 *
 * Provides sort by dropdown and order toggle for task list.
 * Allows users to sort tasks by created_at, updated_at, due_date, or priority.
 */

'use client';

import React from 'react';
import { TaskFilters } from '../types/task';

interface SortControlsProps {
  filters: TaskFilters;
  onFiltersChange: (filters: TaskFilters) => void;
}

export const SortControls: React.FC<SortControlsProps> = ({ filters, onFiltersChange }) => {
  const handleSortByChange = (value: string) => {
    onFiltersChange({
      ...filters,
      sort_by: value as TaskFilters['sort_by']
    });
  };

  const handleOrderToggle = () => {
    onFiltersChange({
      ...filters,
      order: filters.order === 'asc' ? 'desc' : 'asc'
    });
  };

  return (
    <div className="flex items-center gap-2">
      {/* Sort By */}
      <div className="relative">
        <label htmlFor="sort-by" className="sr-only">
          Sort by
        </label>
        <select
          id="sort-by"
          value={filters.sort_by || 'created_at'}
          onChange={(e) => handleSortByChange(e.target.value)}
          className="px-3 py-2 pr-8 text-sm border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 appearance-none bg-white"
        >
          <option value="created_at">Date Created</option>
          <option value="updated_at">Date Updated</option>
          <option value="due_date">Due Date</option>
          <option value="priority">Priority</option>
        </select>
        {/* Custom dropdown arrow */}
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-500">
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Order Toggle */}
      <button
        type="button"
        onClick={handleOrderToggle}
        className="inline-flex items-center px-3 py-2 text-sm border border-gray-300 rounded-md shadow-sm bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200"
        title={`Currently: ${filters.order === 'asc' ? 'Ascending' : 'Descending'}`}
      >
        {filters.order === 'asc' ? (
          <svg className="h-4 w-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h6m4 4l4-4m0 0l-4 4m4-4l4 4" />
          </svg>
        ) : (
          <svg className="h-4 w-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4" />
          </svg>
        )}
        <span className="ml-1 text-xs text-gray-500">
          {filters.order === 'asc' ? 'Asc' : 'Desc'}
        </span>
      </button>
    </div>
  );
};

export default SortControls;
