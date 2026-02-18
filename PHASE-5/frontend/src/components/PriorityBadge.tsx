/**
 * PriorityBadge Component
 *
 * Displays a task's priority as a color-coded badge.
 * Colors:
 * - HIGH: Red
 * - MEDIUM: Yellow
 * - LOW: Green
 */

import React from 'react';
import { Priority, PRIORITY_COLORS } from '../types/task';

interface PriorityBadgeProps {
  priority?: Priority;
  className?: string;
}

export const PriorityBadge: React.FC<PriorityBadgeProps> = ({
  priority,
  className = ''
}) => {
  if (!priority) {
    return null;
  }

  const colors = PRIORITY_COLORS[priority];

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors.bg} ${colors.text} ${colors.border} border ${className}`}
      title={`Priority: ${priority}`}
    >
      {priority.charAt(0) + priority.slice(1).toLowerCase()}
    </span>
  );
};

/**
 * Small priority indicator (dot variant)
 */
export const PriorityIndicator: React.FC<{ priority?: Priority }> = ({ priority }) => {
  if (!priority) {
    return null;
  }

  const colors = {
    HIGH: 'bg-red-500',
    MEDIUM: 'bg-yellow-500',
    LOW: 'bg-green-500'
  };

  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${colors[priority]}`}
      title={`Priority: ${priority}`}
    />
  );
};

export default PriorityBadge;
