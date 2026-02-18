/**
 * Task Types
 *
 * Type definitions for the Todo application task management features.
 * Includes Task entity, Priority enum, RecurrenceRule enum, and TaskFilters interface.
 */

export type Priority = "HIGH" | "MEDIUM" | "LOW";

export type RecurrenceRule = "DAILY" | "WEEKLY" | "MONTHLY";

/**
 * Tag entity
 */
export interface Tag {
  id: number;
  name: string;
  color: string | null;
  priority: number;
  user_id: number;
  created_at: string;
}

/**
 * Task entity
 */
export interface Task {
  id: number;
  title: string;
  description?: string;
  completed: boolean;
  priority?: Priority;
  due_date?: string;  // ISO-8601
  recurrence_rule?: RecurrenceRule;
  reminder_time?: string;  // ISO-8601 (NEW)
  created_at: string;  // ISO-8601
  updated_at: string;  // ISO-8601
  tags: Tag[];
}

/**
 * Task creation request
 */
export interface TaskCreateRequest {
  title: string;
  description?: string;
  priority?: Priority;
  due_date?: string;  // ISO-8601
  recurrence_rule?: RecurrenceRule;
  reminder_time?: string;  // ISO-8601 (NEW)
  tag_ids?: number[];
}

/**
 * Task update request
 */
export interface TaskUpdateRequest {
  title?: string;
  description?: string;
  completed?: boolean;
  priority?: Priority;
  due_date?: string;
  recurrence_rule?: RecurrenceRule;
  reminder_time?: string;  // (NEW)
  tag_ids?: number[];
}

/**
 * Task filters for querying tasks
 */
export interface TaskFilters {
  completed?: boolean;
  priority?: Priority;
  tags?: number[];
  search?: string;
  sort_by?: "created_at" | "updated_at" | "due_date" | "priority";
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
}

/**
 * Priority color mapping for UI
 */
export const PRIORITY_COLORS: Record<Priority, { bg: string; text: string; border: string }> = {
  HIGH: {
    bg: "bg-red-100",
    text: "text-red-800",
    border: "border-red-200"
  },
  MEDIUM: {
    bg: "bg-yellow-100",
    text: "text-yellow-800",
    border: "border-yellow-200"
  },
  LOW: {
    bg: "bg-green-100",
    text: "text-green-800",
    border: "border-green-200"
  }
} as const;

/**
 * Task status helpers
 */
export const TASK_STATUS = {
  isOverdue: (task: Task): boolean => {
    if (!task.due_date || task.completed) return false;
    return new Date(task.due_date) < new Date();
  },
  isDueToday: (task: Task): boolean => {
    if (!task.due_date || task.completed) return false;
    const dueDate = new Date(task.due_date);
    const today = new Date();
    return (
      dueDate.getDate() === today.getDate() &&
      dueDate.getMonth() === today.getMonth() &&
      dueDate.getFullYear() === today.getFullYear()
    );
  },
  hasPriority: (task: Task): boolean => !!task.priority && task.priority !== "LOW",
} as const;
