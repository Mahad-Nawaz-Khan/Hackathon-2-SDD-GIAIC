# Next.js & Clerk Skill

## Overview
A reusable Claude skill that provides a comprehensive guide for building Next.js (v14+) applications with Clerk authentication. The guide covers best practices, code patterns, and deployment considerations.

---

## Guide

### 1. Next.js 14+ App Router Best Practices
- Use the **app/** directory with `layout.tsx`, `page.tsx`, and `loading.tsx`.
- Leverage **Route Groups** (`(admin)`) for logical grouping without affecting URLs.
- Prefer **Server Components** for data‑heavy pages; use `export const dynamic = 'force-static'` or `'force-dynamic'` as needed.
- Keep **client‑only UI** (hooks, interactivity) inside `use client` components.
- Utilize **react‑cache** and **`fetch`** with `cache: 'force-no-store'` for fresh data.

### 2. Clerk Auth Hooks & Middleware
- Install Clerk: `npm i @clerk/nextjs @clerk/clerk-sdk-node`
- **Middleware** (in `app/middleware.ts`):
  ```ts
  import { authMiddleware } from '@clerk/nextjs/server'

  export default authMiddleware({
    publicRoutes: ['/sign-in', '/sign-up', '/api/public/*'],
  })

  export const config = { matcher: ['/((?!_next|.*\..*).*)'] }
  ```
- **Auth Hooks**:
  - `useAuth` (client) for UI state.
  - `auth()` (server) for session data inside `server` components or API routes.

### 3. Server vs Client Component Patterns
| Concern | Server Component | Client Component |
|---|---|---|
| Data fetching | `fetch`/`prisma` directly | Use `useEffect` + API route |
| Auth access | `auth()` from Clerk SDK | `useAuth()` hook |
| Interactivity | ❌ (no state) | ✅ (`useState`, event handlers) |
| SEO | ✅ (HTML rendered) | ❓ (hydrate) |

### 4. TypeScript Conventions
- **Strict mode** in `tsconfig.json` (`strict: true`).
- Export **type‑only** imports: `import type { User } from '@clerk/nextjs'`.
- Define **DTOs** for API contracts (`interface CreatePostDto { title: string; content: string }`).
- Use **`zod`** for runtime validation together with TypeScript inference.

### 5. API Routes (App Router)
Create `app/api/…/route.ts`:
```ts
import { auth } from '@clerk/nextjs/server'
import { NextResponse } from 'next/server'
import { z } from 'zod'

export async function POST(req: Request) {
  const { userId } = auth()
  if (!userId) return new NextResponse('Unauthorized', { status: 401 })

  const schema = z.object({ title: z.string(), content: z.string() })
  const body = await req.json()
  const data = schema.parse(body)

  // TODO: persist data (e.g., Prisma)
  return NextResponse.json({ success: true, id: 'new-id' })
}
```
- Use **`export const dynamic = 'force-dynamic'`** if you need fresh data.

### 6. Data Fetching Patterns
- **Server‑Component fetch**:
  ```tsx
  export default async function Posts() {
    const res = await fetch('/api/posts', { cache: 'no-store' })
    const posts = await res.json()
    return <ul>{posts.map(p => <li key={p.id}>{p.title}</li>)}</ul>
  }
  ```
- **Client‑side SWR** with `useSWR` for revalidation.
- Leverage **React Query** or **TanStack Query** for caching and retry logic.

### 7. Form Handling
- Use **React Hook Form** with **Zod** resolver.
- Example client component:
  ```tsx
  'use client'
  import { useForm } from 'react-hook-form'
  import { zodResolver } from '@hookform/resolvers/zod'
  import { z } from 'zod'

  const schema = z.object({ title: z.string().min(1), content: z.string().min(1) })

  export default function PostForm() {
    const { register, handleSubmit, formState: { errors } } = useForm({ resolver: zodResolver(schema) })
    const onSubmit = async (data: any) => {
      const res = await fetch('/api/posts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
      // handle response
    }
    return (
      <form onSubmit={handleSubmit(onSubmit)}>
        <input {...register('title')} />
        {errors.title && <span>{errors.title.message}</span>}
        <textarea {...register('content')} />
        {errors.content && <span>{errors.content.message}</span>}
        <button type='submit'>Create</button>
      </form>
    )
  }
  ```

### 8. Error Boundaries
- Create a **RootErrorBoundary** in `app/error.tsx`:
  ```tsx
  'use client'
  import { ErrorBoundary } from 'react-error-boundary'

  export default function RootErrorBoundary({ children }: { children: React.ReactNode }) {
    return (
      <ErrorBoundary fallbackRender={({ error }) => <p>Something went wrong: {error.message}</p>}> {children} </ErrorBoundary>
    )
  }
  ```
- Wrap **layout.tsx** with `<RootErrorBoundary>`.

### 9. Production Deployment Considerations
- **Environmental variables**: `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` – never commit secrets.
- **Edge Runtime**: Deploy API routes to Vercel Edge for low latency (`export const runtime = 'edge'`).
- **Static Optimization**: Mark pages with `export const revalidate = 60` for ISR.
- **Performance**: Use **Next/Image**, **React Server Components**, and **prefetch** links.
- **Security**: Enable **CSP**, **Helmet**, and **Rate limiting** on API routes.
- **Monitoring**: Integrate with Vercel Analytics or OpenTelemetry.

---

*Generated by the `nextjs-and-clerk` skill.*
