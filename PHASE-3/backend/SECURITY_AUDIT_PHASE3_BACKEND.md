# PHASE-3 Backend Vulnerability Check

## Scope
- Reviewed `PHASE-3/backend/src` for common backend security issues (authentication, authorization, SSRF, information disclosure, query safety).
- Attempted dependency vulnerability scan with `pip-audit`.

## Tooling Status
- `pip-audit` could not be installed in this environment due restricted package index/proxy access.
- Result: dependency CVE scanning was not completed automatically.

## Findings

### 1) High: Potential SSRF/issuer trust issue in JWT verification (**fixed**)
- **Issue**: JWT `iss` claim was used before signature verification to build the JWKS URL, allowing attacker-controlled issuer values to influence outbound HTTP calls.
- **Risk**: SSRF and trust of untrusted identity providers.
- **Fix applied**:
  - Enforce HTTPS issuer URLs.
  - Add issuer allow-list enforcement (`CLERK_ALLOWED_ISSUERS` and `CLERK_ISSUER`).
  - Validate JWKS override URL is HTTPS.
  - Construct default JWKS URL from validated issuer netloc.

### 2) Medium: Detailed authentication errors leaked internals (**fixed**)
- **Issue**: API responses returned raw JWT error text.
- **Risk**: Information disclosure useful for token forgery/brute-force tuning.
- **Fix applied**: Replaced detailed client-facing messages with generic `Token verification failed` while retaining server logs.

### 3) Medium: Missing strict audience validation by default (**configuration required**)
- **Issue**: Audience check only runs when `CLERK_JWT_AUDIENCE` is set.
- **Risk**: Token confusion across apps/services if issuer shares keyspace.
- **Recommendation**: Set `CLERK_JWT_AUDIENCE` in production.

### 4) Low: Broad CORS methods/headers
- **Issue**: `allow_methods=["*"]` and `allow_headers=["*"]` are permissive.
- **Risk**: Larger browser attack surface than necessary.
- **Recommendation**: Restrict to required methods and headers.

## Operational Recommendations
1. Add dependency CVE scanning in CI (e.g., `pip-audit` in pipeline).
2. Add static security linting in CI (e.g., Bandit, Semgrep).
3. Rotate Clerk keys if they were ever exposed in non-template env files.
4. Set production security env vars:
   - `CLERK_ISSUER`
   - `CLERK_ALLOWED_ISSUERS`
   - `CLERK_JWT_AUDIENCE`

