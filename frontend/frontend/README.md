# AthletiQ Frontend

React + TypeScript + Framer Motion app for the AthletiQ FastAPI backend.

## Run

Start MongoDB and the API first (see [backend/README.md](../backend/README.md)), then:

```bash
npm --prefix frontend install
npm --prefix frontend run dev  # http://localhost:5173
```

Vite proxies `/api` to `http://127.0.0.1:8000`; override with `ATHLETIQ_API=http://host:port`.

## Pages

- **Sign in / Create account**: email + password accounts stored in MongoDB; sessions persist across reloads and sign-out revokes them
- **Upload Match**: drag-and-drop a match video, pick the start time and how many seconds to analyse, upload with live progress
- **My Analyses**: every upload with live pipeline progress, a phase-by-phase stepper, log, cancel/delete, and "View analysis" when ready
- **Match Overview**: live pitch, possession, team comparison, KPIs, top performers, pipeline phases
- **Players**: squad cards with filter, search and sort; click any player for the detail drawer
- **Rankings**: animated podium and leaderboard by DSA index, distance, sprints, touches or speed
- **Tactical Map**: proximity network on the pitch, sized by touches, distance or DSA score
- **AI Analyst**: chat backed by `POST /api/matches/{id}/tactical-query`
- **Scouting Report**: rendered markdown report with download

The match picker lists your completed analyses plus the shared demo matches.
