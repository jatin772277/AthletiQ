# AthletiQ Frontend

React + TypeScript + Framer Motion dashboard for the AthletiQ FastAPI backend.

## Run

```bash
python backend/run.py          # API on :8000 (from the repo root)
npm --prefix frontend install
npm --prefix frontend run dev  # http://localhost:5173
```

Vite proxies `/api` to `http://127.0.0.1:8000`; override with `ATHLETIQ_API=http://host:port`.

## Views

- **Match Overview**: live pitch, possession donut, team comparison, KPIs, top performers, pipeline phases
- **Players**: 32 squad cards with filter, search and sort; click any player for the detail drawer
- **Rankings**: animated podium and leaderboard, re-sorted by DSA index, distance, sprints, touches or speed
- **Tactical Map**: proximity network on the pitch, sized by touches, distance or DSA score
- **AI Analyst**: chat backed by `POST /api/matches/{id}/tactical-query`
- **Scouting Report**: rendered markdown report with download
