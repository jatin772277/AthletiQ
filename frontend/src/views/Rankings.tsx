import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import type { ViewProps } from "../App";
import { CountUp, GrowBar, ease } from "../components/Motion";
import { Seg } from "../components/Seg";
import { num, shortRole, teamHex } from "../format";
import type { RankingItem } from "../types";

type Metric = "score" | "distance" | "sprints" | "touches" | "speed";
const METRICS: Record<Metric, { label: string; get: (r: RankingItem) => number; unit: string; d: number }> = {
  score: { label: "DSA Index", get: (r) => r.dsa_activity_score_0_100, unit: "", d: 1 },
  distance: { label: "Distance", get: (r) => r.collective_distance_meters, unit: "m", d: 1 },
  sprints: { label: "Sprints", get: (r) => r.collective_sprint_count, unit: "", d: 0 },
  touches: { label: "Touches", get: (r) => r.collective_touches, unit: "", d: 0 },
  speed: { label: "Avg speed", get: (r) => (r.collective_average_speed_mps ?? 0) * 3.6, unit: "km/h", d: 1 },
};

export function Rankings({ data, openPlayer }: ViewProps) {
  const [metric, setMetric] = useState<Metric>("score");
  const [team, setTeam] = useState<"all" | "team_a" | "team_b">("all");
  const cards = useMemo(() => new Map(data.players.map((p) => [p.player_id, p])), [data.players]);
  const m = METRICS[metric];

  const rows = useMemo(
    () => data.rankings.filter((r) => team === "all" || r.team_label === team).sort((a, b) => m.get(b) - m.get(a) || a.dsa_rank - b.dsa_rank),
    [data.rankings, team, m],
  );
  const max = Math.max(...rows.map(m.get), 1);
  const podium = [rows[1], rows[0], rows[2]].filter(Boolean);

  return (
    <div>
      <div className="toolbar">
        <Seg value={metric} onChange={setMetric} options={(Object.keys(METRICS) as Metric[]).map((k) => ({ value: k, label: METRICS[k].label }))} />
        <Seg value={team} onChange={setTeam} options={[{ value: "all", label: "Both" }, { value: "team_a", label: "Team A" }, { value: "team_b", label: "Team B" }]} />
      </div>

      <div className="card" style={{ marginBottom: 18, padding: "26px 20px 0" }}>
        <div style={{ display: "flex", justifyContent: "center", alignItems: "flex-end", gap: "clamp(10px, 3vw, 28px)" }}>
          {podium.map((r) => {
            const place = rows.indexOf(r) + 1;
            const h = place === 1 ? 150 : place === 2 ? 112 : 84;
            const color = teamHex(r.team_label);
            return (
              <motion.div key={`${metric}-${team}-${r.canonical_id}`} style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "clamp(90px, 18vw, 170px)", cursor: "pointer" }}
                onClick={() => openPlayer(r.canonical_id)} whileHover={{ y: -4 }}>
                <motion.div className={`jersey ${r.team_label}`} style={{ width: 58, height: 58, fontSize: 26, borderRadius: 16, boxShadow: place === 1 ? "0 0 36px rgba(198,255,61,.35)" : undefined, borderColor: place === 1 ? "var(--volt)" : undefined }}
                  initial={{ y: -30, opacity: 0, scale: 0.6 }} animate={{ y: 0, opacity: 1, scale: 1 }} transition={{ type: "spring", stiffness: 300, damping: 16, delay: 0.5 + (3 - place) * 0.12 }}>
                  {r.canonical_id}
                </motion.div>
                <div style={{ fontWeight: 600, fontSize: 12.5, marginTop: 8, textAlign: "center" }}>{shortRole(cards.get(r.canonical_id)?.tactical_profile ?? "")}</div>
                <div className={place === 1 ? "volt" : ""} style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 24 }}>
                  <CountUp value={m.get(r)} decimals={m.d} /> <small className="muted" style={{ fontSize: 12 }}>{m.unit}</small>
                </div>
                <motion.div
                  style={{
                    width: "100%", marginTop: 8, borderRadius: "14px 14px 0 0",
                    background: `linear-gradient(180deg, ${place === 1 ? "rgba(198,255,61,.35)" : color + "40"}, transparent)`,
                    border: `1px solid ${place === 1 ? "rgba(198,255,61,.45)" : color + "55"}`, borderBottom: 0,
                    display: "grid", placeItems: "start center", paddingTop: 10,
                  }}
                  initial={{ height: 0 }} animate={{ height: h }} transition={{ duration: 0.9, ease, delay: 0.15 + (3 - place) * 0.1 }}>
                  <span style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: place === 1 ? 46 : 34, color: place === 1 ? "var(--volt)" : "var(--muted)", lineHeight: 1 }}>{place}</span>
                </motion.div>
              </motion.div>
            );
          })}
        </div>
      </div>

      <div className="rank-row head">
        <span>#</span><span /><span>Player</span>
        <span className="hide-sm">Distance</span><span className="hide-sm hide-md">Sprints</span><span className="hide-sm hide-md">Touches</span><span className="hide-sm">DSA rank</span>
        <span>{m.label}</span>
      </div>
      <motion.div layout>
        {rows.map((r, i) => {
          const card = cards.get(r.canonical_id);
          return (
            <motion.div key={r.canonical_id} layout="position" className="rank-row" onClick={() => openPlayer(r.canonical_id)}
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ layout: { type: "spring", stiffness: 380, damping: 36 }, delay: Math.min(i, 20) * 0.02, duration: 0.45, ease }}
              whileHover={{ scale: 1.006 }}>
              <span className={`rank-num ${i < 3 ? "top" : ""}`}>{i + 1}</span>
              <span className={`jersey ${r.team_label}`}>{r.canonical_id}</span>
              <span style={{ minWidth: 0 }}>
                <div style={{ fontWeight: 600 }}>Player {r.canonical_id}</div>
                <div className="muted" style={{ fontSize: 12, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{card ? shortRole(card.tactical_profile) : ""}</div>
              </span>
              <span className="stat hide-sm">{num(r.collective_distance_meters, 1)}<small>m</small></span>
              <span className="stat hide-sm hide-md">{r.collective_sprint_count}</span>
              <span className="stat hide-sm hide-md">{r.collective_touches}</span>
              <span className="stat hide-sm muted">#{r.dsa_rank}</span>
              <span className="bar-cell">
                <span className="track"><GrowBar key={metric + team} className="fill" color="" pct={(m.get(r) / max) * 100} delay={0.1 + Math.min(i, 20) * 0.02} /></span>
                <span className="stat" style={{ minWidth: 54, textAlign: "right" }}>{num(m.get(r), m.d)}</span>
              </span>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
}
