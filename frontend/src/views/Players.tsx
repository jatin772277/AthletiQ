import { AnimatePresence, motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useMemo, useState, type MouseEvent } from "react";
import type { ViewProps } from "../App";
import { IconSearch } from "../components/Icons";
import { ease } from "../components/Motion";
import { Seg } from "../components/Seg";
import { num, shortRole, ZONES } from "../format";
import type { PlayerCard } from "../types";

type TeamFilter = "all" | "team_a" | "team_b";
type SortKey = "rank" | "distance" | "speed" | "sprints" | "touches";

const SORTS: Record<SortKey, { label: string; get: (p: PlayerCard) => number; asc?: boolean }> = {
  rank: { label: "DSA rank", get: (p) => p.dsa_rank, asc: true },
  distance: { label: "Distance", get: (p) => p.collective_metrics.total_distance_meters },
  speed: { label: "Peak speed", get: (p) => p.collective_metrics.peak_speed_kmh },
  sprints: { label: "Sprints", get: (p) => p.collective_metrics.sprint_count },
  touches: { label: "Touches", get: (p) => p.collective_metrics.total_touches_detected },
};

export function Players({ data, openPlayer }: ViewProps) {
  const [team, setTeam] = useState<TeamFilter>("all");
  const [sort, setSort] = useState<SortKey>("rank");
  const [q, setQ] = useState("");

  const list = useMemo(() => {
    const s = SORTS[sort];
    const needle = q.trim().toLowerCase();
    return data.players
      .filter((p) => team === "all" || p.team === team)
      .filter((p) => !needle || String(p.player_id) === needle || p.tactical_profile.toLowerCase().includes(needle))
      .sort((a, b) => (s.asc ? s.get(a) - s.get(b) : s.get(b) - s.get(a)));
  }, [data.players, team, sort, q]);

  return (
    <div>
      <div className="toolbar">
        <Seg value={team} onChange={setTeam} options={[{ value: "all", label: "All 32" }, { value: "team_a", label: "Team A" }, { value: "team_b", label: "Team B" }]} />
        <div className="search"><IconSearch size={16} /><input placeholder="Player # or role…" value={q} onChange={(e) => setQ(e.target.value)} /></div>
        <select className="select" value={sort} onChange={(e) => setSort(e.target.value as SortKey)} aria-label="Sort players" style={{ marginLeft: "auto" }}>
          {(Object.keys(SORTS) as SortKey[]).map((k) => <option key={k} value={k}>Sort: {SORTS[k].label}</option>)}
        </select>
      </div>

      <motion.div layout className="player-grid">
        <AnimatePresence mode="popLayout">
          {list.map((p, i) => <PlayerTile key={p.player_id} p={p} i={i} onOpen={() => openPlayer(p.player_id)} />)}
        </AnimatePresence>
      </motion.div>
      {list.length === 0 && <div className="center-state" style={{ minHeight: 200 }}><span className="muted">No players match “{q}”.</span></div>}
    </div>
  );
}

function PlayerTile({ p, i, onOpen }: { p: PlayerCard; i: number; onOpen: () => void }) {
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rx = useSpring(useTransform(my, [-0.5, 0.5], [7, -7]), { stiffness: 250, damping: 20 });
  const ry = useSpring(useTransform(mx, [-0.5, 0.5], [-9, 9]), { stiffness: 250, damping: 20 });
  const glowX = useTransform(mx, [-0.5, 0.5], ["0%", "100%"]);
  const glowY = useTransform(my, [-0.5, 0.5], ["0%", "100%"]);
  const glow = useTransform([glowX, glowY], ([x, y]) => `radial-gradient(240px circle at ${x} ${y}, ${p.team === "team_a" ? "rgba(54,215,255,.16)" : "rgba(255,93,125,.16)"}, transparent 70%)`);

  const glowOpacity = useSpring(0, { stiffness: 200, damping: 25 });

  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    mx.set((e.clientX - r.left) / r.width - 0.5);
    my.set((e.clientY - r.top) / r.height - 0.5);
    glowOpacity.set(1);
  };
  const reset = () => { mx.set(0); my.set(0); glowOpacity.set(0); };
  const c = p.collective_metrics;
  const zoneTotal = ZONES.reduce((s, z) => s + c.intensity_zones_meters[z.key], 0) || 1;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 24, scale: 0.94 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
      transition={{ duration: 0.5, ease, delay: Math.min(i, 16) * 0.025 }}
      style={{ perspective: 900 }}
    >
      <motion.div className="card pcard" onMouseMove={onMove} onMouseLeave={reset} onClick={onOpen} style={{ rotateX: rx, rotateY: ry, transformStyle: "preserve-3d" }} whileTap={{ scale: 0.97 }}>
        <motion.div style={{ position: "absolute", inset: 0, background: glow, opacity: glowOpacity, pointerEvents: "none" }} />
        <div className={`stripe ${p.team}`} />
        <div className={`rank-badge ${p.dsa_rank <= 3 ? "top" : ""}`}>#{p.dsa_rank}</div>
        <div className="top">
          <div className={`jersey ${p.team}`}>{p.player_id}</div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 700 }}>Player {p.player_id}</div>
            <div className="role">{shortRole(p.tactical_profile)}</div>
          </div>
        </div>
        <div className="stats">
          <div className="mini-stat"><div className="v">{num(c.total_distance_meters, 0)}<small className="muted" style={{ fontSize: 11 }}>m</small></div><div className="l">Distance</div></div>
          <div className="mini-stat"><div className="v">{num(c.peak_speed_kmh, 1)}</div><div className="l">Peak km/h</div></div>
          <div className="mini-stat"><div className="v">{c.total_touches_detected}</div><div className="l">Touches</div></div>
        </div>
        <div className="zone-bar">
          {ZONES.map((z, zi) => (
            <motion.div key={z.key} style={{ background: z.color, height: "100%" }}
              initial={{ width: 0 }} animate={{ width: `${(c.intensity_zones_meters[z.key] / zoneTotal) * 100}%` }}
              transition={{ duration: 0.9, ease, delay: 0.2 + zi * 0.05 }} />
          ))}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10, fontSize: 11.5 }} className="muted">
          <span>DSA <b className="volt">{num(p.dsa_activity_index, 1)}</b></span>
          <span>ML <b style={{ color: "var(--text)" }}>{num(p.ml_performance_score, 1)}</b></span>
          <span>{c.sprint_count} sprints</span>
        </div>
      </motion.div>
    </motion.div>
  );
}
