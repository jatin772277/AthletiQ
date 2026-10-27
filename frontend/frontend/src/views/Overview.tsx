import { motion } from "framer-motion";
import { useMemo } from "react";
import type { ViewProps } from "../App";
import { IconBall, IconBolt, IconCheck, IconRoute, IconShield } from "../components/Icons";
import { CountUp, GrowBar, Rise, Stagger, ease } from "../components/Motion";
import { Pitch } from "../components/Pitch";
import { num, shortRole, teamHex } from "../format";
import type { PlayerCard } from "../types";

// Deterministic pseudo-random so the "live" drift is stable between renders.
const rand = (seed: number) => {
  const x = Math.sin(seed * 9301 + 49297) * 233280;
  return x - Math.floor(x);
};

export function LivePlayers({ players, highlight }: { players: PlayerCard[]; highlight?: number | null }) {
  return (
    <>
      {players.map((p, i) => {
        const [x, y] = p.spatial_distribution.pitch_centroid_meters;
        const c = teamHex(p.team);
        const dx = (rand(p.player_id) - 0.5) * 4;
        const dy = (rand(p.player_id + 99) - 0.5) * 3;
        return (
          <motion.g
            key={p.player_id}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: highlight && highlight !== p.player_id ? 0.35 : 1, scale: 1 }}
            transition={{ delay: 0.6 + i * 0.025, type: "spring", stiffness: 260, damping: 18 }}
            style={{ transformOrigin: `${x}px ${y}px`, transformBox: "view-box" }}
          >
            <motion.g
              animate={{ x: [0, dx, -dx * 0.6, 0], y: [0, dy, -dy * 0.8, 0] }}
              transition={{ duration: 6 + rand(p.player_id + 7) * 5, repeat: Infinity, ease: "easeInOut" }}
            >
              <circle cx={x} cy={y} r={2.6} fill={c} opacity={0.18} />
              <circle cx={x} cy={y} r={1.35} fill={c} filter="url(#glow)" />
              <text x={x} y={y - 2.6} textAnchor="middle" className="node-label" fill="rgba(255,255,255,0.85)">{p.player_id}</text>
            </motion.g>
          </motion.g>
        );
      })}
    </>
  );
}

function BallFlight({ players, touches }: { players: PlayerCard[]; touches: Record<string, number> }) {
  const pts = useMemo(() => {
    const ranked = [...players]
      .filter((p) => (touches[p.player_id] ?? 0) > 0)
      .sort((a, b) => (touches[b.player_id] ?? 0) - (touches[a.player_id] ?? 0))
      .slice(0, 7);
    // Order the passing chain left→right then back, so the ball sweeps the pitch.
    const chain = ranked.map((p) => p.spatial_distribution.pitch_centroid_meters);
    return chain.length ? [...chain, chain[0]] : [];
  }, [players, touches]);
  if (pts.length < 2) return null;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const d = "M" + pts.map((p) => p.join(" ")).join("L");
  return (
    <g>
      <motion.path d={d} fill="none" stroke="rgba(198,255,61,0.35)" strokeWidth={0.3} strokeDasharray="1 1.2"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 2.4, ease, delay: 1.2 }} />
      <motion.circle r={1} fill="#fff" filter="url(#glow)"
        initial={{ cx: xs[0], cy: ys[0], opacity: 0 }}
        animate={{ cx: xs, cy: ys, opacity: 1 }}
        transition={{ duration: pts.length * 1.3, repeat: Infinity, ease: "easeInOut", delay: 1.4, opacity: { delay: 1.4 } }} />
    </g>
  );
}

function Donut({ a, b }: { a: number; b: number }) {
  const r = 52, c = 2 * Math.PI * r;
  const total = a + b || 1;
  const pa = a / total;
  return (
    <div style={{ position: "relative", width: 150, height: 150, flex: "none" }}>
      <svg width={150} height={150} viewBox="0 0 150 150" style={{ transform: "rotate(-90deg)" }}>
        <circle cx={75} cy={75} r={r} stroke="var(--surface-2)" strokeWidth={14} fill="none" />
        {/* Team B fills the full ring; Team A's arc is drawn over it. */}
        <motion.circle cx={75} cy={75} r={r} stroke="var(--team-b)" strokeWidth={14} fill="none" strokeDasharray={c}
          initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: 0 }} transition={{ duration: 1.5, ease, delay: 0.3 }}
          style={{ filter: "drop-shadow(0 0 6px rgba(255,93,125,.5))" }} />
        <motion.circle cx={75} cy={75} r={r} stroke="var(--team-a)" strokeWidth={14} fill="none" strokeDasharray={c}
          initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - pa) }} transition={{ duration: 1.5, ease, delay: 0.5 }}
          style={{ filter: "drop-shadow(0 0 6px rgba(54,215,255,.6))" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center" }}>
        <div>
          <div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 34, lineHeight: 1 }} className="ta"><CountUp value={a} decimals={1} suffix="%" /></div>
          <div className="eyebrow" style={{ fontSize: 9.5 }}>Team A ball</div>
        </div>
      </div>
    </div>
  );
}

function Compare({ label, a, b, unit = "", d = 0 }: { label: string; a: number; b: number; unit?: string; d?: number }) {
  const max = Math.max(a, b, 1);
  return (
    <div className="compare-row">
      <div className="label">{label}</div>
      <div className="num ta"><CountUp value={a} decimals={d} />{unit && <small className="muted" style={{ fontSize: 12 }}>{unit}</small>}</div>
      <div className="compare-track">
        <div className="half l"><GrowBar pct={(a / max) * 100} color="linear-gradient(270deg, var(--team-a), rgba(54,215,255,.2))" style={{ borderRadius: 6 }} /></div>
        <div className="half"><GrowBar pct={(b / max) * 100} color="linear-gradient(90deg, var(--team-b), rgba(255,93,125,.2))" style={{ borderRadius: 6 }} /></div>
      </div>
      <div className="num tb r"><CountUp value={b} decimals={d} />{unit && <small className="muted" style={{ fontSize: 12 }}>{unit}</small>}</div>
    </div>
  );
}

function parsePhases(manifest: string) {
  return manifest
    .split("\n")
    .map((l) => l.match(/^\[(PHASE[^\]]*)\]\s*(.+?)\s*\|\s*STATUS:\s*(\w+)\s*\|\s*(.+)$/))
    .filter((m): m is RegExpMatchArray => !!m)
    .map((m) => ({ phase: m[1], name: m[2], status: m[3], detail: m[4] }));
}

export function Overview({ data, openPlayer, go }: ViewProps) {
  const { summary: s, players, possession, rankings, manifest } = data;
  const phases = useMemo(() => parsePhases(manifest), [manifest]);
  const top = rankings.slice(0, 5);
  const cardById = useMemo(() => new Map(players.map((p) => [p.player_id, p])), [players]);
  const peak = useMemo(() => players.reduce((m, p) => (p.collective_metrics.peak_speed_kmh > m.collective_metrics.peak_speed_kmh ? p : m), players[0]), [players]);

  return (
    <Stagger className="grid" >
      <div className="grid g-hero">
        <Rise className="card hero">
          <div className="hero-top">
            <div className="scoreline">
              <div className="team"><span className="eyebrow">Home</span><span className="name ta">Team A</span></div>
              <span className="vs">VS</span>
              <div className="team"><span className="eyebrow">Away</span><span className="name tb">Team B</span></div>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <span className="chip">{s.video_filename}</span>
              <span className="chip mono">{num(s.total_frames)} frames · {num(s.duration_seconds, 1)}s</span>
              <span className="chip live">{s.total_active_players} tracked</span>
            </div>
          </div>
          <div className="hero-pitch">
            <Pitch scan>
              <BallFlight players={players} touches={possession.player_touches} />
              <LivePlayers players={players} />
            </Pitch>
          </div>
        </Rise>

        <Rise className="card">
          <div className="card-head">
            <h3 className="card-title">Possession & Control</h3>
            <span className="chip mono">{s.pitch_dimensions}</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 22, marginBottom: 10, flexWrap: "wrap" }}>
            <Donut a={s.team_a.possession_pct} b={s.team_b.possession_pct} />
            <div style={{ display: "grid", gap: 12, flex: 1, minWidth: 140 }}>
              <div><div className="eyebrow">Team A</div><div className="ta" style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 28 }}>{num(s.team_a.possession_pct, 1)}%</div></div>
              <div><div className="eyebrow">Team B</div><div className="tb" style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 28 }}>{num(s.team_b.possession_pct, 1)}%</div></div>
            </div>
          </div>
          <Compare label="Distance covered" a={s.team_a.total_collective_distance_meters} b={s.team_b.total_collective_distance_meters} unit="m" />
          <Compare label="Sprint episodes" a={s.team_a.total_collective_sprints} b={s.team_b.total_collective_sprints} />
          <Compare label="Ball touches" a={s.team_a.total_collective_touches} b={s.team_b.total_collective_touches} />
          <Compare label="Avg distance / player" a={s.team_a.average_player_collective_distance_m} b={s.team_b.average_player_collective_distance_m} unit="m" d={1} />
        </Rise>
      </div>

      <div className="grid g-4">
        <Kpi icon={<IconRoute />} label="Total distance" value={s.match_total_distance_meters} unit="m" d={1} a={s.team_a.total_collective_distance_meters} b={s.team_b.total_collective_distance_meters} />
        <Kpi icon={<IconBolt />} label="Sprint episodes" value={s.match_total_sprints} a={s.team_a.total_collective_sprints} b={s.team_b.total_collective_sprints} />
        <Kpi icon={<IconBall />} label="Ball touches" value={s.match_total_touches} a={s.team_a.total_collective_touches} b={s.team_b.total_collective_touches} />
        <Rise className="card kpi">
          <div className="icon"><IconShield /></div>
          <div className="eyebrow">Peak speed</div>
          <div className="value"><CountUp value={peak?.collective_metrics.peak_speed_kmh ?? 0} decimals={1} /><span className="unit">km/h</span></div>
          <div className="foot">
            <span className={`chip ${peak?.team}`} style={{ cursor: "pointer" }} onClick={() => peak && openPlayer(peak.player_id)}>Player {peak?.player_id}</span>
            <span>{num(s.observation_retention_pct, 1)}% observations kept</span>
          </div>
        </Rise>
      </div>

      <div className="grid g-2">
        <Rise className="card">
          <div className="card-head">
            <div><h3 className="card-title">Top Performers</h3><div className="card-sub">DSA max-heap activity index</div></div>
            <button className="btn-ghost" onClick={() => go("rankings")}>Full rankings →</button>
          </div>
          {top.map((r, i) => {
            const card = cardById.get(r.canonical_id);
            return (
              <motion.div key={r.canonical_id} className="leader" style={{ cursor: "pointer" }} onClick={() => openPlayer(r.canonical_id)}
                initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 + i * 0.08, ease, duration: 0.5 }} whileHover={{ x: 4 }}>
                <div className={`rank-num ${i < 3 ? "top" : ""}`}>{r.dsa_rank}</div>
                <div className={`jersey ${r.team_label}`}>{r.canonical_id}</div>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600 }}>Player {r.canonical_id} <span className={r.team_label === "team_a" ? "ta" : "tb"} style={{ fontSize: 12 }}>· {r.team_label === "team_a" ? "A" : "B"}</span></div>
                  <div className="muted" style={{ fontSize: 12, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {card ? shortRole(card.tactical_profile) : ""} · {num(r.collective_distance_meters, 1)} m · {r.collective_sprint_count} sprints
                  </div>
                  <div className="bar-cell" style={{ marginTop: 6 }}><div className="track"><GrowBar className="fill" pct={r.dsa_activity_score_0_100} color="" delay={0.4 + i * 0.08} /></div></div>
                </div>
                <div className="score-pill volt">{num(r.dsa_activity_score_0_100, 1)}</div>
              </motion.div>
            );
          })}
        </Rise>

        <Rise className="card">
          <div className="card-head">
            <div><h3 className="card-title">Pipeline Execution</h3><div className="card-sub">Computer-vision → analytics chain</div></div>
            <span className="chip live">{phases.filter((p) => p.status === "SUCCESS").length}/{phases.length} passed</span>
          </div>
          <div style={{ maxHeight: 390, overflowY: "auto", paddingRight: 6 }}>
            {phases.map((p, i) => (
              <motion.div key={p.phase} className="phase" initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.35 + i * 0.06, ease }}>
                <motion.div className="dot" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.45 + i * 0.06, type: "spring", stiffness: 400, damping: 15 }}>
                  <IconCheck size={14} />
                </motion.div>
                <div>
                  <div className="t">{p.name} <span className="mono muted" style={{ fontSize: 10.5, marginLeft: 6 }}>{p.phase}</span></div>
                  <div className="d">{p.detail}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </Rise>
      </div>
    </Stagger>
  );
}

function Kpi({ icon, label, value, unit, d = 0, a, b }: { icon: React.ReactNode; label: string; value: number; unit?: string; d?: number; a: number; b: number }) {
  const total = a + b || 1;
  return (
    <Rise className="card kpi">
      <div className="icon">{icon}</div>
      <div className="eyebrow">{label}</div>
      <div className="value"><CountUp value={value} decimals={d} />{unit && <span className="unit">{unit}</span>}</div>
      <div className="foot">
        <span className="ta mono">{Math.round((a / total) * 100)}%</span>
        <div className="split-bar">
          <GrowBar pct={(a / total) * 100} color="var(--team-a)" delay={0.3} />
          <GrowBar pct={(b / total) * 100} color="var(--team-b)" delay={0.3} />
        </div>
        <span className="tb mono">{Math.round((b / total) * 100)}%</span>
      </div>
    </Rise>
  );
}
