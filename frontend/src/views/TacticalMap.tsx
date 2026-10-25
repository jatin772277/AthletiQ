import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useState } from "react";
import type { ViewProps } from "../App";
import { GrowBar, ease } from "../components/Motion";
import { Pitch } from "../components/Pitch";
import { Seg } from "../components/Seg";
import { num, shortRole, teamHex, teamName } from "../format";

type Size = "touches" | "distance" | "score";
type EdgeFilter = "all" | "team" | "cross";

export function TacticalMap({ data, openPlayer }: ViewProps) {
  const [hover, setHover] = useState<number | null>(null);
  const [size, setSize] = useState<Size>("touches");
  const [edgeFilter, setEdgeFilter] = useState<EdgeFilter>("all");
  const byId = useMemo(() => new Map(data.players.map((p) => [p.player_id, p])), [data.players]);
  const touches = data.possession.player_touches;

  const edges = useMemo(() => {
    const maxW = Math.max(...data.graph.edges.map((e) => e.proximity_weight), 1);
    return data.graph.edges
      .map((e) => {
        const a = byId.get(e.player_u), b = byId.get(e.player_v);
        if (!a || !b) return null;
        return { ...e, a, b, same: a.team === b.team, t: e.proximity_weight / maxW };
      })
      .filter((e): e is NonNullable<typeof e> => !!e)
      .filter((e) => edgeFilter === "all" || (edgeFilter === "team" ? e.same : !e.same));
  }, [data.graph.edges, byId, edgeFilter]);

  const sizeOf = useMemo(() => {
    const val = (id: number) => {
      const p = byId.get(id)!;
      return size === "touches" ? touches[id] ?? 0 : size === "distance" ? p.collective_metrics.total_distance_meters : p.dsa_activity_index;
    };
    const max = Math.max(...data.players.map((p) => val(p.player_id)), 1);
    return (id: number) => 1.1 + (val(id) / max) * 2.6;
  }, [size, byId, data.players, touches]);

  const degree = useMemo(() => {
    const d = new Map<number, number>();
    for (const e of data.graph.edges) {
      d.set(e.player_u, (d.get(e.player_u) ?? 0) + e.proximity_weight);
      d.set(e.player_v, (d.get(e.player_v) ?? 0) + e.proximity_weight);
    }
    return d;
  }, [data.graph.edges]);

  const hubs = useMemo(() => [...degree.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6), [degree]);
  const sameCount = data.graph.edges.filter((e) => byId.get(e.player_u)?.team === byId.get(e.player_v)?.team).length;
  const focus = hover != null ? byId.get(hover) : undefined;
  const linked = (id: number) => hover == null || id === hover || edges.some((e) => (e.player_u === hover && e.player_v === id) || (e.player_v === hover && e.player_u === id));

  return (
    <div>
      <div className="toolbar">
        <Seg value={edgeFilter} onChange={setEdgeFilter} options={[{ value: "all", label: "All links" }, { value: "team", label: "Teammates" }, { value: "cross", label: "Duels" }]} />
        <Seg value={size} onChange={setSize} options={[{ value: "touches", label: "Size: touches" }, { value: "distance", label: "Size: distance" }, { value: "score", label: "Size: DSA" }]} />
      </div>
      <div className="map-layout">
        <motion.div className="card map-card" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6, ease }}>
          <Pitch>
            <g>
              {edges.map((e, i) => {
                const [x1, y1] = e.a.spatial_distribution.pitch_centroid_meters;
                const [x2, y2] = e.b.spatial_distribution.pitch_centroid_meters;
                const active = hover != null && (e.player_u === hover || e.player_v === hover);
                const dim = hover != null && !active;
                const color = e.same ? teamHex(e.a.team) : "#ffb547";
                return (
                  <motion.line key={`${e.player_u}-${e.player_v}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeLinecap="round"
                    strokeWidth={0.15 + e.t * 0.9}
                    strokeDasharray={active ? "1.4 1" : undefined}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: dim ? 0.05 : active ? 1 : 0.18 + e.t * 0.5, strokeDashoffset: active ? [0, -4.8] : 0 }}
                    transition={{
                      pathLength: { duration: 1, ease, delay: 0.4 + i * 0.008 },
                      opacity: { duration: 0.25 },
                      strokeDashoffset: active ? { duration: 0.6, repeat: Infinity, ease: "linear" } : { duration: 0 },
                    }}
                    style={active ? { filter: `drop-shadow(0 0 1px ${color})` } : undefined} />
                );
              })}
            </g>
            {data.players.map((p, i) => {
              const [x, y] = p.spatial_distribution.pitch_centroid_meters;
              const r = sizeOf(p.player_id);
              const c = teamHex(p.team);
              const on = linked(p.player_id);
              return (
                <motion.g key={p.player_id} style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHover(p.player_id)} onMouseLeave={() => setHover(null)} onClick={() => openPlayer(p.player_id)}
                  initial={{ opacity: 0 }} animate={{ opacity: on ? 1 : 0.2 }} transition={{ duration: 0.3, delay: hover == null ? 0.2 + i * 0.02 : 0 }}>
                  {hover === p.player_id && (
                    <motion.circle cx={x} cy={y} fill="none" stroke={c} strokeWidth={0.3} initial={{ r, opacity: 0.9 }} animate={{ r: r + 5, opacity: 0 }} transition={{ duration: 1.2, repeat: Infinity }} />
                  )}
                  <motion.circle cx={x} cy={y} fill={c} fillOpacity={0.2} stroke={c} strokeWidth={0.35} animate={{ r: r + 0.8 }} transition={{ type: "spring", stiffness: 200, damping: 18 }} />
                  <motion.circle cx={x} cy={y} fill={c} filter="url(#glow)" animate={{ r: r * 0.55 }} transition={{ type: "spring", stiffness: 200, damping: 18 }} />
                  <text x={x} y={y + 0.9} textAnchor="middle" className="node-label" fill="#061018">{p.player_id}</text>
                </motion.g>
              );
            })}
          </Pitch>
          <div className="legend" style={{ padding: "6px 8px 2px" }}>
            <span><i style={{ background: "var(--team-a)" }} />Team A link</span>
            <span><i style={{ background: "var(--team-b)" }} />Team B link</span>
            <span><i style={{ background: "var(--amber)" }} />Cross-team duel</span>
            <span className="muted">Line weight = time spent in proximity · hover a player</span>
          </div>
        </motion.div>

        <div className="grid" style={{ alignContent: "start" }}>
          <motion.div className="card" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2, ease, duration: 0.5 }} style={{ minHeight: 168 }}>
            <AnimatePresence mode="wait">
              {focus ? (
                <motion.div key={focus.player_id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                    <div className={`jersey ${focus.team}`}>{focus.player_id}</div>
                    <div><div style={{ fontWeight: 700 }}>Player {focus.player_id}</div><div className="muted" style={{ fontSize: 12 }}>{teamName(focus.team)} · {shortRole(focus.tactical_profile)}</div></div>
                  </div>
                  <div className="pcard" style={{ padding: 0 }}>
                    <div className="stats">
                      <div className="mini-stat"><div className="v">{touches[focus.player_id] ?? 0}</div><div className="l">Touches</div></div>
                      <div className="mini-stat"><div className="v">{edges.filter((e) => e.player_u === focus.player_id || e.player_v === focus.player_id).length}</div><div className="l">Links</div></div>
                      <div className="mini-stat"><div className="v">{degree.get(focus.player_id) ?? 0}</div><div className="l">Proximity</div></div>
                    </div>
                  </div>
                  <div className="muted" style={{ fontSize: 11.5, marginTop: 10 }}>Click to open the full player profile</div>
                </motion.div>
              ) : (
                <motion.div key="network" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="eyebrow">Interaction network</div>
                  <div style={{ display: "flex", gap: 22, marginTop: 8 }}>
                    <div><div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 34 }}>{data.graph.total_edges}</div><div className="muted" style={{ fontSize: 12 }}>proximity links</div></div>
                    <div><div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 34 }} className="volt">{num((sameCount / Math.max(data.graph.total_edges, 1)) * 100, 0)}%</div><div className="muted" style={{ fontSize: 12 }}>between teammates</div></div>
                  </div>
                  <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>Hover any player to trace their on-pitch connections.</div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          <motion.div className="card" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3, ease, duration: 0.5 }}>
            <div className="card-head"><h3 className="card-title">Network Hubs</h3><span className="card-sub">weighted degree</span></div>
            {hubs.map(([id, w], i) => {
              const p = byId.get(id);
              return (
                <div key={id} className="leader" style={{ gridTemplateColumns: "44px 1fr auto", cursor: "pointer" }} onMouseEnter={() => setHover(id)} onMouseLeave={() => setHover(null)} onClick={() => openPlayer(id)}>
                  <div className={`jersey ${p?.team ?? "team_a"}`}>{id}</div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>Player {id}</div>
                    <div className="bar-cell" style={{ marginTop: 5 }}><div className="track"><GrowBar pct={(w / hubs[0][1]) * 100} color={p ? teamHex(p.team) : "var(--volt)"} delay={0.4 + i * 0.07} style={{ borderRadius: 8 }} /></div></div>
                  </div>
                  <div className="mono muted" style={{ fontSize: 12 }}>{w}</div>
                </div>
              );
            })}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
