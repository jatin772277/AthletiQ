import { motion } from "framer-motion";
import { useEffect, useMemo } from "react";
import { num, teamColor, teamHex, teamName, ZONES } from "../format";
import type { MatchBundle, PlayerCard } from "../types";
import { IconClose } from "./Icons";
import { CountUp, GrowBar, RingGauge, Rise, Stagger, ease } from "./Motion";
import { Pitch, PITCH_H, PITCH_W } from "./Pitch";

export function PlayerDrawer({ player: p, data, onClose }: { player: PlayerCard; data: MatchBundle; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => { window.removeEventListener("keydown", onKey); document.body.style.overflow = ""; };
  }, [onClose]);

  const c = p.collective_metrics;
  const prim = p.primary_trajectory_kinematics;
  const bucket = p.removed_players_bucket;
  const sp = p.spatial_distribution;
  const zoneTotal = ZONES.reduce((s, z) => s + c.intensity_zones_meters[z.key], 0) || 1;
  const maxPeak = Math.max(...data.players.map((x) => x.collective_metrics.peak_speed_kmh), 1);
  const partners = useMemo(
    () =>
      data.graph.edges
        .filter((e) => e.player_u === p.player_id || e.player_v === p.player_id)
        .map((e) => ({ id: e.player_u === p.player_id ? e.player_v : e.player_u, w: e.proximity_weight }))
        .sort((a, b) => b.w - a.w)
        .slice(0, 5),
    [data.graph.edges, p.player_id],
  );
  const byId = useMemo(() => new Map(data.players.map((x) => [x.player_id, x])), [data.players]);

  return (
    <>
      <motion.div className="backdrop" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
      <motion.aside
        className="drawer"
        role="dialog"
        aria-label={`Player ${p.player_id}`}
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", stiffness: 260, damping: 30 }}
      >
        <button className="drawer-close" onClick={onClose} aria-label="Close"><IconClose /></button>
        <Stagger>
          <Rise style={{ display: "flex", alignItems: "center", gap: 18 }}>
            <motion.div className={`jersey lg ${p.team}`} initial={{ rotateY: 90 }} animate={{ rotateY: 0 }} transition={{ duration: 0.7, ease, delay: 0.1 }}
              style={{ boxShadow: `0 0 40px ${teamHex(p.team)}44` }}>
              {p.player_id}
            </motion.div>
            <div>
              <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
                <span className={`chip ${p.team}`}>{teamName(p.team)}</span>
                <span className="chip live">Rank #{p.dsa_rank}</span>
              </div>
              <h2>Player {p.player_id}</h2>
              <div className="muted" style={{ marginTop: 4 }}>{p.tactical_profile}</div>
            </div>
          </Rise>

          <Rise>
            <section className="gauges">
              <RingGauge value={p.dsa_activity_index} label="DSA activity" />
              <RingGauge value={p.ml_performance_score} label="ML score" color="var(--team-a)" />
              <RingGauge value={p.team_confidence * 100} label="Team confidence" color="var(--amber)" display={<CountUp value={p.team_confidence * 100} suffix="%" />} />
            </section>
          </Rise>

          <Rise>
            <section>
              <div className="eyebrow" style={{ marginBottom: 10 }}>Kinematics</div>
              <div className="grid g-2" style={{ gap: 12 }}>
                <Stat label="Total distance" value={c.total_distance_meters} unit="m" d={2} />
                <Stat label="Sprint distance" value={c.sprint_distance_meters} unit="m" d={1} />
                <div className="card" style={{ padding: 16 }}>
                  <div className="eyebrow">Peak speed</div>
                  <div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 30 }}><CountUp value={c.peak_speed_kmh} decimals={1} /> <small className="muted" style={{ fontSize: 14 }}>km/h</small></div>
                  <div className="bar-cell"><div className="track"><GrowBar className="fill" color="" pct={(c.peak_speed_kmh / maxPeak) * 100} delay={0.3} /></div></div>
                  <div className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>Avg {num(c.average_speed_kmh, 1)} km/h · {num(c.average_speed_mps, 2)} m/s</div>
                </div>
                <div className="card" style={{ padding: 16 }}>
                  <div className="eyebrow">Explosive output</div>
                  <div style={{ display: "flex", gap: 22, marginTop: 4 }}>
                    <div><div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 30 }}><CountUp value={c.sprint_count} /></div><div className="muted" style={{ fontSize: 11.5 }}>sprints</div></div>
                    <div><div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 30 }}><CountUp value={c.total_touches_detected} /></div><div className="muted" style={{ fontSize: 11.5 }}>touches</div></div>
                    <div><div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 30 }}><CountUp value={c.total_observations} /></div><div className="muted" style={{ fontSize: 11.5 }}>frames</div></div>
                  </div>
                </div>
              </div>
            </section>
          </Rise>

          <Rise>
            <section>
              <div className="eyebrow" style={{ marginBottom: 10 }}>Intensity zones</div>
              <div className="stack-bar">
                {ZONES.map((z, i) => (
                  <GrowBar key={z.key} pct={(c.intensity_zones_meters[z.key] / zoneTotal) * 100} color={z.color} delay={0.2 + i * 0.12} style={{ borderRadius: 4 }} />
                ))}
              </div>
              <div className="legend">
                {ZONES.map((z) => <span key={z.key}><i style={{ background: z.color }} />{z.label} {num(c.intensity_zones_meters[z.key], 1)} m</span>)}
              </div>
            </section>
          </Rise>

          <Rise>
            <section>
              <div className="eyebrow" style={{ marginBottom: 10 }}>Spatial footprint</div>
              <div className="card" style={{ padding: 10 }}>
                <Pitch>
                  {[sp.defensive_third_pct, sp.middle_third_pct, sp.attacking_third_pct].map((pct, i) => (
                    <motion.rect key={i} x={(PITCH_W / 3) * i} y={0} width={PITCH_W / 3} height={PITCH_H} fill={teamHex(p.team)}
                      initial={{ opacity: 0 }} animate={{ opacity: (pct / 100) * 0.5 }} transition={{ duration: 1, delay: 0.4 + i * 0.15 }} />
                  ))}
                  {[sp.defensive_third_pct, sp.middle_third_pct, sp.attacking_third_pct].map((pct, i) => (
                    <text key={i} x={(PITCH_W / 3) * (i + 0.5)} y={PITCH_H - 4} textAnchor="middle" fill="rgba(255,255,255,.75)" style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 5 }}>
                      {num(pct, 1)}%
                    </text>
                  ))}
                  <motion.circle cx={sp.pitch_centroid_meters[0]} cy={sp.pitch_centroid_meters[1]} fill="none" stroke={teamHex(p.team)} strokeWidth={0.4}
                    initial={{ r: 0, opacity: 1 }} animate={{ r: 9, opacity: 0 }} transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }} />
                  <circle cx={sp.pitch_centroid_meters[0]} cy={sp.pitch_centroid_meters[1]} r={2} fill={teamHex(p.team)} filter="url(#glow)" />
                </Pitch>
                <div className="legend" style={{ justifyContent: "space-between", padding: "0 6px 4px" }}>
                  <span>Defensive</span><span>Middle</span><span>Attacking</span>
                  <span className="mono">centroid ({num(sp.pitch_centroid_meters[0], 1)}, {num(sp.pitch_centroid_meters[1], 1)}) m</span>
                </div>
              </div>
            </section>
          </Rise>

          <Rise>
            <section>
              <div className="eyebrow" style={{ marginBottom: 10 }}>Identity stitching provenance</div>
              <div className="provenance">
                <div className="box"><div className="v ta">{num(prim.distance_meters, 1)}m</div><div className="l">Primary track · {prim.observations} fr</div></div>
                <div className="op">+</div>
                <div className="box"><div className="v" style={{ color: "var(--amber)" }}>{num(bucket.absorbed_distance_meters, 1)}m</div><div className="l">{bucket.removed_players_count} absorbed · {bucket.absorbed_observations} fr</div></div>
                <div className="op">=</div>
                <div className="box" style={{ borderColor: "rgba(198,255,61,.35)" }}><div className="v volt">{num(c.total_distance_meters, 1)}m</div><div className="l">Collective · {num(c.observation_retention_pct, 0)}% kept</div></div>
              </div>
              <Tracklets player={p} />
            </section>
          </Rise>

          {partners.length > 0 && (
            <Rise>
              <section>
                <div className="eyebrow" style={{ marginBottom: 10 }}>Closest interaction partners</div>
                {partners.map((pt, i) => {
                  const other = byId.get(pt.id);
                  return (
                    <div key={pt.id} className="leader" style={{ gridTemplateColumns: "44px 1fr 120px auto" }}>
                      <div className={`jersey ${other?.team ?? "team_a"}`}>{pt.id}</div>
                      <div style={{ fontWeight: 600 }}>Player {pt.id} <span className="muted" style={{ fontSize: 12 }}>{other ? teamName(other.team) : ""}</span></div>
                      <div className="bar-cell"><div className="track"><GrowBar pct={(pt.w / partners[0].w) * 100} color={other ? teamColor(other.team) : "var(--volt)"} delay={0.2 + i * 0.06} style={{ borderRadius: 8 }} /></div></div>
                      <div className="mono muted" style={{ fontSize: 12 }}>{pt.w}</div>
                    </div>
                  );
                })}
              </section>
            </Rise>
          )}
        </Stagger>
      </motion.aside>
    </>
  );
}

function Stat({ label, value, unit, d = 0 }: { label: string; value: number; unit: string; d?: number }) {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="eyebrow">{label}</div>
      <div style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 30 }}><CountUp value={value} decimals={d} /> <small className="muted" style={{ fontSize: 14 }}>{unit}</small></div>
    </div>
  );
}

/** Absorbed tracklets laid out on the frame axis, packed into non-overlapping lanes. */
function Tracklets({ player }: { player: PlayerCard }) {
  const tr = player.removed_players_bucket.absorbed_tracklets;
  const layout = useMemo(() => {
    if (!tr.length) return null;
    const lo = Math.min(...tr.map((t) => t.frame_span[0]));
    const hi = Math.max(...tr.map((t) => t.frame_span[1]));
    const laneEnds: number[] = [];
    const items = [...tr].sort((a, b) => a.frame_span[0] - b.frame_span[0]).map((t) => {
      let lane = laneEnds.findIndex((end) => end < t.frame_span[0] - 4);
      if (lane === -1) { lane = laneEnds.length; laneEnds.push(0); }
      laneEnds[lane] = t.frame_span[1];
      return { t, lane };
    });
    return { lo, hi: Math.max(hi, lo + 1), items, lanes: laneEnds.length };
  }, [tr]);
  if (!layout) return <div className="muted" style={{ marginTop: 12, fontSize: 12.5 }}>No absorbed tracklets — this identity was tracked continuously.</div>;
  const laneH = Math.min(14, 84 / layout.lanes);
  const span = layout.hi - layout.lo;
  return (
    <div style={{ marginTop: 14 }}>
      <div className="timeline" style={{ height: layout.lanes * laneH + 40 }}>
        <div style={{ position: "absolute", left: 10, right: 10, top: 0, bottom: 0 }}>
        {layout.items.map(({ t, lane }, i) => (
          <motion.div
            key={t.raw_track_id}
            className="seg-bar"
            title={`Raw track ${t.raw_track_id}: frames ${t.frame_span[0]}–${t.frame_span[1]}, ${num(t.distance_meters, 2)} m, peak ${num(t.peak_speed_kmh, 1)} km/h`}
            style={{
              left: `${((t.frame_span[0] - layout.lo) / span) * 100}%`,
              top: 10 + lane * laneH,
              height: Math.max(4, laneH - 4),
              width: `max(4px, ${((t.frame_span[1] - t.frame_span[0]) / span) * 100}%)`,
              background: t.touches_detected > 0 ? "var(--volt)" : "var(--amber)",
              boxShadow: t.touches_detected > 0 ? "0 0 10px rgba(198,255,61,.6)" : "none",
              transformOrigin: "left",
            }}
            initial={{ scaleX: 0, opacity: 0 }}
            animate={{ scaleX: 1, opacity: 0.9 }}
            transition={{ duration: 0.6, ease, delay: 0.3 + i * 0.035 }}
          />
        ))}
        </div>
        <div className="axis"><span>frame {layout.lo}</span><span>frame {layout.hi}</span></div>
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--amber)" }} />Absorbed tracklet</span>
        <span><i style={{ background: "var(--volt)" }} />Tracklet with ball touch</span>
        <span className="muted">Hover a segment for details</span>
      </div>
    </div>
  );
}
