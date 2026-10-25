import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useState, type ComponentType } from "react";
import { api } from "./api";
import { IconOverview, IconPitch, IconPlayers, IconReport, IconSpark, IconTrophy, Logo } from "./components/Icons";
import { ease } from "./components/Motion";
import { PlayerDrawer } from "./components/PlayerDrawer";
import type { MatchBundle, MatchListItem } from "./types";
import { Assistant } from "./views/Assistant";
import { Overview } from "./views/Overview";
import { Players } from "./views/Players";
import { Rankings } from "./views/Rankings";
import { Report } from "./views/Report";
import { TacticalMap } from "./views/TacticalMap";

export interface ViewProps {
  matchId: string;
  data: MatchBundle;
  openPlayer: (id: number) => void;
  go: (v: ViewKey) => void;
}

const VIEWS = {
  overview: { label: "Match Overview", title: ["Match", "Overview"], icon: IconOverview, comp: Overview },
  players: { label: "Players", title: ["Squad", "Kinematics"], icon: IconPlayers, comp: Players },
  rankings: { label: "Rankings", title: ["Performance", "Rankings"], icon: IconTrophy, comp: Rankings },
  tactical: { label: "Tactical Map", title: ["Tactical", "Network"], icon: IconPitch, comp: TacticalMap },
  assistant: { label: "AI Analyst", title: ["Tactical", "AI Analyst"], icon: IconSpark, comp: Assistant },
  report: { label: "Scouting Report", title: ["Scouting", "Report"], icon: IconReport, comp: Report },
} satisfies Record<string, { label: string; title: [string, string]; icon: ComponentType<{ size?: number }>; comp: ComponentType<ViewProps> }>;

export type ViewKey = keyof typeof VIEWS;

const initialView = (): ViewKey => {
  const h = window.location.hash.slice(1);
  return h in VIEWS ? (h as ViewKey) : "overview";
};

export default function App() {
  const [view, setView] = useState<ViewKey>(initialView);
  const [matches, setMatches] = useState<MatchListItem[]>([]);
  const [matchId, setMatchId] = useState<string>("");
  const [data, setData] = useState<MatchBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState(false);
  const [player, setPlayer] = useState<number | null>(null);

  const go = useCallback((v: ViewKey) => {
    setView(v);
    window.location.hash = v;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const boot = useCallback(async () => {
    setError(null);
    try {
      const [health, list] = await Promise.all([api.health(), api.matches()]);
      setOnline(true);
      setMatches(list);
      const preferred = list.find((m) => m.match_id === health.default_match) ?? list[0];
      if (!preferred) throw new Error("No processed matches found. Run the AthletiQ pipeline first.");
      setMatchId((cur) => cur || preferred.match_id);
    } catch (e) {
      setOnline(false);
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => { boot(); }, [boot]);

  useEffect(() => {
    const onHash = () => setView(initialView());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (!matchId) return;
    let cancelled = false;
    setData(null);
    api.bundle(matchId).then(
      (b) => !cancelled && setData(b),
      (e) => !cancelled && setError((e as Error).message),
    );
    return () => { cancelled = true; };
  }, [matchId]);

  const View = VIEWS[view].comp;
  const [t1, t2] = VIEWS[view].title;
  const selected = player != null ? data?.players.find((p) => p.player_id === player) : undefined;

  return (
    <>
      <div className="ambient"><div className="glow a" /><div className="glow b" /><div className="glow c" /><div className="grid" /></div>
      <div className="shell">
        <aside className="sidebar">
          <div className="brand">
            <motion.div className="brand-mark" initial={{ rotate: -90, scale: 0 }} animate={{ rotate: 0, scale: 1 }} transition={{ type: "spring", stiffness: 220, damping: 14 }}>
              <Logo />
            </motion.div>
            <div>
              <div className="brand-name">Athleti<em>Q</em></div>
              <div className="brand-sub">Performance Intelligence</div>
            </div>
          </div>
          <nav className="nav">
            <div className="nav-label">Analytics</div>
            {(Object.keys(VIEWS) as ViewKey[]).map((k) => {
              const Icon = VIEWS[k].icon;
              return (
                <button key={k} className={`nav-item ${view === k ? "active" : ""}`} onClick={() => go(k)}>
                  {view === k && <motion.span layoutId="nav-pill" className="pill" transition={{ type: "spring", stiffness: 380, damping: 32 }} />}
                  <Icon />
                  {VIEWS[k].label}
                </button>
              );
            })}
          </nav>
          <div className="sidebar-foot">
            <div className="status-row"><span className={`status-dot ${online ? "" : "off"}`} />{online ? "Engine online" : "Engine offline"}</div>
            <div className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>
              CV tracking · PnLCalib homography · DSA ranking · ML scoring
            </div>
          </div>
        </aside>

        <main className="main">
          <header className="topbar">
            <div>
              <div className="crumb">AthletiQ / {matchId || "—"}</div>
              <AnimatePresence mode="wait">
                <motion.h1 key={view} className="page-title" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.35, ease }}>
                  {t1} <span className="accent">{t2}</span>
                </motion.h1>
              </AnimatePresence>
            </div>
            <div className="topbar-right">
              {data && <span className="chip live"><span className="status-dot" />Analysis complete</span>}
              {matches.length > 0 && (
                <select className="select" value={matchId} onChange={(e) => setMatchId(e.target.value)} aria-label="Select match">
                  {matches.map((m) => <option key={m.match_id} value={m.match_id}>{m.match_id}</option>)}
                </select>
              )}
            </div>
          </header>

          {error ? (
            <ErrorState message={error} retry={boot} />
          ) : !data ? (
            <LoadingState />
          ) : (
            <AnimatePresence mode="wait">
              <motion.div key={view + matchId} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8, filter: "blur(4px)" }} transition={{ duration: 0.35, ease }}>
                <View matchId={matchId} data={data} openPlayer={setPlayer} go={go} />
              </motion.div>
            </AnimatePresence>
          )}
        </main>
      </div>

      <nav className="mobile-nav">
        {(Object.keys(VIEWS) as ViewKey[]).map((k) => {
          const Icon = VIEWS[k].icon;
          return <button key={k} className={view === k ? "active" : ""} onClick={() => go(k)} aria-label={VIEWS[k].label}><Icon size={20} /></button>;
        })}
      </nav>

      <AnimatePresence>
        {selected && data && <PlayerDrawer key={selected.player_id} player={selected} data={data} onClose={() => setPlayer(null)} />}
      </AnimatePresence>
    </>
  );
}

function LoadingState() {
  return (
    <div className="center-state">
      <div>
        <div className="loader" />
        <div className="eyebrow">Loading match intelligence</div>
      </div>
    </div>
  );
}

function ErrorState({ message, retry }: { message: string; retry: () => void }) {
  return (
    <div className="center-state">
      <motion.div className="card" style={{ maxWidth: 460, padding: 30 }} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}>
        <div className="eyebrow tb">Connection lost</div>
        <h2 className="card-title" style={{ fontSize: 28, margin: "8px 0 10px" }}>Can't reach the engine</h2>
        <p className="muted" style={{ marginTop: 0 }}>{message}</p>
        <p className="muted" style={{ fontSize: 12.5 }}>Start the API with <code className="mono volt">python backend/run.py</code> and retry.</p>
        <button className="btn-volt" style={{ height: 42, marginTop: 8 }} onClick={retry}>Retry connection</button>
      </motion.div>
    </div>
  );
}
