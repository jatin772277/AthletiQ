import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useState, type ComponentType } from "react";
import { api, session } from "./api";
import { IconFilm, IconLogout, IconOverview, IconPitch, IconPlayers, IconReport, IconSpark, IconTrophy, IconUpload, Logo } from "./components/Icons";
import { ease } from "./components/Motion";
import { PlayerDrawer } from "./components/PlayerDrawer";
import type { MatchBundle, MatchListItem, User } from "./types";
import { Analyses } from "./views/Analyses";
import { Assistant } from "./views/Assistant";
import { AuthScreen } from "./views/AuthScreen";
import { Overview } from "./views/Overview";
import { Players } from "./views/Players";
import { Rankings } from "./views/Rankings";
import { Report } from "./views/Report";
import { TacticalMap } from "./views/TacticalMap";
import { Upload } from "./views/Upload";

export interface ViewProps {
  matchId: string;
  data: MatchBundle;
  openPlayer: (id: number) => void;
  go: (v: ViewKey) => void;
}

type Icon = ComponentType<{ size?: number }>;
interface ViewDef { label: string; title: [string, string]; icon: Icon; group: "analytics" | "studio"; comp?: ComponentType<ViewProps> }

const VIEWS = {
  overview: { label: "Match Overview", title: ["Match", "Overview"], icon: IconOverview, group: "analytics", comp: Overview },
  players: { label: "Players", title: ["Squad", "Kinematics"], icon: IconPlayers, group: "analytics", comp: Players },
  rankings: { label: "Rankings", title: ["Performance", "Rankings"], icon: IconTrophy, group: "analytics", comp: Rankings },
  tactical: { label: "Tactical Map", title: ["Tactical", "Network"], icon: IconPitch, group: "analytics", comp: TacticalMap },
  assistant: { label: "AI Analyst", title: ["Tactical", "AI Analyst"], icon: IconSpark, group: "analytics", comp: Assistant },
  report: { label: "Scouting Report", title: ["Scouting", "Report"], icon: IconReport, group: "analytics", comp: Report },
  upload: { label: "Upload Match", title: ["Upload", "Match"], icon: IconUpload, group: "studio" },
  analyses: { label: "My Analyses", title: ["My", "Analyses"], icon: IconFilm, group: "studio" },
} satisfies Record<string, ViewDef>;

export type ViewKey = keyof typeof VIEWS;
const KEYS = Object.keys(VIEWS) as ViewKey[];

const initialView = (): ViewKey => {
  const h = window.location.hash.slice(1);
  return h in VIEWS ? (h as ViewKey) : "overview";
};

export default function App() {
  const [user, setUser] = useState<User | null | undefined>(session.token ? undefined : null);

  useEffect(() => {
    session.onUnauthorized(() => { session.set(null); setUser(null); });
    if (session.token) api.me().then(setUser, () => { session.set(null); setUser(null); });
  }, []);

  if (user === undefined) return <FullLoader />;
  if (user === null) return <AuthScreen onAuthed={setUser} />;
  return <Dashboard user={user} onSignOut={() => setUser(null)} />;
}

function Dashboard({ user, onSignOut }: { user: User; onSignOut: () => void }) {
  const [view, setView] = useState<ViewKey>(initialView);
  const [matches, setMatches] = useState<MatchListItem[] | null>(null);
  const [matchId, setMatchId] = useState<string>("");
  const [data, setData] = useState<MatchBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState(false);
  const [player, setPlayer] = useState<number | null>(null);
  const [reloadVersion, setReloadVersion] = useState(0);

  const go = useCallback((v: ViewKey) => {
    setView(v);
    window.location.hash = v;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const loadMatches = useCallback(async (select?: string) => {
    setError(null);
    try {
      const [health, list] = await Promise.all([api.health(), api.matches()]);
      setOnline(health.database === "connected");
      const ready = list.filter((m) => m.has_rag_package);
      setMatches(ready);
      setMatchId((cur) => {
        if (select && ready.some((m) => m.match_id === select)) return select;
        if (cur && ready.some((m) => m.match_id === cur)) return cur;
        return (ready.find((m) => m.match_id === health.default_match) ?? ready[0])?.match_id ?? "";
      });
    } catch (e) {
      setOnline(false);
      setError((e as Error).message);
    }
  }, []);

  const retryConnection = useCallback(() => {
    setError(null);
    setReloadVersion((v) => v + 1);
    loadMatches();
  }, [loadMatches]);

  useEffect(() => { loadMatches(); }, [loadMatches, reloadVersion]);

  useEffect(() => {
    const onHash = () => setView(initialView());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (!matchId) return;
    let cancelled = false;
    setData(null);
    setError(null);
    api.bundle(matchId).then(
      (b) => !cancelled && setData(b),
      (e) => !cancelled && setError((e as Error).message),
    );
    return () => { cancelled = true; };
  }, [matchId, reloadVersion]);

  const signOut = async () => {
    try { await api.logout(); } catch { /* the session is dropped locally either way */ }
    session.set(null);
    onSignOut();
  };

  const openAnalysis = async (id: string) => {
    await loadMatches(id);
    go("overview");
  };

  const def: ViewDef = VIEWS[view];
  const [t1, t2] = def.title;
  const selected = player != null ? data?.players.find((p) => p.player_id === player) : undefined;
  const current = matches?.find((m) => m.match_id === matchId);

  let body;
  if (view === "upload") body = <Upload onQueued={() => loadMatches()} goAnalyses={() => go("analyses")} />;
  else if (view === "analyses") body = <Analyses onOpen={openAnalysis} goUpload={() => go("upload")} onCompleted={() => loadMatches()} />;
  else if (error) body = <ErrorState message={error} retry={retryConnection} goAnalyses={() => go("analyses")} goUpload={() => go("upload")} />;
  else if (matches && matches.length === 0) body = <NoMatches goUpload={() => go("upload")} />;
  else if (!data) body = <LoadingState />;
  else {
    const View = def.comp!;
    body = <View matchId={matchId} data={data} openPlayer={setPlayer} go={go} />;
  }

  const initials = user.name.split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();

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
          {(["studio", "analytics"] as const).map((group) => (
            <nav className="nav" key={group}>
              <div className="nav-label">{group === "studio" ? "Studio" : "Analytics"}</div>
              {KEYS.filter((k) => VIEWS[k].group === group).map((k) => {
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
          ))}
          <div className="sidebar-foot">
            <div className="user-row">
              <div className="user-avatar">{initials}</div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div className="user-name">{user.name}</div>
                <div className="user-email">{user.email}</div>
              </div>
              <button className="icon-btn" onClick={signOut} title="Sign out" aria-label="Sign out"><IconLogout size={17} /></button>
            </div>
            <div className="status-row" style={{ marginTop: 12 }}><span className={`status-dot ${online ? "" : "off"}`} />{online ? "Engine & database online" : "Database offline"}</div>
          </div>
        </aside>

        <main className="main">
          <header className="topbar">
            <div>
              <div className="crumb">AthletiQ / {def.group === "studio" ? "studio" : current?.title ?? (matchId || "—")}</div>
              <AnimatePresence mode="wait">
                <motion.h1 key={view} className="page-title" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.35, ease }}>
                  {t1} <span className="accent">{t2}</span>
                </motion.h1>
              </AnimatePresence>
            </div>
            <div className="topbar-right">
              {def.group === "analytics" && data && <span className="chip live"><span className="status-dot" />Analysis complete</span>}
              {def.group === "analytics" && matches && matches.length > 0 && (
                <select className="select" value={matchId} onChange={(e) => setMatchId(e.target.value)} aria-label="Select match">
                  {matches.some((m) => !m.is_demo) && (
                    <optgroup label="My analyses">
                      {matches.filter((m) => !m.is_demo).map((m) => <option key={m.match_id} value={m.match_id}>{m.title}</option>)}
                    </optgroup>
                  )}
                  <optgroup label="Demo matches">
                    {matches.filter((m) => m.is_demo).map((m) => <option key={m.match_id} value={m.match_id}>{m.title}</option>)}
                  </optgroup>
                </select>
              )}
              <button className="icon-btn mobile-only" onClick={signOut} aria-label="Sign out"><IconLogout size={18} /></button>
            </div>
          </header>

          <AnimatePresence mode="wait">
            {/* Studio views don't depend on the selected match; keying them on it would remount (and reset) the upload form. */}
            <motion.div key={def.group === "studio" ? view : view + matchId + (data ? "1" : "0")} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8, filter: "blur(4px)" }} transition={{ duration: 0.35, ease }}>
              {body}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      <nav className="mobile-nav">
        {KEYS.map((k) => {
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

function FullLoader() {
  return <div className="center-state" style={{ minHeight: "100vh" }}><div className="loader" /></div>;
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

function NoMatches({ goUpload }: { goUpload: () => void }) {
  return (
    <div className="center-state">
      <motion.div className="card upload-done" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}>
        <div className="done-badge" style={{ background: "var(--surface-2)", color: "var(--volt)" }}><IconUpload size={30} /></div>
        <h2 className="card-title" style={{ fontSize: 26 }}>No analysed matches yet</h2>
        <p className="muted" style={{ margin: 0 }}>Upload a match video to unlock overview, player, ranking and tactical insights.</p>
        <button className="btn-volt" style={{ height: 44 }} onClick={goUpload}>Upload a match</button>
      </motion.div>
    </div>
  );
}

function ErrorState({ message, retry, goAnalyses, goUpload }: { message: string; retry: () => void; goAnalyses: () => void; goUpload: () => void }) {
  const isConnection = /cannot reach|failed to fetch|network|connection lost/i.test(message);
  return (
    <div className="center-state">
      <motion.div className="card" style={{ maxWidth: 480, padding: 30 }} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}>
        <div className="eyebrow tb">{isConnection ? "Connection lost" : "Match Load Notice"}</div>
        <h2 className="card-title" style={{ fontSize: 28, margin: "8px 0 10px" }}>
          {isConnection ? "Can't reach the engine" : "Unable to load match"}
        </h2>
        <p className="muted" style={{ marginTop: 0 }}>{message}</p>
        {isConnection ? (
          <p className="muted" style={{ fontSize: 12.5 }}>Start the API with <code className="mono volt">python backend/run.py</code> and retry.</p>
        ) : (
          <p className="muted" style={{ fontSize: 12.5 }}>The match data may still be processing or was not found.</p>
        )}
        <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
          <button className="btn-volt" style={{ height: 42 }} onClick={retry}>Retry</button>
          <button className="btn-ghost" style={{ height: 42 }} onClick={goAnalyses}>My Analyses</button>
          <button className="btn-ghost" style={{ height: 42 }} onClick={goUpload}>Upload match</button>
        </div>
      </motion.div>
    </div>
  );
}
