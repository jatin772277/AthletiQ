import { AnimatePresence, motion, useAnimationControls } from "framer-motion";
import { useState, type FormEvent } from "react";
import { api, session } from "../api";
import { IconBall, IconBolt, IconPitch, IconShield, Logo } from "../components/Icons";
import { ease } from "../components/Motion";
import { Pitch } from "../components/Pitch";
import { Seg } from "../components/Seg";
import type { User } from "../types";

// Decorative 4-3-3 vs 4-4-2 shapes in pitch metres.
const HOME = [[8, 34], [22, 12], [20, 27], [20, 41], [22, 56], [36, 20], [34, 34], [36, 48], [48, 14], [50, 34], [48, 54]];
const AWAY = [[97, 34], [83, 12], [85, 27], [85, 41], [83, 56], [68, 14], [70, 28], [70, 40], [68, 54], [58, 28], [58, 42]];
const PASSES = [[34, 34], [36, 48], [48, 54], [50, 34], [36, 20], [48, 14], [50, 34], [34, 34]];

function DemoPitch() {
  return (
    <Pitch scan>
      <motion.circle r={1} fill="#fff" filter="url(#glow)"
        animate={{ cx: PASSES.map((p) => p[0]), cy: PASSES.map((p) => p[1]) }}
        transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }} />
      {[...HOME.map((p) => ({ p, c: "#36d7ff" })), ...AWAY.map((p) => ({ p, c: "#ff5d7d" }))].map(({ p: [x, y], c }, i) => (
        <motion.g key={i} initial={{ opacity: 0 }} animate={{ opacity: 1, x: [0, (i % 3) - 1, 0], y: [0, ((i * 7) % 5) / 2 - 1, 0] }}
          transition={{ opacity: { delay: 0.8 + i * 0.04 }, x: { duration: 5 + (i % 4), repeat: Infinity }, y: { duration: 6 + (i % 3), repeat: Infinity } }}>
          <circle cx={x} cy={y} r={2.4} fill={c} opacity={0.18} />
          <circle cx={x} cy={y} r={1.25} fill={c} filter="url(#glow)" />
        </motion.g>
      ))}
    </Pitch>
  );
}

const FEATURES = [
  { icon: IconBolt, title: "Upload any match", text: "Drop in footage and the full CV pipeline runs for you." },
  { icon: IconPitch, title: "Tactical intelligence", text: "Tracking, calibration, possession and interaction networks." },
  { icon: IconBall, title: "Player kinematics", text: "Distance, sprints, speed zones and touches for every player." },
  { icon: IconShield, title: "Private by default", text: "Your videos and analyses are visible only to your account." },
];

export function AuthScreen({ onAuthed }: { onAuthed: (u: User) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const shake = useAnimationControls();

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (mode === "register" && password.length < 8) {
      setError("Password must be at least 8 characters");
      shake.start({ x: [0, -10, 10, -6, 6, 0], transition: { duration: 0.4 } });
      return;
    }
    setBusy(true);
    try {
      const res = mode === "login" ? await api.login(email, password) : await api.register(name, email, password);
      session.set(res.access_token);
      onAuthed(res.user);
    } catch (err) {
      setError((err as Error).message);
      shake.start({ x: [0, -10, 10, -6, 6, 0], transition: { duration: 0.4 } });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <div className="ambient"><div className="glow a" /><div className="glow b" /><div className="glow c" /><div className="grid" /></div>

      <section className="auth-hero">
        <motion.div className="brand" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
          <motion.div className="brand-mark" initial={{ rotate: -90, scale: 0 }} animate={{ rotate: 0, scale: 1 }} transition={{ type: "spring", stiffness: 220, damping: 14 }}>
            <Logo />
          </motion.div>
          <div>
            <div className="brand-name">Athleti<em>Q</em></div>
            <div className="brand-sub">Performance Intelligence</div>
          </div>
        </motion.div>
        <motion.h1 className="auth-headline" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease, delay: 0.1 }}>
          Turn match footage into <span className="volt">elite performance</span> intelligence.
        </motion.h1>
        <motion.div className="card auth-pitch" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8, ease, delay: 0.2 }}>
          <DemoPitch />
        </motion.div>
        <div className="auth-features">
          {FEATURES.map((f, i) => (
            <motion.div key={f.title} className="auth-feature" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 + i * 0.08, ease }}>
              <div className="icon"><f.icon /></div>
              <div><div className="t">{f.title}</div><div className="d">{f.text}</div></div>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="auth-panel">
        <motion.div style={{ width: "min(420px, 100%)" }} initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease, delay: 0.15 }}>
        <motion.form className="card auth-card" onSubmit={submit} animate={shake}>
          <div className="eyebrow">{mode === "login" ? "Welcome back" : "Get started"}</div>
          <h2 className="auth-title">{mode === "login" ? "Sign in" : "Create account"}</h2>
          <Seg value={mode} onChange={(m) => { setMode(m); setError(null); }} options={[{ value: "login", label: "Sign in" }, { value: "register", label: "Create account" }]} />

          <div className="fields">
            <AnimatePresence initial={false}>
              {mode === "register" && (
                <motion.label key="name" className="field" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.25 }}>
                  <span>Full name</span>
                  <input value={name} onChange={(e) => setName(e.target.value)} required autoComplete="name" placeholder="Alex Morgan" />
                </motion.label>
              )}
            </AnimatePresence>
            <label className="field">
              <span>Email</span>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" placeholder="coach@club.com" />
            </label>
            <label className="field">
              <span>Password</span>
              <div className="pw">
                <input type={show ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)} required
                  autoComplete={mode === "login" ? "current-password" : "new-password"} placeholder={mode === "register" ? "At least 8 characters" : "••••••••"} />
                <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}>{show ? "Hide" : "Show"}</button>
              </div>
            </label>
          </div>

          <AnimatePresence>
            {error && (
              <motion.div className="auth-error" role="alert" initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>{error}</motion.div>
            )}
          </AnimatePresence>

          <motion.button type="submit" className="btn-volt auth-submit" disabled={busy} whileTap={{ scale: 0.97 }}>
            {busy ? <span className="spinner" /> : mode === "login" ? "Sign in" : "Create account"}
          </motion.button>
          <div className="muted" style={{ fontSize: 12.5, textAlign: "center" }}>
            {mode === "login" ? "New to AthletiQ? " : "Already have an account? "}
            <button type="button" className="link" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}>
              {mode === "login" ? "Create an account" : "Sign in"}
            </button>
          </div>
        </motion.form>
        </motion.div>
      </section>
    </div>
  );
}
