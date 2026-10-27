import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { IconFilm, IconUpload } from "../components/Icons";
import { ease } from "../components/Motion";
import type { Job, JobStatus } from "../types";

// Mirrors the backend's progress checkpoints (pipeline_runner.PHASES).
const STEPS: { at: number; label: string }[] = [
  { at: 2, label: "Tracking" },
  { at: 62, label: "Stitching" },
  { at: 70, label: "Calibration" },
  { at: 78, label: "Spatial" },
  { at: 82, label: "Possession" },
  { at: 86, label: "Network" },
  { at: 90, label: "DSA" },
  { at: 94, label: "ML score" },
  { at: 97, label: "Export" },
];

const STATUS: Record<JobStatus, { label: string; cls: string }> = {
  queued: { label: "Queued", cls: "" },
  running: { label: "Analysing", cls: "live" },
  completed: { label: "Ready", cls: "live" },
  failed: { label: "Failed", cls: "team_b" },
  cancelled: { label: "Cancelled", cls: "" },
};

const active = (j: Job) => j.status === "queued" || j.status === "running";
const ago = (iso: string) => {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
};
const elapsed = (j: Job) => {
  if (!j.started_at) return null;
  const end = j.finished_at ? new Date(j.finished_at).getTime() : Date.now();
  const s = Math.max(0, Math.round((end - new Date(j.started_at).getTime()) / 1000));
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
};

export function Analyses({ onOpen, goUpload, onCompleted }: { onOpen: (id: string) => void; goUpload: () => void; onCompleted: () => void }) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openLog, setOpenLog] = useState<string | null>(null);
  const prev = useRef<Map<string, JobStatus>>(new Map());

  const load = useCallback(async () => {
    try {
      const list = await api.jobs();
      // Tell the shell when something finished so the match picker refreshes.
      if (list.some((j) => j.status === "completed" && prev.current.get(j.id) && prev.current.get(j.id) !== "completed")) onCompleted();
      prev.current = new Map(list.map((j) => [j.id, j.status]));
      setJobs(list);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [onCompleted]);

  useEffect(() => { load(); }, [load]);
  const anyActive = jobs?.some(active) ?? false;
  useEffect(() => {
    const t = setInterval(load, anyActive ? 2000 : 15000);
    return () => clearInterval(t);
  }, [load, anyActive]);

  const act = async (fn: () => Promise<unknown>) => {
    try { await fn(); } catch (e) { setError((e as Error).message); }
    load();
  };

  if (!jobs) return <div className="center-state"><div className="loader" /></div>;

  if (jobs.length === 0)
    return (
      <motion.div className="card upload-done" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
        <div className="done-badge" style={{ background: "var(--surface-2)", color: "var(--volt)" }}><IconFilm size={30} /></div>
        <h2 className="card-title" style={{ fontSize: 26 }}>No analyses yet</h2>
        <p className="muted" style={{ margin: 0 }}>Upload your first match video and AthletiQ will run the complete pipeline on it.</p>
        <button className="btn-volt" style={{ height: 44 }} onClick={goUpload}><IconUpload size={16} /> Upload a match</button>
      </motion.div>
    );

  return (
    <div>
      <div className="toolbar">
        <span className="chip">{jobs.length} analyses</span>
        {anyActive && <span className="chip live"><span className="status-dot" />Live updates</span>}
        <button className="btn-volt" style={{ height: 40, marginLeft: "auto" }} onClick={goUpload}><IconUpload size={16} /> New upload</button>
      </div>
      {error && <div className="auth-error" style={{ marginBottom: 12 }}>{error}</div>}
      <motion.div layout className="grid" style={{ gap: 14 }}>
        <AnimatePresence initial={false}>
          {jobs.map((j, i) => {
            const st = STATUS[j.status];
            return (
              <motion.div key={j.id} layout className={`card job ${j.status}`} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: -30 }}
                transition={{ duration: 0.4, ease, delay: Math.min(i, 8) * 0.04 }}>
                <div className="job-head">
                  <div className="job-icon"><IconFilm /></div>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="job-title">{j.title}</div>
                    <div className="muted job-meta">
                      {j.original_filename} · {j.video.width}×{j.video.height} · {Math.round(j.video.fps)} fps ·
                      analysing {(j.max_frames / j.video.fps).toFixed(1)}s ({j.max_frames} frames) · uploaded {ago(j.created_at)}
                    </div>
                  </div>
                  <span className={`chip ${st.cls}`}>{j.status === "running" && <span className="status-dot" />}{st.label}</span>
                </div>

                {j.status !== "cancelled" && (
                  <>
                    <div className="job-progress-row">
                      <div className="progress">
                        <motion.div className={`progress-fill ${j.status}`} initial={false} animate={{ width: `${j.status === "failed" ? Math.max(j.progress, 4) : j.progress}%` }} transition={{ duration: 0.8, ease }} />
                      </div>
                      <span className="mono job-pct">{Math.round(j.progress)}%</span>
                    </div>
                    <div className="stepper">
                      {STEPS.map((s, si) => {
                        const next = STEPS[si + 1]?.at ?? 100;
                        const state = j.status === "completed" || j.progress >= next ? "done" : j.progress >= s.at ? (j.status === "failed" ? "err" : "now") : "";
                        return <span key={s.label} className={`step ${state}`}><i />{s.label}</span>;
                      })}
                    </div>
                  </>
                )}

                <div className="job-foot">
                  <span className={j.status === "failed" ? "tb" : "muted"} style={{ fontSize: 12.5 }}>
                    {j.status === "failed" ? j.error : j.phase}
                    {elapsed(j) && <span className="muted"> · {elapsed(j)}</span>}
                  </span>
                  <div className="job-actions">
                    {j.log_tail.length > 0 && <button className="btn-ghost sm" onClick={() => setOpenLog(openLog === j.id ? null : j.id)}>{openLog === j.id ? "Hide log" : "Log"}</button>}
                    {active(j) && <button className="btn-ghost sm" onClick={() => act(() => api.cancelJob(j.id))}>Cancel</button>}
                    {!active(j) && <button className="btn-ghost sm danger" onClick={() => confirm(`Delete "${j.title}" and its video?`) && act(() => api.deleteJob(j.id))}>Delete</button>}
                    {j.status === "completed" && <motion.button className="btn-volt sm" whileTap={{ scale: 0.95 }} onClick={() => onOpen(j.id)}>View analysis →</motion.button>}
                  </div>
                </div>
                <AnimatePresence>
                  {openLog === j.id && (
                    <motion.pre className="job-log" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}>
                      {j.log_tail.join("\n")}
                    </motion.pre>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
