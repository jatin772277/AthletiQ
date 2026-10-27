import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState, type DragEvent } from "react";
import { api } from "../api";
import { IconCheck, IconClose, IconFilm, IconUpload } from "../components/Icons";
import { ease } from "../components/Motion";
import type { Job } from "../types";

const ACCEPT = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];
const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtSize = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);

export function Upload({ onQueued, goAnalyses }: { onQueued: (job: Job) => void; goAnalyses: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [title, setTitle] = useState("");
  const [start, setStart] = useState(0);
  const [length, setLength] = useState(25);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [queued, setQueued] = useState<Job | null>(null);
  // Browsers can't decode every codec (e.g. MPEG-4 Part 2, many AVI/MKV files); the server still can.
  const [noPreview, setNoPreview] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  // Some unsupported codecs never fire loadedmetadata or error, so give up on the preview after a few seconds.
  useEffect(() => {
    if (!url || duration) return;
    const t = setTimeout(() => setNoPreview(true), 4000);
    return () => clearTimeout(t);
  }, [url, duration]);

  const pick = (f: File | undefined) => {
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPT.includes(ext)) {
      setError(`Unsupported file type. Use ${ACCEPT.join(", ")}`);
      return;
    }
    setError(null);
    setQueued(null);
    setFile(f);
    setTitle(f.name.replace(/\.[^.]+$/, ""));
    setUrl(URL.createObjectURL(f));
    setStart(0);
    setDuration(0);
    setLength(25);
    setNoPreview(false);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    pick(e.dataTransfer.files[0]);
  };

  // Without a preview the length is unknown; the server clamps the window to the real video length.
  const maxLen = noPreview ? 600 : Math.max(1, Math.floor(duration - start));
  const clip = Math.min(length, maxLen);
  const ready = noPreview || duration > 0;

  const submit = async () => {
    if (!file) return;
    setError(null);
    setProgress(0);
    abort.current = new AbortController();
    try {
      const job = await api.upload(file, { title, startSeconds: start, durationSeconds: clip, onProgress: setProgress, signal: abort.current.signal });
      setQueued(job);
      onQueued(job);
      setFile(null);
      setUrl(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setProgress(null);
    }
  };

  const seek = (t: number) => { if (videoRef.current) videoRef.current.currentTime = t; };

  return (
    <div className="grid upload-layout">
      <AnimatePresence mode="wait">
        {queued ? (
          <motion.div key="done" className="card upload-done" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
            <motion.div className="done-badge" initial={{ scale: 0, rotate: -45 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: "spring", stiffness: 300, damping: 14, delay: 0.1 }}>
              <IconCheck size={34} />
            </motion.div>
            <h2 className="card-title" style={{ fontSize: 28 }}>Match queued for analysis</h2>
            <p className="muted" style={{ maxWidth: 460, textAlign: "center" }}>
              <b style={{ color: "var(--text)" }}>{queued.title}</b> is in the pipeline: tracking, calibration, possession, network and ML scoring.
              You can leave this page; progress is saved to your account.
            </p>
            <div style={{ display: "flex", gap: 10 }}>
              <button className="btn-volt" style={{ height: 44 }} onClick={goAnalyses}>Track progress →</button>
              <button className="btn-ghost" onClick={() => setQueued(null)}>Upload another</button>
            </div>
          </motion.div>
        ) : !file ? (
          <motion.div key="drop" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ ease, duration: 0.4 }}>
            <div className={`card dropzone ${drag ? "over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
              onClick={() => input.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && input.current?.click()}>
              <motion.div className="dz-icon" animate={{ y: drag ? -10 : [0, -8, 0] }} transition={drag ? { type: "spring" } : { duration: 2.4, repeat: Infinity, ease: "easeInOut" }}>
                <IconUpload size={34} />
              </motion.div>
              <h2 className="card-title" style={{ fontSize: 26 }}>{drag ? "Release to upload" : "Drop your match video"}</h2>
              <p className="muted" style={{ margin: 0 }}>or click to browse · MP4, MOV, AVI, MKV, WEBM</p>
              <div className="dz-steps">
                {["Upload", "Track players & ball", "Calibrate pitch", "Analyse", "Explore insights"].map((s, i) => (
                  <motion.span key={s} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + i * 0.07 }}>
                    <i>{i + 1}</i>{s}
                  </motion.span>
                ))}
              </div>
              <input ref={input} type="file" accept={ACCEPT.join(",") + ",video/*"} hidden onChange={(e) => pick(e.target.files?.[0])} />
            </div>
            {error && <div className="auth-error" style={{ marginTop: 12 }}>{error}</div>}
          </motion.div>
        ) : (
          <motion.div key="setup" className="grid g-upload" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ ease, duration: 0.4 }}>
            <div className="card" style={{ padding: 12 }}>
              {noPreview ? (
                <div className="no-preview">
                  <IconFilm size={34} />
                  <div style={{ fontWeight: 600 }}>Preview not available for this video format</div>
                  <div className="muted" style={{ fontSize: 12.5 }}>Your browser can't play this codec, but AthletiQ can still analyse it. Set the start time and length by hand.</div>
                </div>
              ) : (
              <video ref={videoRef} src={url ?? undefined} controls muted className="preview" onError={() => setNoPreview(true)}
                onLoadedMetadata={(e) => {
                  const d = e.currentTarget.duration;
                  if (!Number.isFinite(d) || d <= 0) return setNoPreview(true);
                  setDuration(d);
                  setLength(Math.min(25, Math.floor(d) || 25));
                }} />
              )}
              {!noPreview && (<>
              <div className="timeline-pick">
                <div className="window" style={{ left: `${duration ? (start / duration) * 100 : 0}%`, width: `${duration ? (clip / duration) * 100 : 0}%` }} />
              </div>
              <div className="legend" style={{ justifyContent: "space-between", padding: "0 4px" }}>
                <span className="mono">0:00</span>
                <span className="volt">Analysing {fmtTime(start)} → {fmtTime(start + clip)}</span>
                <span className="mono">{fmtTime(duration)}</span>
              </div>
              </>)}
            </div>

            <div className="card upload-form">
              <div className="card-head">
                <h3 className="card-title">Analysis setup</h3>
                <button className="drawer-close" style={{ position: "static" }} onClick={() => { setFile(null); setUrl(null); }} aria-label="Remove video" disabled={progress !== null}><IconClose /></button>
              </div>
              <div className="file-chip"><span className="mono">{file.name}</span><span className="muted">{fmtSize(file.size)}</span></div>

              <label className="field">
                <span>Match title</span>
                <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} placeholder="e.g. U19 League — Week 6" />
              </label>

              {noPreview ? (
                <div className="grid g-2" style={{ gap: 12 }}>
                  <label className="field">
                    <span>Start at (seconds)</span>
                    <input type="number" min={0} step={1} value={start} onChange={(e) => setStart(Math.max(0, Number(e.target.value) || 0))} />
                  </label>
                  <label className="field">
                    <span>Length (seconds)</span>
                    <input type="number" min={1} max={maxLen} step={1} value={clip} onChange={(e) => setLength(Math.max(1, Number(e.target.value) || 1))} />
                  </label>
                </div>
              ) : (<>
              <label className="field">
                <span>Start at <b className="volt">{fmtTime(start)}</b></span>
                <input type="range" min={0} max={Math.max(0, Math.floor(duration) - 1)} step={1} value={start}
                  onChange={(e) => { const v = Number(e.target.value); setStart(v); seek(v); }} />
              </label>
              <label className="field">
                <span>Analyse <b className="volt">{clip}s</b> of play</span>
                <input type="range" min={1} max={Math.min(maxLen, 300)} step={1} value={clip} onChange={(e) => setLength(Number(e.target.value))} />
              </label>
              </>)}
              <p className="muted" style={{ fontSize: 12, margin: 0 }}>
                Longer clips give richer stats but take longer to process: tracking runs on every frame. Skip kick-off ceremonies and replays for the best results.
              </p>

              {progress !== null ? (
                <div>
                  <div className="legend" style={{ justifyContent: "space-between", marginBottom: 8 }}>
                    <span>Uploading…</span><span className="mono volt">{Math.round(progress * 100)}%</span>
                  </div>
                  <div className="progress"><motion.div className="progress-fill running" animate={{ width: `${progress * 100}%` }} transition={{ ease: "linear" }} /></div>
                  <button className="btn-ghost" style={{ marginTop: 12 }} onClick={() => abort.current?.abort()}>Cancel upload</button>
                </div>
              ) : (
                <motion.button className="btn-volt" style={{ height: 48, justifyContent: "center" }} onClick={submit} disabled={!ready} whileTap={{ scale: 0.97 }}>
                  <IconUpload size={18} /> Upload & run full pipeline
                </motion.button>
              )}
              {error && <div className="auth-error">{error}</div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
