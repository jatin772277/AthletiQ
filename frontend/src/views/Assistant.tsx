import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import type { ViewProps } from "../App";
import { api } from "../api";
import { IconSend, Logo } from "../components/Icons";
import { ease } from "../components/Motion";

interface Msg {
  id: number;
  role: "user" | "ai";
  text: string;
  sources?: string[];
  error?: boolean;
}

/** Reveals text progressively, like a live broadcast ticker. */
function Typewriter({ text, onTick }: { text: string; onTick: () => void }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (n >= text.length) return;
    const step = Math.max(2, Math.round(text.length / 120));
    const t = setTimeout(() => { setN((v) => Math.min(text.length, v + step)); onTick(); }, 12);
    return () => clearTimeout(t);
  }, [n, text, onTick]);
  return <Markdown>{text.slice(0, n)}</Markdown>;
}

export function Assistant({ matchId, data }: ViewProps) {
  const top = data.rankings[0]?.canonical_id;
  const fastest = [...data.players].sort((a, b) => b.collective_metrics.peak_speed_kmh - a.collective_metrics.peak_speed_kmh)[0]?.player_id;
  const suggestions = [
    top != null && `How did Player ${top} perform?`,
    "Analyze team possession and touches dominance",
    fastest != null && `Evaluate Player ${fastest}'s sprint profile`,
    "Give me a tactical summary of the match",
  ].filter(Boolean) as string[];

  const [msgs, setMsgs] = useState<Msg[]>([
    {
      id: 0,
      role: "ai",
      text: `**AthletiQ Tactical AI ready.** I'm grounded in the verified tracking data for **${matchId}**: ${data.summary.total_active_players} players, ${data.summary.total_frames} frames, ${data.summary.match_total_touches} touches. Ask about a player (e.g. *"Player 14"*), possession, work-rate or stitching buckets.`,
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const animated = useRef(new Set<number>([0]));

  // Block body on purpose: newer browsers return a Promise from scrollTo, which must not leak out as an effect cleanup.
  const scrollDown = useCallback(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight }); }, []);
  useEffect(() => { scrollDown(); }, [msgs, busy, scrollDown]);

  const ask = async (q: string) => {
    const query = q.trim();
    if (!query || busy) return;
    setInput("");
    setMsgs((m) => [...m, { id: Date.now(), role: "user", text: query }]);
    setBusy(true);
    try {
      const res = await api.ask(matchId, query);
      setMsgs((m) => [...m, { id: Date.now() + 1, role: "ai", text: res.answer, sources: res.sources_used }]);
    } catch (e) {
      setMsgs((m) => [...m, { id: Date.now() + 1, role: "ai", text: `Query failed: ${(e as Error).message}`, error: true }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card chat">
      <div className="chat-scroll" ref={scroller}>
        <AnimatePresence initial={false}>
          {msgs.map((m) => {
            const fresh = m.role === "ai" && !animated.current.has(m.id);
            if (fresh) animated.current.add(m.id);
            return (
              <motion.div key={m.id} className={`msg ${m.role}`} initial={{ opacity: 0, y: 14, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.35, ease }}>
                <div className={`avatar ${m.role === "ai" ? "ai" : "me"}`}>{m.role === "ai" ? <Logo size={18} /> : "You"}</div>
                <div className="bubble" style={m.error ? { borderColor: "rgba(255,93,125,.4)" } : undefined}>
                  {fresh ? <Typewriter text={m.text} onTick={scrollDown} /> : <Markdown>{m.text}</Markdown>}
                  {m.sources && m.sources.length > 0 && (
                    <div className="src">{m.sources.map((s) => <span key={s}>{s}</span>)}</div>
                  )}
                </div>
              </motion.div>
            );
          })}
          {busy && (
            <motion.div key="typing" className="msg ai" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="avatar ai"><Logo size={18} /></div>
              <div className="bubble"><span className="typing"><i /><i /><i /></span></div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <div className="composer">
        <div className="suggestions">
          {suggestions.map((s, i) => (
            <motion.button key={s} onClick={() => ask(s)} disabled={busy} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 + i * 0.06 }}>
              {s}
            </motion.button>
          ))}
        </div>
        <form className="input-row" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about work-rate, a player, possession…" aria-label="Ask the tactical AI" />
          <motion.button type="submit" className="btn-volt" disabled={busy || !input.trim()} whileTap={{ scale: 0.94 }}>
            <IconSend size={16} /> Ask
          </motion.button>
        </form>
      </div>
    </div>
  );
}
