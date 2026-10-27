import { motion } from "framer-motion";
import { useEffect, useState, type ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ViewProps } from "../App";
import { api } from "../api";
import { IconDownload } from "../components/Icons";
import { ease } from "../components/Motion";

const reveal = (Tag: "h2" | "h3" | "blockquote" | "table") =>
  function Revealed({ children }: { children?: ReactNode }) {
    const M = motion[Tag];
    return (
      <M initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-40px" }} transition={{ duration: 0.5, ease }}>
        {children}
      </M>
    );
  };

const components = { h2: reveal("h2"), h3: reveal("h3"), blockquote: reveal("blockquote"), table: reveal("table") };

export function Report({ matchId }: ViewProps) {
  const [md, setMd] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setMd(null);
    setErr(null);
    api.report(matchId).then(setMd, (e) => setErr((e as Error).message));
  }, [matchId]);

  const download = () => {
    if (!md) return;
    const url = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `${matchId}_scouting_report.md` });
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="toolbar">
        <span className="chip live">Executive scouting report</span>
        <span className="chip mono">{matchId}</span>
        <button className="btn-ghost" style={{ marginLeft: "auto" }} onClick={download} disabled={!md}><IconDownload size={16} /> Download .md</button>
      </div>
      <div className="card report">
        {err ? (
          <p className="tb">{err}</p>
        ) : !md ? (
          <div className="grid" style={{ gap: 14 }}>
            {[70, 40, 100, 90, 95, 60, 100, 80].map((w, i) => <div key={i} className="skeleton" style={{ height: i === 0 ? 40 : 16, width: `${w}%` }} />)}
          </div>
        ) : (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
            <Markdown remarkPlugins={[remarkGfm]} components={components}>{md}</Markdown>
          </motion.div>
        )}
      </div>
    </div>
  );
}
