import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { ease } from "./Motion";

export const PITCH_W = 105;
export const PITCH_H = 68;
const PAD = 4;

const line = { stroke: "rgba(255,255,255,0.32)", strokeWidth: 0.35, fill: "none" };

function Draw({ d, delay = 0 }: { d: string; delay?: number }) {
  return (
    <motion.path
      d={d}
      {...line}
      initial={{ pathLength: 0, opacity: 0 }}
      animate={{ pathLength: 1, opacity: 1 }}
      transition={{ duration: 1.6, ease, delay }}
    />
  );
}

/** FIFA 105x68 m pitch in metre coordinates; children are drawn in the same space. */
export function Pitch({ children, scan = false, className, defs }: { children?: ReactNode; scan?: boolean; className?: string; defs?: ReactNode }) {
  const W = PITCH_W, H = PITCH_H;
  const box = (x: number, w: number, h: number) => `M${x} ${(H - h) / 2}h${w}v${h}h${-w}`;
  const arc = (cx: number, dir: 1 | -1) => {
    // Penalty arc: radius 9.15 around the spot, clipped by the 16.5 m box edge.
    const dx = 16.5 - 11;
    const dy = Math.sqrt(9.15 ** 2 - dx ** 2);
    return `M${cx + dir * 16.5} ${H / 2 - dy}A9.15 9.15 0 0 ${dir === 1 ? 1 : 0} ${cx + dir * 16.5} ${H / 2 + dy}`;
  };
  const stripes = 12;
  return (
    <svg className={className} viewBox={`${-PAD} ${-PAD} ${W + PAD * 2} ${H + PAD * 2}`} width="100%" style={{ display: "block", overflow: "visible" }}>
      <defs>
        <linearGradient id="turf" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#0f2a1c" />
          <stop offset="1" stopColor="#0a1d14" />
        </linearGradient>
        <radialGradient id="flood" cx="0.5" cy="0.45" r="0.7">
          <stop offset="0" stopColor="rgba(198,255,61,0.10)" />
          <stop offset="1" stopColor="rgba(0,0,0,0)" />
        </radialGradient>
        <linearGradient id="scan" x1="0" x2="1">
          <stop offset="0" stopColor="rgba(198,255,61,0)" />
          <stop offset="0.85" stopColor="rgba(198,255,61,0.16)" />
          <stop offset="1" stopColor="rgba(198,255,61,0.55)" />
        </linearGradient>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="0.9" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <clipPath id="pitch-clip"><rect x="0" y="0" width={W} height={H} rx="1" /></clipPath>
        {defs}
      </defs>
      <rect x={-PAD} y={-PAD} width={W + PAD * 2} height={H + PAD * 2} rx="3" fill="url(#turf)" />
      {Array.from({ length: stripes }).map((_, i) =>
        i % 2 ? <rect key={i} x={(W / stripes) * i} y={0} width={W / stripes} height={H} fill="rgba(255,255,255,0.025)" /> : null,
      )}
      <rect x={-PAD} y={-PAD} width={W + PAD * 2} height={H + PAD * 2} fill="url(#flood)" />
      <Draw d={`M0 0H${W}V${H}H0Z`} />
      <Draw d={`M${W / 2} 0V${H}`} delay={0.15} />
      <motion.circle cx={W / 2} cy={H / 2} r={9.15} {...line} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6, ease, delay: 0.25 }} />
      <circle cx={W / 2} cy={H / 2} r={0.45} fill="rgba(255,255,255,0.5)" />
      <Draw d={box(0, 16.5, 40.32)} delay={0.3} />
      <Draw d={`M${W} ${(H - 40.32) / 2}h-16.5v40.32h16.5`} delay={0.3} />
      <Draw d={box(0, 5.5, 18.32)} delay={0.45} />
      <Draw d={`M${W} ${(H - 18.32) / 2}h-5.5v18.32h5.5`} delay={0.45} />
      <Draw d={arc(0, 1)} delay={0.55} />
      <Draw d={arc(W, -1)} delay={0.55} />
      <rect x={-1.6} y={H / 2 - 3.66} width={1.6} height={7.32} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth={0.3} />
      <rect x={W} y={H / 2 - 3.66} width={1.6} height={7.32} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth={0.3} />
      {scan && (
        <g clipPath="url(#pitch-clip)">
          <motion.rect
            y={0}
            width={22}
            height={H}
            fill="url(#scan)"
            initial={{ x: -22 }}
            animate={{ x: W }}
            transition={{ duration: 4.5, repeat: Infinity, ease: "linear", repeatDelay: 1.2 }}
          />
        </g>
      )}
      {children}
    </svg>
  );
}
