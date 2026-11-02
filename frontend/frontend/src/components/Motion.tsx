import { animate, motion, useInView, useMotionValue, useTransform, type Variants } from "framer-motion";
import { useEffect, useRef, type ReactNode } from "react";

export const ease = [0.22, 1, 0.36, 1] as const;

export const stagger: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.05 } },
};

export const rise: Variants = {
  hidden: { opacity: 0, y: 22, filter: "blur(6px)" },
  show: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.6, ease } },
};

/** Container that staggers its <Rise> children in once on mount. */
export function Stagger({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div className={className} variants={stagger} initial="hidden" animate="show">
      {children}
    </motion.div>
  );
}

export function Rise({ children, className, style }: { children: ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <motion.div className={className} style={style} variants={rise}>
      {children}
    </motion.div>
  );
}

/** Animated number that counts up when scrolled into view. */
export function CountUp({ value, decimals = 0, duration = 1.6, suffix = "" }: { value: number; decimals?: number; duration?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) =>
    v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix,
  );
  useEffect(() => {
    if (!inView) return;
    const controls = animate(mv, value, { duration, ease });
    return () => controls.stop();
  }, [inView, value, duration, mv]);
  return <motion.span ref={ref}>{text}</motion.span>;
}

/** Bar that grows from 0 to `pct` percent of its track. */
export function GrowBar({ pct, color, delay = 0, vertical = false, className, style }: { pct: number; color: string; delay?: number; vertical?: boolean; className?: string; style?: React.CSSProperties }) {
  const dim = vertical ? "height" : "width";
  return (
    <motion.div
      className={className}
      style={{ background: color, ...(vertical ? { width: "100%" } : { height: "100%" }), ...style }}
      initial={{ [dim]: "0%" }}
      animate={{ [dim]: `${Math.max(0, Math.min(100, pct))}%` }}
      transition={{ duration: 1.1, ease, delay }}
    />
  );
}

/** Circular progress gauge. */
export function RingGauge({ value, max = 100, color = "var(--volt)", size = 108, stroke = 9, label, display }: { value: number; max?: number; color?: string; size?: number; stroke?: number; label: string; display?: ReactNode }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value / max));
  return (
    <div className="gauge card">
      <div style={{ position: "relative", width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
          <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--surface-2)" strokeWidth={stroke} fill="none" />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            stroke={color}
            strokeWidth={stroke}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={c}
            initial={{ strokeDashoffset: c }}
            animate={{ strokeDashoffset: c * (1 - pct) }}
            transition={{ duration: 1.4, ease, delay: 0.15 }}
            style={{ filter: `drop-shadow(0 0 6px ${color})` }}
          />
        </svg>
        <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>
          <span className="g-val">{display ?? <CountUp value={value} decimals={1} />}</span>
        </div>
      </div>
      <div className="g-label">{label}</div>
    </div>
  );
}
