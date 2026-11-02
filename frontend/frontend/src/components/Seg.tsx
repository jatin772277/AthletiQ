import { motion } from "framer-motion";
import { useId } from "react";

/** Segmented control with a sliding highlight. */
export function Seg<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  const id = useId();
  return (
    <div className="seg" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={value === o.value} className={value === o.value ? "on" : ""} onClick={() => onChange(o.value)}>
          {value === o.value && <motion.span layoutId={`seg-${id}`} className="seg-pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}
