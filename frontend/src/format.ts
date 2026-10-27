import type { TeamId } from "./types";

export const teamName = (t: TeamId) => (t === "team_a" ? "Team A" : "Team B");
export const teamColor = (t: TeamId) => (t === "team_a" ? "var(--team-a)" : "var(--team-b)");
export const teamHex = (t: TeamId) => (t === "team_a" ? "#36d7ff" : "#ff5d7d");

export const num = (v: number, d = 0) =>
  v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

export const ZONES = [
  { key: "walking", label: "Walking", color: "var(--walk)" },
  { key: "jogging", label: "Jogging", color: "var(--jog)" },
  { key: "running", label: "Running", color: "var(--run)" },
  { key: "sprinting", label: "Sprinting", color: "var(--sprint)" },
] as const;

/** Short role tag from the pipeline's long tactical profile string. */
export const shortRole = (profile: string) => profile.split("/")[0].trim();
