// SourcePanel.tsx - the passages the answer was built from, with a link to the original page.

import type { Source } from "@/lib/types";

interface Props {
  sources: Source[];
  selected: number | null;
  onSelect: (n: number | null) => void;
}

export default function SourcePanel({ sources, selected, onSelect }: Props) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-3">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-900/70">Sources</h3>
      <ul className="space-y-2">
        {sources.map((s) => {
          const open = selected === s.number;
          return (
            <li
              key={s.chunk_id}
              id={`source-${s.number}`}
              className={`rounded-lg border px-3 py-2 text-sm transition-colors ${
                open ? "border-emerald-600 bg-emerald-50" : "border-emerald-100 bg-white"
              }`}
            >
              <button
                type="button"
                onClick={() => onSelect(open ? null : s.number)}
                className="flex w-full items-start gap-2 text-left"
                aria-expanded={open}
              >
                <span className="mt-0.5 rounded bg-emerald-700 px-1.5 text-xs font-semibold text-white">{s.number}</span>
                <span className="flex-1 font-medium text-emerald-950">{s.title}</span>
                <span className={`text-xs ${s.cited ? "text-emerald-700" : "text-slate-400"}`}>
                  {s.cited ? "cited" : "retrieved, not cited"}
                </span>
              </button>
              {open && (
                <div className="mt-2 space-y-2 border-t border-emerald-100 pt-2 text-slate-700">
                  <p>{s.excerpt}</p>
                  <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-xs font-medium text-emerald-700 underline">
                    Open the original page
                  </a>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
