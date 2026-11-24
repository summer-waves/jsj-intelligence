"use client";

// ChatMessage.tsx - one question and its answer: text with clickable citations, notices, sources, timings.

import { useState } from "react";
import type { AskResponse } from "@/lib/types";
import CitationChip from "./CitationChip";
import SafetyBanner from "./SafetyBanner";
import SourcePanel from "./SourcePanel";

export interface Turn {
  id: number;
  question: string;
  response?: AskResponse;
  error?: string;
}

// Matches [2]  [1, 3]  [1;2]
const CITATION_SPLIT = /(\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\])/g;

function renderAnswer(
  r: AskResponse,
  selected: number | null,
  onSelect: (n: number) => void,
): React.ReactNode[] {
  const valid = new Set(r.sources.map((s) => s.number));
  return r.answer.split(CITATION_SPLIT).map((part, i) => {
    if (!part.startsWith("[")) return <span key={i}>{part}</span>;
    const numbers = (part.match(/\d+/g) ?? []).map(Number);
    return (
      <span key={i}>
        {numbers.map((n) => (
          <CitationChip key={n} number={n} valid={valid.has(n)} active={selected === n} onSelect={onSelect} />
        ))}
      </span>
    );
  });
}

export default function ChatMessage({ turn }: { turn: Turn }) {
  const [selected, setSelected] = useState<number | null>(null);
  const r = turn.response;

  function selectFromChip(n: number) {
    setSelected(n);
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  return (
    <div className="space-y-3">
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-emerald-700 px-4 py-2 text-white">{turn.question}</div>

      <div className="max-w-[95%] space-y-3 rounded-2xl rounded-bl-sm border border-emerald-100 bg-white/90 px-4 py-3 shadow-sm">
        {!r && !turn.error && <p className="animate-pulse text-slate-500">Searching the library...</p>}
        {turn.error && <SafetyBanner tone="health">{turn.error}</SafetyBanner>}

        {r && (
          <>
            {r.notices.emergency && <SafetyBanner tone="emergency">{r.notices.emergency}</SafetyBanner>}
            {r.notices.health && <SafetyBanner tone="health">{r.notices.health}</SafetyBanner>}
            {r.withheld_reason && (
              <SafetyBanner tone="info">A safety check replaced the model's answer ({r.withheld_reason}).</SafetyBanner>
            )}

            <p className={`leading-relaxed ${r.refused ? "text-slate-600" : "text-emerald-950"}`}>
              {renderAnswer(r, selected, selectFromChip)}
            </p>

            {r.uncited_sentences.length > 0 && (
              <p className="text-xs text-amber-800">
                {r.uncited_sentences.length} sentence{r.uncited_sentences.length > 1 ? "s have" : " has"} no citation: treat{" "}
                {r.uncited_sentences.length > 1 ? "them" : "it"} with extra care.
              </p>
            )}

            <SourcePanel sources={r.sources} selected={selected} onSelect={setSelected} />

            <p className="text-xs text-slate-500">{r.notices.safety}</p>
            <p className="text-[11px] text-slate-400">
              {r.model} · {r.retrieval} search · {r.library_mode} library · {(r.timings.total_ms / 1000).toFixed(1)} s
            </p>
          </>
        )}
      </div>
    </div>
  );
}
