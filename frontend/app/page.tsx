"use client";

// page.tsx - the chat page: question box, list of answered questions, a small status line.

import { useEffect, useRef, useState } from "react";
import ChatMessage, { type Turn } from "@/components/ChatMessage";
import { askQuestion, getHealth } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

const EXAMPLES = [
  "Can yoga help with stress?",
  "What does the NIH say about massage therapy?",
  "Is yoga safe?",
  "What is the capital of France?",
];

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null | undefined>(undefined);
  const bottomRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    getHealth().then(setHealth);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    const id = nextId.current++;
    setInput("");
    setBusy(true);
    setTurns((t) => [...t, { id, question }]);
    try {
      const response = await askQuestion(question);
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, response } : x)));
    } catch (e) {
      const error = e instanceof Error ? e.message : "Something went wrong.";
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error } : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#F3F8F6] to-[#DCEBE6] text-[#1F3A33]">
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-4 py-8">
        <header className="mb-6">
          <h1 className="text-3xl font-bold tracking-tight">JSJ Intelligence</h1>
          <p className="mt-1 text-sm text-emerald-900/80">
            Ask a question. Every answer comes only from the library and cites its sources.
          </p>
          <p className="mt-2 text-xs text-slate-500">
            {health === undefined && "Checking the API..."}
            {health === null && "The API is not reachable. Start the backend, then reload this page."}
            {health && `API online · ${health.library_mode} library · ${health.chunks} passages · ${health.model}`}
          </p>
        </header>

        <section className="flex-1 space-y-6">
          {turns.length === 0 && (
            <div className="rounded-2xl border border-emerald-100 bg-white/80 p-4">
              <p className="mb-3 text-sm text-slate-600">Try one of these:</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => send(q)}
                    className="rounded-full border border-emerald-300 bg-white px-3 py-1 text-sm text-emerald-800 hover:bg-emerald-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t) => (
            <ChatMessage key={t.id} turn={t} />
          ))}
          <div ref={bottomRef} />
        </section>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="sticky bottom-0 mt-6 flex gap-2 rounded-2xl border border-emerald-200 bg-white p-2 shadow-md"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={500}
            placeholder="Ask a question about relaxation, stress, yoga or massage"
            className="flex-1 rounded-xl px-3 py-2 text-[#1F3A33] outline-none placeholder:text-slate-400"
            aria-label="Your question"
          />
          <button
            type="submit"
            disabled={busy || input.trim().length < 3}
            className="rounded-xl bg-emerald-700 px-4 py-2 font-medium text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? "..." : "Ask"}
          </button>
        </form>
      </main>
    </div>
  );
}
