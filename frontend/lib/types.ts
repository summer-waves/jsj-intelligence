// types.ts - the exact shape of what the backend returns from POST /ask.
// It mirrors backend/app/schemas.py. If you change one side, change the other.

export interface Source {
  number: number; // the number used in the answer, for example 2 for [2]
  chunk_id: string;
  doc_id: string;
  title: string;
  url: string;
  excerpt: string;
  score: number;
  cited: boolean; // true if the answer cites this source
}

export interface Notices {
  safety: string;
  health: string | null;
  emergency: string | null;
}

export interface Timings {
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface AskResponse {
  question: string;
  answer: string;
  refused: boolean;
  withheld_reason: string | null;
  citations_used: number[];
  invalid_citations: number[];
  uncited_sentences: string[];
  sources: Source[];
  notices: Notices;
  model: string;
  retrieval: string;
  library_mode: string;
  timings: Timings;
}

export interface HealthResponse {
  status: string;
  library_mode: string;
  chunks: number;
  retrieval: string;
  model: string;
}
