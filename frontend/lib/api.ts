// api.ts - the only file that talks to the backend.
// The address comes from NEXT_PUBLIC_API_URL (see .env.local.example); on your laptop it is the local API.

import type { AskResponse, HealthResponse } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (res.status === 422) return "Please type a longer question (3 to 500 characters).";
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // the body was not JSON; fall through to the generic message
  }
  return `The server answered with an error (${res.status}).`;
}

export async function askQuestion(question: string): Promise<AskResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error("Could not reach the API. Is the backend running at " + API_URL + "?");
  }
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as AskResponse;
}

export async function getHealth(): Promise<HealthResponse | null> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok ? ((await res.json()) as HealthResponse) : null;
  } catch {
    return null;
  }
}
