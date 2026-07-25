"use client";

import { useEffect, useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

type HealthState = "loading" | "ok" | "unreachable";

export default function Home() {
  const [state, setState] = useState<HealthState>("loading");

  useEffect(() => {
    fetch(`${BACKEND_URL}/health`)
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((data) => setState(data.status === "ok" ? "ok" : "unreachable"))
      .catch(() => setState("unreachable"));
  }, []);

  return (
    <main>
      <h1>llm-litmus</h1>
      <p>Backend status: {state}</p>
    </main>
  );
}
