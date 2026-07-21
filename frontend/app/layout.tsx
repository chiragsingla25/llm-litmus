import type { ReactNode } from "react";

export const metadata = {
  title: "llm-litmus",
  description: "A litmus test for your local LLMs.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
