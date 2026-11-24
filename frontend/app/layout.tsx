import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JSJ Intelligence",
  description: "A cited, safety-checked RAG assistant over a document library.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
