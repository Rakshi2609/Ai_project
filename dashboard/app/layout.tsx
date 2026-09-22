import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cobot Trust AI - Real-Time Multimodal Human-Robot Collaboration",
  description:
    "Next.js Real-Time Multimodal System predicting human trust in collaborative robots (UR5) via real webcam expression, real audio prosody, 3D kinematics, and adaptive closed-loop mitigation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Orbitron:wght@500;700;900&family=Space+Grotesk:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen flex flex-col bg-[#050811] text-gray-100 antialiased selection:bg-teal-500 selection:text-black font-sans">
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
        <footer className="border-t border-white/5 py-4 text-center text-xs text-gray-500 font-mono">
          BCSE306L DA-1 / DA-2 • Multimodal Machine Learning for Human-Robot Trust • VIT Chennai
        </footer>
      </body>
    </html>
  );
}
