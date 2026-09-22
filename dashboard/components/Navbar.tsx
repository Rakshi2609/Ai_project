"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bot, Camera, Radio, BarChart3, ShieldCheck, Activity, Cpu } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  const navLinks = [
    { href: "/", label: "Live Cockpit", icon: Activity },
    { href: "/capture", label: "Face & Voice Capture", icon: Camera },
    { href: "/robot", label: "Cobot Digital Twin", icon: Bot },
    { href: "/analytics", label: "Benchmarks & AI Studio", icon: BarChart3 },
  ];

  return (
    <header className="border-b border-cyan-500/20 bg-dark-950/85 backdrop-blur-2xl sticky top-0 z-50 shadow-[0_4px_30px_rgba(0,0,0,0.8)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
        
        {/* Logo and Brand */}
        <Link href="/" className="flex items-center space-x-3 group">
          <div className="relative">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-teal-400 via-cyan-500 to-blue-500 rounded-xl blur opacity-60 group-hover:opacity-100 transition duration-500" />
            <div className="relative w-10 h-10 rounded-xl bg-dark-900 border border-teal-400/40 flex items-center justify-center shadow-lg shadow-teal-500/20">
              <Bot className="w-5 h-5 text-teal-300 animate-pulse" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-sm sm:text-base tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-teal-200 via-cyan-100 to-sky-300 font-display">
                Cobot Trust AI
              </span>
              <span className="px-2 py-0.5 text-[9px] font-mono tracking-widest uppercase bg-teal-500/15 text-teal-300 border border-teal-400/40 rounded-full">
                Next.js
              </span>
            </div>
            <p className="text-[11px] text-gray-400 font-mono">Real-Time Multimodal Closed-Loop System</p>
          </div>
        </Link>

        {/* Proper Route Navigation */}
        <nav className="hidden md:flex items-center space-x-1 bg-dark-900/90 p-1.5 rounded-2xl border border-white/10 shadow-inner">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`px-3.5 py-1.5 text-xs font-semibold rounded-xl transition flex items-center space-x-1.5 ${
                  isActive
                    ? "bg-gradient-to-r from-teal-500/25 to-cyan-500/20 text-teal-200 border border-teal-400/50 shadow-[0_0_15px_rgba(20,184,166,0.3)] font-bold"
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? "text-teal-300" : "text-gray-400"}`} />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Status indicator */}
        <div className="flex items-center space-x-2.5">
          <Link
            href="/capture"
            className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-mono font-medium bg-teal-500/10 text-teal-300 border border-teal-500/30 hover:bg-teal-500/20 transition"
          >
            <Camera className="w-3 h-3 mr-1.5 text-teal-400 animate-pulse" />
            <span>Cam Feed</span>
          </Link>
          <span className="hidden sm:inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-mono font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
            <span className="w-1.5 h-1.5 mr-1.5 rounded-full bg-emerald-400 animate-ping" />
            &lt; 250ms RT Edge
          </span>
          <span className="px-2.5 py-1 rounded-full text-[11px] font-mono font-bold bg-teal-500/15 text-teal-300 border border-teal-500/30">
            MSE &lt; 0.08
          </span>
        </div>

      </div>

      {/* Mobile Nav */}
      <div className="md:hidden flex items-center justify-around border-t border-white/5 bg-dark-950/90 py-2 px-2">
        {navLinks.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`p-2 rounded-lg flex flex-col items-center text-[10px] ${
                isActive ? "text-teal-300 font-bold" : "text-gray-400"
              }`}
            >
              <Icon className="w-4 h-4 mb-0.5" />
              <span>{link.label.split(" ")[0]}</span>
            </Link>
          );
        })}
      </div>
    </header>
  );
}
