import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "MusicMyLove", description: "Five songs in. Your next favorites out." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en" suppressHydrationWarning><body suppressHydrationWarning>{children}</body></html>; }
