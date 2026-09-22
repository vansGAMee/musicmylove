import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "export",
  allowedDevOrigins: ["localhost:3000", "127.0.0.1:3000", "127.0.0.1", "localhost"],
};

export default nextConfig;
