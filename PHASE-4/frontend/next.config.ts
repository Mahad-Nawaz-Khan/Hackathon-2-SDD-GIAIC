import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone', // Enable standalone output for Docker
  experimental: {
    serverActions: {},
  },
  turbopack: {
    // Enable Turbopack
    rules: {
      '*.svg': {
        loaders: [
          {
            loader: '@svgr/webpack',
            options: {
              icon: true,
              svgo: false,
            },
          },
        ],
        as: '*.js',
      },
    },
  },
  async rewrites() {
    // Use internal service name for server-side rewrites
    // This works inside Kubernetes cluster
    const apiBaseUrl = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "http://todo-chatbot-backend:8000";

    return [
      {
        source: "/api/v1/:path*",
        destination: `${apiBaseUrl}/api/v1/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "img.clerk.com",
      },
    ],
  },
  // Enable compression
  compress: true,
  // Optimize bundle loading
  modularizeImports: {
    "@heroicons/react": {
      transform: "@heroicons/react/{{kebabCase member}}",
    },
  },
};

export default nextConfig;