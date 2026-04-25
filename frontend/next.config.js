/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,  // Evita doble-inicialización en dev (crítico para WebSocket + WebGL)
};

module.exports = nextConfig;
