/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false, // Desactivamos para evitar doble renderizado del canvas
  experimental: {
    // Esto autoriza a tu navegador a recibir datos desde la IP de WSL
    allowedDevOrigins: ["localhost:3000", "172.28.112.1:3000"]
  }
};

module.exports = nextConfig;