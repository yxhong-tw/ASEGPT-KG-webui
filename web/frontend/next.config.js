/** @type {import('next').NextConfig} */
const nextConfig = {
  distDir: 'build',
  reactStrictMode: true,
  trailingSlash: true,
  transpilePackages: ['vis-network'],
}

module.exports = nextConfig
