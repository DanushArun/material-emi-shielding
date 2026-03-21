'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Shield } from 'lucide-react'
import { checkHealth } from '@/lib/api'

type ApiStatus = 'checking' | 'healthy' | 'offline'

const NAV_LINKS: { label: string; href: string }[] = [
  { label: 'Home',       href: '/' },
  { label: 'Simulation', href: '/simulation' },
  { label: 'Materials',  href: '/materials' },
]

export default function Header() {
  const pathname = usePathname()
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [version, setVersion] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    checkHealth()
      .then((data) => {
        if (cancelled) return
        setApiStatus(data.status === 'healthy' ? 'healthy' : 'offline')
        if (data.version) setVersion(data.version)
      })
      .catch(() => {
        if (!cancelled) setApiStatus('offline')
      })
    return () => { cancelled = true }
  }, [])

  return (
    <header className="sticky top-0 z-50 bg-[#0a0a0f]/80 backdrop-blur-md border-b border-[#2a2a3e]">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between gap-6">

        {/* Left: brand */}
        <Link
          href="/"
          className="flex items-center gap-2.5 flex-shrink-0 group"
          aria-label="EMI Shield Designer home"
        >
          <Shield className="w-5 h-5 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
          <span className="text-[14px] font-semibold tracking-tight text-[#e8e8f0] group-hover:text-white transition-colors">
            EMI Shield Designer
          </span>
        </Link>

        {/* Center: nav */}
        <nav className="hidden md:flex items-center gap-1" aria-label="Primary navigation">
          {NAV_LINKS.map(({ label, href }) => {
            const active = href === '/' ? pathname === '/' : pathname.startsWith(href)
            return (
              <Link
                key={href}
                href={href}
                className={[
                  'px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors',
                  active
                    ? 'text-[#e8e8f0] bg-[#1a1a2e]'
                    : 'text-[#9898b0] hover:text-[#e8e8f0] hover:bg-[#12121a]',
                ].join(' ')}
              >
                {label}
              </Link>
            )
          })}
        </nav>

        {/* Right: API status + version */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <ApiStatusBadge status={apiStatus} />
          <VersionBadge version={version} />
        </div>
      </div>
    </header>
  )
}

// ---------------------------------------------------------------------------
// API status indicator
// ---------------------------------------------------------------------------
function ApiStatusBadge({ status }: { status: ApiStatus }) {
  const map: Record<ApiStatus, { dot: string; label: string }> = {
    checking: { dot: 'bg-amber-400 animate-pulse', label: 'Checking' },
    healthy:  { dot: 'bg-emerald-400',             label: 'API Online' },
    offline:  { dot: 'bg-red-500',                 label: 'API Offline' },
  }
  const { dot, label } = map[status]

  return (
    <div
      className="flex items-center gap-1.5"
      title={`Backend API: ${label}`}
      aria-label={`Backend API status: ${label}`}
    >
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`} />
      <span className="text-xs text-[#9898b0] hidden sm:inline">{label}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Version badge
// ---------------------------------------------------------------------------
function VersionBadge({ version }: { version: string | null }) {
  return (
    <span className="hidden sm:inline-block text-xs font-mono px-2 py-1 rounded border border-[#2a2a3e] bg-[#12121a] text-[#9898b0] select-none">
      {version ? `v${version}` : 'v4.0.0'}
    </span>
  )
}
