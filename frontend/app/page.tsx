'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Shield, Zap, BarChart3, Brain, Activity } from 'lucide-react'
import { checkHealth } from '@/lib/api'

interface HealthState {
  status: 'checking' | 'healthy' | 'offline'
  version: string | null
}

export default function Home() {
  const [health, setHealth] = useState<HealthState>({ status: 'checking', version: null })

  useEffect(() => {
    checkHealth()
      .then((data) => setHealth({ status: data.status === 'healthy' ? 'healthy' : 'offline', version: data.version }))
      .catch(() => setHealth({ status: 'offline', version: null }))
  }, [])

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-[#e8e8f0] flex flex-col">
      {/* Navigation */}
      <header className="sticky top-0 z-50 bg-[#0a0a0f]/80 border-b border-[#2a2a3e] backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          {/* Left: brand */}
          <div className="flex items-center gap-2.5">
            <Shield className="w-6 h-6 text-cyan-400" />
            <span className="text-[15px] font-semibold tracking-tight text-[#e8e8f0]">
              EMI Shield Designer
            </span>
          </div>

          {/* Center: nav links */}
          <nav className="hidden md:flex items-center gap-6">
            <Link
              href="/"
              className="text-sm font-medium text-cyan-400 border-b border-cyan-500/50 pb-0.5"
            >
              Home
            </Link>
            <Link
              href="/simulation"
              className="text-sm font-medium text-[#9898b0] hover:text-[#e8e8f0] transition-colors"
            >
              Simulation
            </Link>
            <Link
              href="/materials"
              className="text-sm font-medium text-[#9898b0] hover:text-[#e8e8f0] transition-colors"
            >
              Materials
            </Link>
          </nav>

          {/* Right: status + version */}
          <div className="flex items-center gap-3">
            <StatusIndicator status={health.status} />
            <span className="hidden sm:inline-block text-xs font-mono px-2 py-1 rounded border border-[#2a2a3e] bg-[#12121a] text-[#9898b0]">
              v4.0.0
            </span>
          </div>
        </div>
      </header>

      <main className="flex-1">
        {/* Hero */}
        <section className="relative max-w-7xl mx-auto px-6 pt-24 pb-20 text-center">
          {/* Ambient glow */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 flex items-start justify-center overflow-hidden"
          >
            <div className="mt-8 h-[480px] w-[800px] rounded-full bg-cyan-500/5 blur-[120px]" />
          </div>

          {/* Eyebrow */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-medium tracking-wide mb-6">
            <Activity className="w-3 h-3" />
            Physics-Accurate EMI Simulation Platform
          </div>

          <h1 className="relative text-5xl sm:text-6xl font-bold tracking-tight leading-[1.08] mb-5">
            <span className="bg-gradient-to-br from-white via-[#e8e8f0] to-[#9898b0] bg-clip-text text-transparent">
              EMI Shield
            </span>{' '}
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Designer
            </span>
          </h1>

          <p className="relative max-w-2xl mx-auto text-[17px] leading-relaxed text-[#9898b0] mb-10">
            AI-powered electromagnetic interference shielding simulation. Model composite
            materials, run frequency sweeps, and receive Gemini-assisted design recommendations
            — all grounded in Transfer Matrix Method physics.
          </p>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link
              href="/simulation"
              className="btn-primary px-7 py-3 text-sm font-semibold rounded-lg inline-flex items-center gap-2 shadow-lg shadow-cyan-900/30"
            >
              <Zap className="w-4 h-4" />
              Start Designing
            </Link>
            <Link
              href="/materials"
              className="btn-secondary px-7 py-3 text-sm font-semibold rounded-lg inline-flex items-center gap-2"
            >
              Browse Materials
            </Link>
          </div>
        </section>

        {/* Feature cards */}
        <section className="max-w-7xl mx-auto px-6 pb-20">
          <div className="grid md:grid-cols-3 gap-5">
            <FeatureCard
              icon={<Zap className="w-5 h-5 text-cyan-400" />}
              iconBg="bg-cyan-500/10 border-cyan-500/20"
              title="Advanced Physics"
              items={[
                'Transfer Matrix Method (TMM)',
                'Percolation threshold modelling',
                "Snoek's limit for permeability",
                'Skin depth & multi-reflection',
              ]}
            />
            <FeatureCard
              icon={<Brain className="w-5 h-5 text-violet-400" />}
              iconBg="bg-violet-500/10 border-violet-500/20"
              title="AI-Powered Design"
              items={[
                'Gemini material recommendations',
                'Composition optimisation',
                'Contextual design rationale',
                'Interactive AI chat interface',
              ]}
            />
            <FeatureCard
              icon={<BarChart3 className="w-5 h-5 text-blue-400" />}
              iconBg="bg-blue-500/10 border-blue-500/20"
              title="Uncertainty Quantification"
              items={[
                'Monte Carlo sensitivity analysis',
                'Sobol index decomposition',
                'Confidence intervals per result',
                'Grain size & cooling rate sweeps',
              ]}
            />
          </div>
        </section>

        {/* System status bar */}
        <section className="max-w-7xl mx-auto px-6 pb-20">
          <div className="card rounded-xl px-6 py-5">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-2.5">
                <Activity className="w-4 h-4 text-[#9898b0]" />
                <span className="text-sm font-medium text-[#9898b0]">System Status</span>
              </div>
              <div className="flex items-center gap-6 flex-wrap">
                <ServiceStatus
                  label="Backend API"
                  status={health.status}
                  version={health.version}
                />
                <div className="h-4 w-px bg-[#2a2a3e]" />
                <ServiceStatus label="Physics Engine" status="healthy" />
                <div className="h-4 w-px bg-[#2a2a3e]" />
                <ServiceStatus label="AI Assistant" status="healthy" />
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[#2a2a3e] py-6">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-2 text-[#9898b0] text-xs">
            <Shield className="w-3.5 h-3.5" />
            EMI Shield Designer v4.0.0
          </div>
          <a
            href="http://localhost:8001/docs"
            target="_blank"
            rel="noreferrer"
            className="text-xs text-[#9898b0] hover:text-cyan-400 transition-colors font-mono"
          >
            API Docs: localhost:8001/docs
          </a>
        </div>
      </footer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusIndicator({ status }: { status: HealthState['status'] }) {
  const map: Record<HealthState['status'], { dot: string; label: string }> = {
    checking: { dot: 'bg-amber-400 animate-pulse', label: 'Checking' },
    healthy:  { dot: 'bg-emerald-400',             label: 'API Online' },
    offline:  { dot: 'bg-red-500',                 label: 'API Offline' },
  }
  const { dot, label } = map[status]

  return (
    <div className="flex items-center gap-1.5">
      <span className={`inline-block w-2 h-2 rounded-full ${dot}`} />
      <span className="text-xs text-[#9898b0]">{label}</span>
    </div>
  )
}

function FeatureCard({
  icon,
  iconBg,
  title,
  items,
}: {
  icon: React.ReactNode
  iconBg: string
  title: string
  items: string[]
}) {
  return (
    <div className="card-hover rounded-xl p-6 flex flex-col gap-4">
      <div className={`w-9 h-9 rounded-lg border flex items-center justify-center ${iconBg}`}>
        {icon}
      </div>
      <h3 className="text-base font-semibold text-[#e8e8f0]">{title}</h3>
      <ul className="space-y-2">
        {items.map((item) => (
          <li key={item} className="flex items-start gap-2 text-sm text-[#9898b0]">
            <span className="mt-[5px] w-1 h-1 rounded-full bg-cyan-500/60 flex-shrink-0" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}

function ServiceStatus({
  label,
  status,
  version,
}: {
  label: string
  status: HealthState['status'] | 'healthy'
  version?: string | null
}) {
  const isHealthy = status === 'healthy'
  const isChecking = status === 'checking'

  return (
    <div className="flex items-center gap-2">
      <span
        className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
          isHealthy ? 'bg-emerald-400' : isChecking ? 'bg-amber-400 animate-pulse' : 'bg-red-500'
        }`}
      />
      <span className="text-xs text-[#9898b0]">{label}</span>
      {version && (
        <span className="text-xs font-mono text-[#9898b0]/60">{version}</span>
      )}
    </div>
  )
}
