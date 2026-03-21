'use client'

import { useEffect, useState, useMemo } from 'react'
import dynamic from 'next/dynamic'
import Header from '@/components/ui/Header'
import { fetchAlloys } from '@/lib/api'
import type { AlloyData } from '@/types'

// The PeriodicTable component is assumed to exist; load it lazily so a build
// without the file degrades gracefully.
const PeriodicTable = dynamic(
  () =>
    import('@/components/simulation/PeriodicTable').then((m) => m.default ?? m),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-64 text-[#9898b0] text-sm">
        Loading Periodic Table...
      </div>
    ),
  }
)

type TabKey = 'periodic' | 'alloys'

export default function MaterialsPage() {
  const [activeTab, setActiveTab] = useState<TabKey>('periodic')
  const [alloys, setAlloys] = useState<AlloyData[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')

  useEffect(() => {
    if (activeTab !== 'alloys') return
    if (alloys.length > 0) return // already fetched

    setLoading(true)
    setError(null)
    fetchAlloys()
      .then(setAlloys)
      .catch((err) => setError(err?.message ?? 'Failed to load alloys'))
      .finally(() => setLoading(false))
  }, [activeTab, alloys.length])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return alloys
    return alloys.filter(
      (a) =>
        a.name.toLowerCase().includes(q) ||
        a.key.toLowerCase().includes(q) ||
        Object.keys(a.composition).some((el) => el.toLowerCase().includes(q))
    )
  }, [alloys, query])

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-[#e8e8f0] flex flex-col">
      <Header />

      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-10">
        {/* Page title */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-[#e8e8f0] tracking-tight">Materials Library</h1>
          <p className="mt-1 text-sm text-[#9898b0]">
            Browse elemental and alloy shielding materials with their electromagnetic properties.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 p-1 bg-[#12121a] border border-[#2a2a3e] rounded-lg w-fit">
          <TabButton
            label="Periodic Table"
            active={activeTab === 'periodic'}
            onClick={() => setActiveTab('periodic')}
          />
          <TabButton
            label="Alloys"
            active={activeTab === 'alloys'}
            onClick={() => setActiveTab('alloys')}
          />
        </div>

        {/* Tab content */}
        {activeTab === 'periodic' && (
          <section>
            <PeriodicTable onSelect={(symbol) => console.log('Selected:', symbol)} />
          </section>
        )}

        {activeTab === 'alloys' && (
          <section>
            {/* Search */}
            <div className="mb-5">
              <input
                type="search"
                placeholder="Search by name, key, or element symbol..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="input-field max-w-sm"
                aria-label="Search alloys"
              />
            </div>

            {/* States */}
            {loading && (
              <AlloySkeletonGrid />
            )}

            {error && !loading && (
              <div className="card rounded-xl p-6 text-sm text-[#ef4444]">
                {error}
              </div>
            )}

            {!loading && !error && filtered.length === 0 && (
              <div className="text-sm text-[#9898b0]">
                {query ? 'No alloys match your search.' : 'No alloys found.'}
              </div>
            )}

            {!loading && !error && filtered.length > 0 && (
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {filtered.map((alloy) => (
                  <AlloyCard key={alloy.key} alloy={alloy} />
                ))}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab button
// ---------------------------------------------------------------------------
function TabButton({
  label,
  active,
  onClick,
}: {
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-4 py-2 rounded-md text-sm font-medium transition-colors',
        active
          ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-sm'
          : 'text-[#9898b0] hover:text-[#e8e8f0]',
      ].join(' ')}
    >
      {label}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Alloy card
// ---------------------------------------------------------------------------
function AlloyCard({ alloy }: { alloy: AlloyData }) {
  const compositionEntries = Object.entries(alloy.composition).sort(
    ([, a], [, b]) => b - a
  )

  return (
    <div className="card-hover rounded-xl p-5 flex flex-col gap-3">
      {/* Header */}
      <div>
        <p className="text-xs font-mono text-[#9898b0] mb-0.5">{alloy.key}</p>
        <h3 className="text-sm font-semibold text-[#e8e8f0] leading-snug">{alloy.name}</h3>
      </div>

      {/* Composition pills */}
      <div className="flex flex-wrap gap-1.5">
        {compositionEntries.map(([el, frac]) => (
          <span
            key={el}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded border border-[#2a2a3e] bg-[#12121a] text-[10px] font-mono text-[#9898b0]"
          >
            <span className="text-cyan-400">{el}</span>
            <span>{(frac * 100).toFixed(frac < 0.01 ? 1 : 0)}%</span>
          </span>
        ))}
      </div>

      {/* Properties */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 pt-1 border-t border-[#2a2a3e]">
        <PropertyRow label="Conductivity" value={`${alloy.electrical_conductivity.toFixed(2)} MS/m`} />
        <PropertyRow label="Permeability" value={`${alloy.relative_permeability.toFixed(0)} µr`} />
        <PropertyRow label="Density" value={`${alloy.density.toFixed(2)} g/cm³`} />
        <PropertyRow label="Permittivity" value={`${alloy.relative_permittivity.toFixed(1)} εr`} />
      </div>

      {/* Note */}
      {alloy.note && (
        <p className="text-[11px] text-[#9898b0]/70 italic leading-relaxed border-t border-[#2a2a3e] pt-2">
          {alloy.note}
        </p>
      )}
    </div>
  )
}

function PropertyRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] text-[#9898b0]">{label}</span>
      <span className="text-xs font-mono text-[#e8e8f0]">{value}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------
function AlloySkeletonGrid() {
  return (
    <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="card rounded-xl p-5 flex flex-col gap-3 animate-pulse"
        >
          <div className="h-3 w-16 bg-[#2a2a3e] rounded" />
          <div className="h-4 w-3/4 bg-[#2a2a3e] rounded" />
          <div className="flex gap-1.5">
            <div className="h-4 w-10 bg-[#2a2a3e] rounded" />
            <div className="h-4 w-10 bg-[#2a2a3e] rounded" />
            <div className="h-4 w-10 bg-[#2a2a3e] rounded" />
          </div>
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#2a2a3e]">
            {Array.from({ length: 4 }).map((_, j) => (
              <div key={j} className="flex flex-col gap-1">
                <div className="h-2 w-12 bg-[#2a2a3e] rounded" />
                <div className="h-3 w-16 bg-[#2a2a3e] rounded" />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
