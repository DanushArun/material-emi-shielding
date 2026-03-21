'use client'

import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Trash2, Plus, RefreshCw, ChevronDown } from 'lucide-react'
import { useCompositionStore } from '@/lib/store'
import { fetchAlloys } from '@/lib/api'
import type { AlloyData } from '@/types'

// ---- element color map keyed by periodic table category ----
const CATEGORY_COLORS: Record<string, string> = {
  'transition metal': 'text-[#00d4ff]',
  'post-transition metal': 'text-[#8b5cf6]',
  'metalloid': 'text-[#f59e0b]',
  'alkali metal': 'text-[#ef4444]',
  'alkaline earth metal': 'text-[#10b981]',
  'lanthanide': 'text-[#ec4899]',
  'other nonmetal': 'text-[#6ee7b7]',
}

function elementColor(symbol: string): string {
  // rough heuristic so we don't need server round-trip per element
  const TRANS = ['Fe','Ni','Co','Cu','Cr','Mn','Mo','W','Ti','V','Nb','Zr','Hf','Ta','Re']
  if (TRANS.includes(symbol)) return CATEGORY_COLORS['transition metal']
  const POST = ['Al','Sn','Pb','Bi','In','Ga']
  if (POST.includes(symbol)) return CATEGORY_COLORS['post-transition metal']
  const META = ['Si','Ge','As','Sb','Te','B']
  if (META.includes(symbol)) return CATEGORY_COLORS['metalloid']
  return 'text-[#e8e8f0]'
}

// ---- Sub-components ----

interface ElementRowProps {
  symbol: string
  percentage: number
  onChange: (symbol: string, val: number) => void
  onRemove: (symbol: string) => void
}

function ElementRow({ symbol, percentage, onChange, onRemove }: ElementRowProps) {
  const color = elementColor(symbol)

  return (
    <div className="flex items-center gap-2 py-2 border-b border-[#2a2a3e] last:border-0 group">
      {/* Symbol badge */}
      <div className="w-10 h-10 rounded-lg bg-[#12121a] border border-[#2a2a3e] flex items-center justify-center shrink-0">
        <span className={`text-sm font-bold font-mono ${color}`}>{symbol}</span>
      </div>

      {/* Percentage input */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1">
          <input
            type="number"
            min={0}
            max={100}
            step={0.1}
            value={percentage}
            onChange={(e) => onChange(symbol, parseFloat(e.target.value) || 0)}
            className="input-field text-right pr-1 font-mono"
            aria-label={`${symbol} percentage`}
          />
          <span className="text-[#9898b0] text-sm shrink-0">%</span>
        </div>
        {/* Mini progress bar */}
        <div className="mt-1 h-1 rounded-full bg-[#1a1a2e] overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-300"
            style={{ width: `${Math.min(percentage, 100)}%` }}
          />
        </div>
      </div>

      {/* Remove button */}
      <button
        onClick={() => onRemove(symbol)}
        className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-[#2a2a3e] text-[#ef4444]"
        aria-label={`Remove ${symbol}`}
      >
        <Trash2 size={14} />
      </button>
    </div>
  )
}

// ---- Preset alloy dropdown ----
interface PresetDropdownProps {
  alloys: AlloyData[]
  onSelect: (alloy: AlloyData) => void
}

function PresetDropdown({ alloys, onSelect }: PresetDropdownProps) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')

  const filtered = alloys.filter(
    (a) =>
      a.name.toLowerCase().includes(search.toLowerCase()) ||
      a.key.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="btn-secondary w-full flex items-center justify-between text-sm"
      >
        <span className="text-[#9898b0]">Load preset alloy...</span>
        <ChevronDown size={14} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-[#16161f] border border-[#2a2a3e] rounded-xl shadow-2xl overflow-hidden">
          <div className="p-2">
            <input
              type="text"
              placeholder="Search alloys..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="input-field text-xs"
              autoFocus
            />
          </div>
          <div className="max-h-56 overflow-y-auto">
            {filtered.length === 0 ? (
              <p className="text-center text-[#9898b0] text-sm py-4">No alloys found</p>
            ) : (
              filtered.map((alloy) => (
                <button
                  key={alloy.key}
                  onClick={() => {
                    onSelect(alloy)
                    setOpen(false)
                    setSearch('')
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-[#1a1a2e] transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-[#e8e8f0] font-medium">{alloy.name}</span>
                    <span className="text-xs text-[#9898b0] font-mono">{alloy.key}</span>
                  </div>
                  <div className="text-xs text-[#9898b0] mt-0.5 truncate">
                    {Object.entries(alloy.composition)
                      .sort(([, a], [, b]) => b - a)
                      .map(([sym, pct]) => `${sym} ${pct}%`)
                      .join(', ')}
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ---- Main component ----

export default function CompositionPanel() {
  const { composition, updateElement, removeElement, setComposition, clearComposition } =
    useCompositionStore()

  const { data: alloys = [], isLoading: alloysLoading } = useQuery<AlloyData[]>({
    queryKey: ['alloys'],
    queryFn: fetchAlloys,
    staleTime: Infinity,
  })

  const elements = Object.entries(composition)
  const total = elements.reduce((sum, [, pct]) => sum + pct, 0)
  const isExact = Math.abs(total - 100) < 0.05
  const isEmpty = elements.length === 0

  const normalize = useCallback(() => {
    if (total === 0) return
    const factor = 100 / total
    const normalized: Record<string, number> = {}
    for (const [sym, pct] of elements) {
      normalized[sym] = Math.round(pct * factor * 10) / 10
    }
    // Fix floating-point drift on last element
    const syms = Object.keys(normalized)
    if (syms.length > 0) {
      const partialSum = syms.slice(0, -1).reduce((s, k) => s + normalized[k], 0)
      normalized[syms[syms.length - 1]] = Math.round((100 - partialSum) * 10) / 10
    }
    setComposition(normalized)
  }, [total, elements, setComposition])

  const handlePresetSelect = useCallback(
    (alloy: AlloyData) => {
      const scaled: Record<string, number> = {}
      const entries = Object.entries(alloy.composition)
      const compTotal = entries.reduce((s, [, v]) => s + v, 0)
      for (const [sym, pct] of entries) {
        scaled[sym] = Math.round((pct / compTotal) * 1000) / 10
      }
      setComposition(scaled)
    },
    [setComposition]
  )

  const totalColor = isExact
    ? 'text-[#10b981]'
    : total > 100
    ? 'text-[#ef4444]'
    : 'text-[#f59e0b]'

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[#e8e8f0] tracking-wide uppercase">
          Composition
        </h2>
        {!isEmpty && (
          <button
            onClick={clearComposition}
            className="text-xs text-[#9898b0] hover:text-[#ef4444] transition-colors"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Preset selector */}
      <PresetDropdown
        alloys={alloysLoading ? [] : alloys}
        onSelect={handlePresetSelect}
      />

      {/* Element list */}
      <div className="card flex-1 overflow-y-auto p-3">
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center h-full py-12 gap-3">
            <div className="w-12 h-12 rounded-full bg-[#1a1a2e] flex items-center justify-center">
              <Plus size={20} className="text-[#3a3a52]" />
            </div>
            <p className="text-sm text-[#9898b0] text-center">
              Add elements from the periodic table or load a preset alloy
            </p>
          </div>
        ) : (
          elements.map(([symbol, percentage]) => (
            <ElementRow
              key={symbol}
              symbol={symbol}
              percentage={percentage}
              onChange={updateElement}
              onRemove={removeElement}
            />
          ))
        )}
      </div>

      {/* Total indicator + normalize */}
      <div className="card p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-[#9898b0]">Total</span>
          <span className={`text-sm font-bold font-mono ${totalColor}`}>
            {total.toFixed(1)}%
          </span>
        </div>

        {/* Stacked bar */}
        <div className="h-2 rounded-full bg-[#1a1a2e] overflow-hidden mb-3">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              isExact
                ? 'bg-gradient-to-r from-emerald-500 to-green-400'
                : total > 100
                ? 'bg-[#ef4444]'
                : 'bg-gradient-to-r from-cyan-500 to-blue-600'
            }`}
            style={{ width: `${Math.min(total, 100)}%` }}
          />
        </div>

        <div className="flex items-center gap-2">
          {isExact ? (
            <span className="flex-1 text-xs text-[#10b981] font-medium">
              Composition valid
            </span>
          ) : (
            <span className="flex-1 text-xs text-[#f59e0b]">
              {total > 100
                ? `Over by ${(total - 100).toFixed(1)}%`
                : `${(100 - total).toFixed(1)}% remaining`}
            </span>
          )}

          <button
            onClick={normalize}
            disabled={isEmpty || total === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-[#1a1a2e] border border-[#2a2a3e]
                       rounded-lg text-[#9898b0] hover:text-[#e8e8f0] hover:border-[#3a3a52]
                       transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <RefreshCw size={12} />
            Normalize
          </button>
        </div>
      </div>
    </div>
  )
}
