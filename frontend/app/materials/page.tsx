'use client'

import { useState, useEffect } from 'react'
import { fetchElements } from '@/lib/api'
import { ElementData } from '@/types'
import { Database, Search } from 'lucide-react'

export default function MaterialsPage() {
  const [elements, setElements] = useState<ElementData[]>([])
  const [search, setSearch] = useState('')

  useEffect(() => {
    fetchElements().then(setElements).catch(console.error)
  }, [])

  const filtered = elements.filter(e => 
    e.name.toLowerCase().includes(search.toLowerCase()) || 
    e.symbol.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-48px)] overflow-hidden bg-bg-canvas">
      
      {/* Header Area */}
      <div className="border-b border-border-panel bg-bg-panel p-4 shrink-0 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-bg-subpanel border border-border-panel rounded flex items-center justify-center">
            <Database className="w-4 h-4 text-accent-selection" />
          </div>
          <div>
            <h2 className="font-semibold text-text-primary text-sm">Material Library</h2>
            <p className="text-[10px] text-text-secondary">System defaults & user-defined parameters</p>
          </div>
        </div>

        <div className="relative w-64">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted" />
          <input 
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Filter by symbol or name..."
            className="w-full bg-bg-subpanel border border-border-panel rounded pl-8 pr-3 py-1.5 text-xs text-text-primary focus:border-accent-selection focus:outline-none"
          />
        </div>
      </div>

      {/* Main Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="bg-bg-panel sticky top-0 z-10 shadow-[0_1px_0_var(--border-panel)]">
            <tr>
              <th className="font-semibold text-text-secondary py-2 px-4 whitespace-nowrap">Symbol</th>
              <th className="font-semibold text-text-secondary py-2 px-4 whitespace-nowrap">Name</th>
              <th className="font-semibold text-text-secondary py-2 px-4 whitespace-nowrap">Atomic No.</th>
              <th className="font-semibold text-text-secondary py-2 px-4 whitespace-nowrap">Conductivity (S/m)</th>
              <th className="font-semibold text-text-secondary py-2 px-4 whitespace-nowrap">Permeability (µr)</th>
              <th className="font-semibold text-text-secondary py-2 px-4 whitespace-nowrap">Category</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((el) => (
              <tr key={el.symbol} className="border-b border-border-panel hover:bg-bg-subpanel transition-colors group">
                <td className="py-2 px-4 font-mono font-bold text-accent-selection">{el.symbol}</td>
                <td className="py-2 px-4 text-text-primary font-medium">{el.name}</td>
                <td className="py-2 px-4 text-text-secondary font-mono">{el.atomic_number}</td>
                <td className="py-2 px-4 text-text-primary font-mono">{el.electrical_conductivity.toExponential(2)}</td>
                <td className="py-2 px-4 text-text-primary font-mono">{el.relative_permeability}</td>
                <td className="py-2 px-4 text-text-secondary">
                  <span className="px-2 py-0.5 rounded bg-bg-canvas border border-border-panel text-[10px] uppercase tracking-wider">
                    {el.category || 'Element'}
                  </span>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="py-8 text-center text-text-muted">No materials match the filter criteria.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
