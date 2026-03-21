'use client'

import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Loader2, AlertCircle } from 'lucide-react'
import { fetchElements } from '@/lib/api'
import type { ElementData } from '@/types'

// ---------------------------------------------------------------------------
// Standard periodic table grid positions
// Row and col are 1-indexed; row 8 and 9 are the lanthanide / actinide rows
// rendered below the main 7-period body.
// ---------------------------------------------------------------------------
const ELEMENT_POSITIONS: Record<string, { row: number; col: number }> = {
  // Period 1
  H:  { row: 1, col: 1  },
  He: { row: 1, col: 18 },
  // Period 2
  Li: { row: 2, col: 1  },
  Be: { row: 2, col: 2  },
  B:  { row: 2, col: 13 },
  C:  { row: 2, col: 14 },
  N:  { row: 2, col: 15 },
  O:  { row: 2, col: 16 },
  F:  { row: 2, col: 17 },
  Ne: { row: 2, col: 18 },
  // Period 3
  Na: { row: 3, col: 1  },
  Mg: { row: 3, col: 2  },
  Al: { row: 3, col: 13 },
  Si: { row: 3, col: 14 },
  P:  { row: 3, col: 15 },
  S:  { row: 3, col: 16 },
  Cl: { row: 3, col: 17 },
  Ar: { row: 3, col: 18 },
  // Period 4
  K:  { row: 4, col: 1  },
  Ca: { row: 4, col: 2  },
  Sc: { row: 4, col: 3  },
  Ti: { row: 4, col: 4  },
  V:  { row: 4, col: 5  },
  Cr: { row: 4, col: 6  },
  Mn: { row: 4, col: 7  },
  Fe: { row: 4, col: 8  },
  Co: { row: 4, col: 9  },
  Ni: { row: 4, col: 10 },
  Cu: { row: 4, col: 11 },
  Zn: { row: 4, col: 12 },
  Ga: { row: 4, col: 13 },
  Ge: { row: 4, col: 14 },
  As: { row: 4, col: 15 },
  Se: { row: 4, col: 16 },
  Br: { row: 4, col: 17 },
  Kr: { row: 4, col: 18 },
  // Period 5
  Rb: { row: 5, col: 1  },
  Sr: { row: 5, col: 2  },
  Y:  { row: 5, col: 3  },
  Zr: { row: 5, col: 4  },
  Nb: { row: 5, col: 5  },
  Mo: { row: 5, col: 6  },
  Tc: { row: 5, col: 7  },
  Ru: { row: 5, col: 8  },
  Rh: { row: 5, col: 9  },
  Pd: { row: 5, col: 10 },
  Ag: { row: 5, col: 11 },
  Cd: { row: 5, col: 12 },
  In: { row: 5, col: 13 },
  Sn: { row: 5, col: 14 },
  Sb: { row: 5, col: 15 },
  Te: { row: 5, col: 16 },
  I:  { row: 5, col: 17 },
  Xe: { row: 5, col: 18 },
  // Period 6
  Cs: { row: 6, col: 1  },
  Ba: { row: 6, col: 2  },
  // La-Lu → lanthanide row (row 8), cols 4-17
  La: { row: 8, col: 4  },
  Ce: { row: 8, col: 5  },
  Pr: { row: 8, col: 6  },
  Nd: { row: 8, col: 7  },
  Pm: { row: 8, col: 8  },
  Sm: { row: 8, col: 9  },
  Eu: { row: 8, col: 10 },
  Gd: { row: 8, col: 11 },
  Tb: { row: 8, col: 12 },
  Dy: { row: 8, col: 13 },
  Ho: { row: 8, col: 14 },
  Er: { row: 8, col: 15 },
  Tm: { row: 8, col: 16 },
  Yb: { row: 8, col: 17 },
  Lu: { row: 6, col: 3  }, // Lu is a d-block element per IUPAC
  Hf: { row: 6, col: 4  },
  Ta: { row: 6, col: 5  },
  W:  { row: 6, col: 6  },
  Re: { row: 6, col: 7  },
  Os: { row: 6, col: 8  },
  Ir: { row: 6, col: 9  },
  Pt: { row: 6, col: 10 },
  Au: { row: 6, col: 11 },
  Hg: { row: 6, col: 12 },
  Tl: { row: 6, col: 13 },
  Pb: { row: 6, col: 14 },
  Bi: { row: 6, col: 15 },
  Po: { row: 6, col: 16 },
  At: { row: 6, col: 17 },
  Rn: { row: 6, col: 18 },
  // Period 7
  Fr: { row: 7, col: 1  },
  Ra: { row: 7, col: 2  },
  // Ac-Lr → actinide row (row 9), cols 4-17
  Ac: { row: 9, col: 4  },
  Th: { row: 9, col: 5  },
  Pa: { row: 9, col: 6  },
  U:  { row: 9, col: 7  },
  Np: { row: 9, col: 8  },
  Pu: { row: 9, col: 9  },
  Am: { row: 9, col: 10 },
  Cm: { row: 9, col: 11 },
  Bk: { row: 9, col: 12 },
  Cf: { row: 9, col: 13 },
  Es: { row: 9, col: 14 },
  Fm: { row: 9, col: 15 },
  Md: { row: 9, col: 16 },
  No: { row: 9, col: 17 },
  Lr: { row: 7, col: 3  }, // Lr is d-block per IUPAC
  Rf: { row: 7, col: 4  },
  Db: { row: 7, col: 5  },
  Sg: { row: 7, col: 6  },
  Bh: { row: 7, col: 7  },
  Hs: { row: 7, col: 8  },
  Mt: { row: 7, col: 9  },
  Ds: { row: 7, col: 10 },
  Rg: { row: 7, col: 11 },
  Cn: { row: 7, col: 12 },
  Nh: { row: 7, col: 13 },
  Fl: { row: 7, col: 14 },
  Mc: { row: 7, col: 15 },
  Lv: { row: 7, col: 16 },
  Ts: { row: 7, col: 17 },
  Og: { row: 7, col: 18 },
}

// ---------------------------------------------------------------------------
// Category detection
// Falls back to deriving category from atomic number / position when the
// API does not return a category string.
// ---------------------------------------------------------------------------
type CategoryKey =
  | 'alkali-metal'
  | 'alkaline-earth-metal'
  | 'transition-metal'
  | 'post-transition-metal'
  | 'metalloid'
  | 'nonmetal'
  | 'halogen'
  | 'noble-gas'
  | 'lanthanide'
  | 'actinide'
  | 'unknown'

const CATEGORY_STYLES: Record<CategoryKey, { bg: string; border: string; text: string }> = {
  'alkali-metal':          { bg: 'bg-blue-900/70',    border: 'border-blue-700',    text: 'text-blue-200'   },
  'alkaline-earth-metal':  { bg: 'bg-blue-800/60',    border: 'border-blue-600',    text: 'text-blue-100'   },
  'transition-metal':      { bg: 'bg-purple-900/70',  border: 'border-purple-700',  text: 'text-purple-200' },
  'post-transition-metal': { bg: 'bg-indigo-900/60',  border: 'border-indigo-600',  text: 'text-indigo-200' },
  'metalloid':             { bg: 'bg-teal-900/60',    border: 'border-teal-600',    text: 'text-teal-200'   },
  'nonmetal':              { bg: 'bg-green-900/60',   border: 'border-green-600',   text: 'text-green-200'  },
  'halogen':               { bg: 'bg-emerald-900/60', border: 'border-emerald-600', text: 'text-emerald-200'},
  'noble-gas':             { bg: 'bg-slate-700/70',   border: 'border-slate-500',   text: 'text-slate-200'  },
  'lanthanide':            { bg: 'bg-rose-900/60',    border: 'border-rose-600',    text: 'text-rose-200'   },
  'actinide':              { bg: 'bg-orange-900/60',  border: 'border-orange-700',  text: 'text-orange-200' },
  'unknown':               { bg: 'bg-[#1a1a2e]',      border: 'border-[#2a2a3e]',   text: 'text-gray-400'   },
}

const ALKALI_METALS       = new Set(['H','Li','Na','K','Rb','Cs','Fr'])
const ALKALINE_EARTH      = new Set(['Be','Mg','Ca','Sr','Ba','Ra'])
const NOBLE_GASES         = new Set(['He','Ne','Ar','Kr','Xe','Rn','Og'])
const HALOGENS            = new Set(['F','Cl','Br','I','At','Ts'])
const NONMETALS           = new Set(['C','N','O','P','S','Se'])
const METALLOIDS          = new Set(['B','Si','Ge','As','Sb','Te','Po'])
const POST_TRANSITION     = new Set(['Al','Ga','In','Sn','Tl','Pb','Bi','Nh','Fl','Mc','Lv'])
const LANTHANIDES         = new Set(['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'])
const ACTINIDES           = new Set(['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr'])

function deriveCategory(symbol: string, apiCategory?: string): CategoryKey {
  if (apiCategory) {
    const normalized = apiCategory.toLowerCase().replace(/\s+/g, '-') as CategoryKey
    if (normalized in CATEGORY_STYLES) return normalized
  }
  if (ALKALI_METALS.has(symbol))    return 'alkali-metal'
  if (ALKALINE_EARTH.has(symbol))   return 'alkaline-earth-metal'
  if (NOBLE_GASES.has(symbol))      return 'noble-gas'
  if (HALOGENS.has(symbol))         return 'halogen'
  if (NONMETALS.has(symbol))        return 'nonmetal'
  if (METALLOIDS.has(symbol))       return 'metalloid'
  if (POST_TRANSITION.has(symbol))  return 'post-transition-metal'
  if (LANTHANIDES.has(symbol))      return 'lanthanide'
  if (ACTINIDES.has(symbol))        return 'actinide'
  const pos = ELEMENT_POSITIONS[symbol]
  if (pos && pos.row >= 4 && pos.col >= 3 && pos.col <= 12) return 'transition-metal'
  return 'unknown'
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------
function fmtConductivity(val: number): string {
  if (!val && val !== 0) return 'N/A'
  if (val >= 1e6) return `${(val / 1e6).toFixed(2)} MS/m`
  if (val >= 1e3) return `${(val / 1e3).toFixed(2)} kS/m`
  return `${val.toFixed(4)} S/m`
}

function fmtDensity(val: number): string {
  if (!val && val !== 0) return 'N/A'
  return `${val.toFixed(2)} g/cm³`
}

function fmtPermeability(val: number): string {
  if (!val && val !== 0) return 'N/A'
  return val.toFixed(4)
}

// ---------------------------------------------------------------------------
// Tooltip component
// ---------------------------------------------------------------------------
interface TooltipProps {
  element: ElementData
  anchorRef: React.RefObject<HTMLDivElement>
  containerRef: React.RefObject<HTMLDivElement>
}

function ElementTooltip({ element, anchorRef, containerRef }: TooltipProps) {
  const [style, setStyle] = useState<React.CSSProperties>({ opacity: 0 })
  const tooltipRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!anchorRef.current || !containerRef.current || !tooltipRef.current) return
    const anchor    = anchorRef.current.getBoundingClientRect()
    const container = containerRef.current.getBoundingClientRect()
    const tip       = tooltipRef.current.getBoundingClientRect()

    // Position relative to the scroll container
    let top  = anchor.bottom - container.top + containerRef.current.scrollTop + 6
    let left = anchor.left   - container.left + anchor.width / 2 - tip.width / 2

    // Clamp within container bounds
    if (left < 4) left = 4
    if (left + tip.width > container.width - 4) left = container.width - tip.width - 4
    // Flip above if overflows bottom
    if (top + tip.height > container.height - 4) {
      top = anchor.top - container.top + containerRef.current.scrollTop - tip.height - 6
    }

    setStyle({ top, left, opacity: 1 })
  }, [anchorRef, containerRef])

  return (
    <div
      ref={tooltipRef}
      style={style}
      className="absolute z-50 pointer-events-none w-52 transition-opacity duration-150"
    >
      <div className="bg-[#0d0d1f] border border-cyan-500/40 rounded-lg p-3 shadow-xl shadow-black/60 text-xs">
        <p className="font-bold text-white text-sm mb-2 leading-tight">
          {element.name}
          <span className="text-gray-400 font-normal ml-1">#{element.atomic_number}</span>
        </p>
        <div className="space-y-1 text-gray-300">
          <div className="flex justify-between gap-2">
            <span className="text-gray-500">Conductivity</span>
            <span className="font-mono text-cyan-300">{fmtConductivity(element.electrical_conductivity)}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-gray-500">Permeability</span>
            <span className="font-mono text-purple-300">{fmtPermeability(element.relative_permeability)}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-gray-500">Density</span>
            <span className="font-mono text-green-300">{fmtDensity(element.density)}</span>
          </div>
          {element.atomic_weight > 0 && (
            <div className="flex justify-between gap-2">
              <span className="text-gray-500">At. weight</span>
              <span className="font-mono text-gray-300">{element.atomic_weight.toFixed(3)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Single element cell
// ---------------------------------------------------------------------------
interface ElementCellProps {
  element: ElementData
  isSelected: boolean
  onSelect: (symbol: string) => void
  containerRef: React.RefObject<HTMLDivElement>
}

function ElementCell({ element, isSelected, onSelect, containerRef }: ElementCellProps) {
  const [hovered, setHovered] = useState(false)
  const cellRef = useRef<HTMLDivElement>(null)
  const category = deriveCategory(element.symbol, element.category)
  const styles   = CATEGORY_STYLES[category]

  return (
    <>
      <div
        ref={cellRef}
        role="button"
        tabIndex={0}
        aria-label={`${element.name} (${element.symbol}), atomic number ${element.atomic_number}`}
        aria-pressed={isSelected}
        onClick={() => onSelect(element.symbol)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(element.symbol) }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onFocus={() => setHovered(true)}
        onBlur={() => setHovered(false)}
        className={[
          'relative flex flex-col items-center justify-center',
          'w-full h-full min-h-0 min-w-0',
          'rounded border cursor-pointer select-none',
          'transition-all duration-150',
          styles.bg,
          isSelected
            ? 'border-cyan-400 shadow-[0_0_10px_2px_rgba(34,211,238,0.5)] z-10 scale-105'
            : hovered
              ? `${styles.border} shadow-[0_0_8px_1px_rgba(255,255,255,0.15)] scale-105 z-10`
              : `${styles.border}`,
        ].join(' ')}
      >
        {/* Atomic number */}
        <span className={`absolute top-[2px] left-[3px] leading-none font-mono ${styles.text} opacity-80`}
          style={{ fontSize: 'clamp(5px, 0.55vw, 8px)' }}>
          {element.atomic_number}
        </span>

        {/* Symbol */}
        <span className={`font-bold leading-none ${isSelected ? 'text-cyan-300' : styles.text}`}
          style={{ fontSize: 'clamp(8px, 1vw, 14px)' }}>
          {element.symbol}
        </span>

        {/* Name */}
        <span className="text-gray-400 leading-none truncate w-full text-center px-px"
          style={{ fontSize: 'clamp(4px, 0.45vw, 7px)' }}>
          {element.name}
        </span>
      </div>

      {/* Tooltip rendered via portal-like absolute positioning inside container */}
      {hovered && (
        <ElementTooltip
          element={element}
          anchorRef={cellRef as React.RefObject<HTMLDivElement>}
          containerRef={containerRef}
        />
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// Placeholder cell (empty grid slot)
// ---------------------------------------------------------------------------
function EmptyCell() {
  return <div className="w-full h-full rounded border border-transparent bg-transparent" />
}

// ---------------------------------------------------------------------------
// Legend strip
// ---------------------------------------------------------------------------
const LEGEND_ITEMS: Array<{ category: CategoryKey; label: string }> = [
  { category: 'alkali-metal',          label: 'Alkali metal'          },
  { category: 'alkaline-earth-metal',  label: 'Alkaline earth'        },
  { category: 'transition-metal',      label: 'Transition metal'      },
  { category: 'post-transition-metal', label: 'Post-transition'       },
  { category: 'metalloid',             label: 'Metalloid'             },
  { category: 'nonmetal',              label: 'Nonmetal'              },
  { category: 'halogen',               label: 'Halogen'               },
  { category: 'noble-gas',             label: 'Noble gas'             },
  { category: 'lanthanide',            label: 'Lanthanide'            },
  { category: 'actinide',              label: 'Actinide'              },
]

function Legend() {
  return (
    <div className="flex flex-wrap gap-2 mt-3 px-1">
      {LEGEND_ITEMS.map(({ category, label }) => {
        const s = CATEGORY_STYLES[category]
        return (
          <div key={category} className="flex items-center gap-1">
            <div className={`w-3 h-3 rounded-sm border ${s.bg} ${s.border}`} />
            <span className="text-gray-400" style={{ fontSize: '10px' }}>{label}</span>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main PeriodicTable component
// ---------------------------------------------------------------------------
export interface PeriodicTableProps {
  /** Called with the element symbol when a cell is clicked. */
  onSelect: (symbol: string) => void
  /** Currently selected element symbols (supports multi-select externally). */
  selectedSymbols?: string[]
}

export default function PeriodicTable({ onSelect, selectedSymbols = [] }: PeriodicTableProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  const {
    data: elements = [],
    isLoading,
    isError,
    error,
  } = useQuery<ElementData[], Error>({
    queryKey: ['elements'],
    queryFn: fetchElements,
    staleTime: 5 * 60 * 1000,
    retry: 2,
  })

  // Build lookup map: symbol -> ElementData
  const elementMap = new Map<string, ElementData>(elements.map((el) => [el.symbol, el]))

  // Build a placeholder element for symbols in the positions map but not in the API
  // (synthetic/rare elements the backend may not carry)
  function makePlaceholder(symbol: string): ElementData {
    return {
      symbol,
      name: symbol,
      atomic_number: 0,
      atomic_weight: 0,
      electrical_conductivity: 0,
      relative_permeability: 1,
      density: 0,
    }
  }

  // ROWS 1-7: main table body.
  // ROW 8: lanthanide series (La-Yb, cols 4-17).
  // ROW 9: actinide series   (Ac-No, cols 4-17).
  // Rows 1-7 form the main 18-column grid; rows 8-9 render below as the f-block.

  const COLS = 18
  const allSymbols = Object.keys(ELEMENT_POSITIONS)

  function buildRow(rowIdx: number): Array<string | null> {
    const cells: Array<string | null> = Array(COLS).fill(null)
    allSymbols.forEach((sym) => {
      const pos = ELEMENT_POSITIONS[sym]
      if (pos.row === rowIdx) {
        cells[pos.col - 1] = sym
      }
    })
    return cells
  }

  const mainRows   = [1, 2, 3, 4, 5, 6, 7].map(buildRow)
  const fBlockRows = [8, 9].map(buildRow)

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 bg-[#0d0d1f] rounded-xl border border-[#2a2a3e]">
        <div className="flex flex-col items-center gap-3 text-gray-400">
          <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
          <span className="text-sm">Loading element data...</span>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center h-64 bg-[#0d0d1f] rounded-xl border border-red-800">
        <div className="flex flex-col items-center gap-3 text-red-400 px-6 text-center">
          <AlertCircle className="w-8 h-8" />
          <span className="text-sm font-medium">Failed to load elements</span>
          <span className="text-xs text-gray-500">{(error as Error)?.message ?? 'Unknown error'}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-[#0d0d1f] rounded-xl border border-[#2a2a3e] p-3 sm:p-4 select-none overflow-x-auto">
      {/* Title */}
      <div className="mb-3 px-1 flex items-center justify-between">
        <h2 className="text-white font-semibold tracking-wide text-sm sm:text-base">
          Periodic Table of Elements
        </h2>
        {selectedSymbols.length > 0 && (
          <span className="text-xs text-cyan-400 font-mono bg-cyan-400/10 border border-cyan-400/30 rounded px-2 py-0.5">
            {selectedSymbols.join(', ')}
          </span>
        )}
      </div>

      {/* Scrollable grid wrapper */}
      <div ref={containerRef} className="relative min-w-[600px]">
        {/* Main grid: rows 1-7, 18 columns */}
        <div
          className="grid gap-[2px]"
          style={{
            gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))`,
            gridTemplateRows: 'repeat(7, minmax(0, 1fr))',
            aspectRatio: `${COLS} / 8`,
          }}
        >
          {mainRows.map((row, rowIdx) =>
            row.map((sym, colIdx) => {
              if (!sym) return <EmptyCell key={`${rowIdx}-${colIdx}`} />
              const el = elementMap.get(sym) ?? makePlaceholder(sym)
              return (
                <ElementCell
                  key={sym}
                  element={el}
                  isSelected={selectedSymbols.includes(sym)}
                  onSelect={onSelect}
                  containerRef={containerRef as React.RefObject<HTMLDivElement>}
                />
              )
            })
          )}
        </div>

        {/* Gap row between main table and f-block */}
        <div className="h-2 sm:h-3" />

        {/* f-block label + rows */}
        <div className="flex items-start gap-[2px]">
          {/* Side label */}
          <div className="flex flex-col justify-around shrink-0" style={{ height: `calc((100% - 4px))` }}>
            <div
              className="text-gray-600 font-mono leading-none"
              style={{ fontSize: 'clamp(5px, 0.55vw, 8px)', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
            >
              Lanthanides / Actinides
            </div>
          </div>

          {/* f-block grid: rows 8 and 9, only cols 4-17 (indices 3-16) */}
          <div
            className="grid gap-[2px] flex-1"
            style={{
              gridTemplateColumns: `repeat(14, minmax(0, 1fr))`,
              gridTemplateRows: 'repeat(2, minmax(0, 1fr))',
              aspectRatio: `${14} / 2.4`,
            }}
          >
            {fBlockRows.map((row, rowIdx) =>
              // Only cols 4-17 (array indices 3-16)
              row.slice(3, 17).map((sym, colIdx) => {
                if (!sym) return <EmptyCell key={`f-${rowIdx}-${colIdx}`} />
                const el = elementMap.get(sym) ?? makePlaceholder(sym)
                return (
                  <ElementCell
                    key={sym}
                    element={el}
                    isSelected={selectedSymbols.includes(sym)}
                    onSelect={onSelect}
                    containerRef={containerRef as React.RefObject<HTMLDivElement>}
                  />
                )
              })
            )}
          </div>
        </div>
      </div>

      {/* Legend */}
      <Legend />

      {/* Element count footer */}
      <p className="mt-2 px-1 text-gray-600 text-right" style={{ fontSize: '10px' }}>
        {elements.length} elements loaded from API
      </p>
    </div>
  )
}
