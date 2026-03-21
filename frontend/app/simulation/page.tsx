'use client'

import { useState, useCallback } from 'react'
import {
  Shield,
  ChevronLeft,
  Loader2,
  Play,
  Bot,
  AlertCircle,
} from 'lucide-react'
import Link from 'next/link'

import { useCompositionStore, useSimulationStore, useResultsStore, useChatStore } from '@/lib/store'
import {
  calculateSE,
  frequencySweep,
  thicknessSweep,
  grainSizeSweep,
  coolingRateSweep,
  optimizeThickness,
} from '@/lib/api'
import type { OptimizationResult } from '@/types'

import CompositionPanel from '@/components/simulation/CompositionPanel'
import ShieldParameters, { type SweepParams } from '@/components/simulation/ShieldParameters'
import ResultsPanel from '@/components/simulation/ResultsPanel'
import ChatSidebar from '@/components/chat/ChatSidebar'

// ---- Validation banner ----

function ValidationBanner({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[#f59e0b]/10 border border-[#f59e0b]/30">
      <AlertCircle size={14} className="text-[#f59e0b] shrink-0" />
      <p className="text-xs text-[#f59e0b]">{message}</p>
    </div>
  )
}

// ---- Main page ----

export default function SimulationPage() {
  // Stores
  const { composition } = useCompositionStore()
  const { frequency_mhz, thickness_mm, grain_size_um, analysisMode } = useSimulationStore()
  const { setCalculationResult, setSweepResult, setLoading, setError, isLoading } =
    useResultsStore()
  const { isOpen: chatOpen, setOpen: setChatOpen } = useChatStore()

  // Local state
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null)
  const [targetSE, setTargetSE] = useState(60)
  const [validationMsg, setValidationMsg] = useState<string | null>(null)
  const [sweepParams, setSweepParams] = useState<SweepParams>({
    freqStart: 100,
    freqEnd: 10000,
    thicknessStart: 0.1,
    thicknessEnd: 10,
    grainStart: 1,
    grainEnd: 1000,
    numPoints: 50,
  })

  const handleSweepParamsChange = useCallback((p: Partial<SweepParams>) => {
    setSweepParams((prev) => ({ ...prev, ...p }))
  }, [])

  // ---- Validation ----
  const validate = useCallback((): string | null => {
    const elements = Object.entries(composition)
    if (elements.length === 0) return 'Add at least one element to the composition.'
    const total = elements.reduce((s, [, v]) => s + v, 0)
    if (Math.abs(total - 100) > 1) return `Composition sums to ${total.toFixed(1)}% — normalize to 100% first.`
    if (frequency_mhz <= 0 && analysisMode !== 'frequency-sweep') return 'Frequency must be greater than 0 MHz.'
    if (thickness_mm <= 0 && analysisMode !== 'thickness-sweep') return 'Thickness must be greater than 0 mm.'
    return null
  }, [composition, frequency_mhz, thickness_mm, analysisMode])

  // ---- Calculate handler ----
  const handleCalculate = useCallback(async () => {
    const msg = validate()
    if (msg) {
      setValidationMsg(msg)
      return
    }
    setValidationMsg(null)

    setLoading(true)
    setError(null)
    setCalculationResult(null)
    setSweepResult(null)
    setOptimizationResult(null)

    try {
      switch (analysisMode) {
        case 'single': {
          const result = await calculateSE({
            composition,
            frequency_mhz,
            thickness_mm,
            grain_size_um: grain_size_um ?? undefined,
          })
          setCalculationResult(result)
          break
        }

        case 'frequency-sweep': {
          const result = await frequencySweep({
            composition,
            thickness_mm,
            freq_start_mhz: sweepParams.freqStart,
            freq_end_mhz: sweepParams.freqEnd,
            num_points: sweepParams.numPoints,
            grain_size_um: grain_size_um ?? undefined,
          })
          setSweepResult(result)
          break
        }

        case 'thickness-sweep': {
          const result = await thicknessSweep({
            composition,
            frequency_mhz,
            thickness_start_mm: sweepParams.thicknessStart,
            thickness_end_mm: sweepParams.thicknessEnd,
            num_points: sweepParams.numPoints,
            grain_size_um: grain_size_um ?? undefined,
          })
          setSweepResult(result)
          break
        }

        case 'grain-sweep': {
          const result = await grainSizeSweep({
            composition,
            frequency_mhz,
            thickness_mm,
            grain_start_um: sweepParams.grainStart,
            grain_end_um: sweepParams.grainEnd,
            num_points: sweepParams.numPoints,
          })
          setSweepResult(result)
          break
        }

        case 'cooling-sweep': {
          const result = await coolingRateSweep({
            composition,
            frequency_mhz,
            thickness_mm,
          })
          setSweepResult(result)
          break
        }

        case 'optimize': {
          const result = await optimizeThickness({
            composition,
            frequency_mhz,
            target_se_db: targetSE,
            grain_size_um: grain_size_um ?? undefined,
          })
          setOptimizationResult(result)
          break
        }
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : typeof err === 'object' && err !== null && 'response' in err
          ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail ??
            'Unknown error'
          : 'Calculation failed'
      setError(message)
    } finally {
      setLoading(false)
    }
  }, [
    validate,
    analysisMode,
    composition,
    frequency_mhz,
    thickness_mm,
    grain_size_um,
    sweepParams,
    targetSE,
    setLoading,
    setError,
    setCalculationResult,
    setSweepResult,
  ])

  const compositionSummary = Object.entries(composition)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3)
    .map(([sym, pct]) => `${sym}${pct.toFixed(0)}`)
    .join('-')

  return (
    <div className="flex h-screen bg-[#0a0a0f] overflow-hidden">
      {/* ================================================================
          TOP NAVIGATION BAR
          ================================================================ */}
      <div className="fixed top-0 left-0 right-0 z-40 h-14 bg-[#12121a] border-b border-[#2a2a3e] flex items-center px-4 gap-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-[#e8e8f0] hover:text-[#00d4ff] transition-colors"
        >
          <Shield size={18} className="text-[#00d4ff]" />
          <span className="text-sm font-semibold">EMI Shield Designer</span>
        </Link>

        <div className="flex-1" />

        {/* Composition badge */}
        {compositionSummary && (
          <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#1a1a2e] border border-[#2a2a3e] text-xs text-[#9898b0] font-mono">
            {compositionSummary}
          </div>
        )}

        {/* Mode indicator */}
        <div className="px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-xs text-[#00d4ff] font-medium capitalize">
          {analysisMode.replace('-', ' ')}
        </div>

        {/* Chat toggle */}
        <button
          onClick={() => setChatOpen(!chatOpen)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all
            ${
              chatOpen
                ? 'bg-cyan-500/10 border-cyan-500/40 text-[#00d4ff]'
                : 'bg-[#1a1a2e] border-[#2a2a3e] text-[#9898b0] hover:text-[#e8e8f0] hover:border-[#3a3a52]'
            }`}
        >
          <Bot size={13} />
          <span className="hidden sm:inline">AI Chat</span>
        </button>
      </div>

      {/* ================================================================
          MAIN THREE-COLUMN LAYOUT  (below navbar)
          ================================================================ */}
      <div className="flex w-full pt-14 overflow-hidden">

        {/* ---- LEFT COLUMN: Composition ---- */}
        <aside className="w-80 shrink-0 border-r border-[#2a2a3e] flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">
            <CompositionPanel />
          </div>
        </aside>

        {/* ---- CENTER COLUMN: Parameters + Results ---- */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">

              {/* Shield Parameters section */}
              <section>
                <h2 className="text-xs font-semibold text-[#9898b0] uppercase tracking-widest mb-4 flex items-center gap-2">
                  <span className="w-1 h-4 rounded-full bg-gradient-to-b from-cyan-500 to-blue-600 inline-block" />
                  Shield Parameters
                </h2>
                <div className="card p-5">
                  <ShieldParameters
                    targetSE={targetSE}
                    onTargetSEChange={setTargetSE}
                    sweepParams={sweepParams}
                    onSweepParamsChange={handleSweepParamsChange}
                  />
                </div>
              </section>

              {/* Validation message */}
              {validationMsg && <ValidationBanner message={validationMsg} />}

              {/* Calculate button */}
              <button
                onClick={handleCalculate}
                disabled={isLoading}
                className="w-full py-3 px-6 rounded-xl font-semibold text-sm text-white
                           bg-gradient-to-r from-cyan-500 to-blue-600
                           hover:from-cyan-400 hover:to-blue-500
                           disabled:opacity-50 disabled:cursor-not-allowed
                           active:scale-[0.99] transition-all duration-200
                           shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Calculating...
                  </>
                ) : (
                  <>
                    <Play size={16} />
                    Calculate
                  </>
                )}
              </button>

              {/* Results section */}
              <section>
                <h2 className="text-xs font-semibold text-[#9898b0] uppercase tracking-widest mb-4 flex items-center gap-2">
                  <span className="w-1 h-4 rounded-full bg-gradient-to-b from-purple-500 to-indigo-600 inline-block" />
                  Results
                </h2>
                <ResultsPanel optimizationResult={optimizationResult} />
              </section>

              {/* Bottom padding */}
              <div className="h-8" />
            </div>
          </div>
        </main>

        {/* ---- RIGHT COLUMN: AI Chat (collapsible) ---- */}
        <aside
          className={`shrink-0 border-l border-[#2a2a3e] flex flex-col overflow-hidden
                       transition-all duration-300 ease-in-out
                       ${chatOpen ? 'w-80' : 'w-0 border-l-0'}`}
          aria-hidden={!chatOpen}
        >
          {chatOpen && (
            <ChatSidebar onClose={() => setChatOpen(false)} />
          )}
        </aside>

        {/* Collapsed chat toggle tab (when sidebar is closed) */}
        {!chatOpen && (
          <button
            onClick={() => setChatOpen(true)}
            className="fixed right-0 top-1/2 -translate-y-1/2 z-30
                       flex flex-col items-center gap-1 px-1.5 py-4
                       bg-[#16161f] border border-[#2a2a3e] border-r-0
                       rounded-l-xl text-[#9898b0] hover:text-[#00d4ff]
                       hover:border-cyan-500/40 transition-all duration-200"
            aria-label="Open AI chat"
          >
            <Bot size={14} />
            <ChevronLeft size={12} />
          </button>
        )}
      </div>
    </div>
  )
}
