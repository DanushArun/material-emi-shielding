'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import {
  Shield,
  ChevronRight,
  ChevronLeft,
  Send,
  Loader2,
  Play,
  Bot,
  User,
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
import type { OptimizationResult, ChatMessage } from '@/types'

import CompositionPanel from '@/components/simulation/CompositionPanel'
import ShieldParameters, { type SweepParams } from '@/components/simulation/ShieldParameters'
import ResultsPanel from '@/components/simulation/ResultsPanel'

// ---- AI Chat sidebar ----

function ChatSidebar({ onClose }: { onClose: () => void }) {
  const { messages, isLoading, addMessage, setLoading } = useChatStore()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input on open
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const { composition } = useCompositionStore()
  const { frequency_mhz, thickness_mm, grain_size_um, analysisMode } = useSimulationStore()
  const { calculationResult, sweepResult } = useResultsStore()

  const sendMessage = useCallback(async () => {
    const text = input.trim()
    if (!text || isLoading) return

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    }
    addMessage(userMsg)
    setInput('')
    setLoading(true)

    try {
      // Build context-rich system message for the backend
      const context = {
        composition,
        frequency_mhz,
        thickness_mm,
        grain_size_um,
        analysisMode,
        latestResult: calculationResult ?? sweepResult ?? null,
      }

      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'}/api/v1/ai/chat`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            context,
            history: messages.slice(-10).map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }),
        }
      )

      if (!res.ok) throw new Error(`AI service returned ${res.status}`)
      const data = await res.json()

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.response ?? data.message ?? 'No response from AI.',
        timestamp: new Date(),
      }
      addMessage(assistantMsg)
    } catch (err) {
      const errMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content:
          'Sorry, the AI assistant is unavailable right now. Please check that the backend service is running.',
        timestamp: new Date(),
      }
      addMessage(errMsg)
    } finally {
      setLoading(false)
    }
  }, [
    input,
    isLoading,
    addMessage,
    setLoading,
    composition,
    frequency_mhz,
    thickness_mm,
    grain_size_um,
    analysisMode,
    calculationResult,
    sweepResult,
    messages,
  ])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-[#2a2a3e] shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-600/20 flex items-center justify-center border border-cyan-500/30">
            <Bot size={14} className="text-[#00d4ff]" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[#e8e8f0]">AI Assistant</p>
            <p className="text-xs text-[#686880]">EMI shielding expert</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-md hover:bg-[#1a1a2e] text-[#9898b0] hover:text-[#e8e8f0] transition-colors"
          aria-label="Close chat"
        >
          <ChevronRight size={16} />
        </button>
      </div>

      {/* Message list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 py-8">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-500/10 to-purple-600/10 border border-[#2a2a3e] flex items-center justify-center">
              <Bot size={20} className="text-[#9898b0]" />
            </div>
            <div className="text-center">
              <p className="text-sm text-[#9898b0] font-medium">Ask me anything</p>
              <p className="text-xs text-[#686880] mt-1 leading-relaxed">
                I can explain shielding theory, suggest materials, or interpret your results.
              </p>
            </div>
            {/* Suggestion pills */}
            <div className="flex flex-col gap-1.5 w-full">
              {[
                'Why does SE increase with frequency?',
                'Which alloy has the best SE?',
                'Explain the absorption loss formula',
              ].map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="text-left text-xs px-3 py-2 rounded-lg bg-[#12121a] border border-[#2a2a3e]
                             hover:border-[#3a3a52] text-[#9898b0] hover:text-[#e8e8f0] transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
          >
            {/* Avatar */}
            <div
              className={`w-7 h-7 rounded-lg shrink-0 flex items-center justify-center ${
                msg.role === 'user'
                  ? 'bg-gradient-to-br from-blue-600 to-indigo-700'
                  : 'bg-gradient-to-br from-cyan-500/20 to-purple-600/20 border border-cyan-500/30'
              }`}
            >
              {msg.role === 'user' ? (
                <User size={13} className="text-white" />
              ) : (
                <Bot size={13} className="text-[#00d4ff]" />
              )}
            </div>

            {/* Bubble */}
            <div
              className={`max-w-[85%] px-3 py-2 rounded-xl text-xs leading-relaxed whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-gradient-to-br from-blue-600/20 to-indigo-700/20 border border-blue-500/30 text-[#e8e8f0]'
                  : 'bg-[#12121a] border border-[#2a2a3e] text-[#e8e8f0]'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-600/20 border border-cyan-500/30 flex items-center justify-center shrink-0">
              <Bot size={13} className="text-[#00d4ff]" />
            </div>
            <div className="px-3 py-2 rounded-xl bg-[#12121a] border border-[#2a2a3e]">
              <div className="flex gap-1 items-center h-4">
                {[0, 150, 300].map((delay) => (
                  <span
                    key={delay}
                    className="w-1.5 h-1.5 rounded-full bg-[#9898b0] animate-bounce"
                    style={{ animationDelay: `${delay}ms` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="shrink-0 p-3 border-t border-[#2a2a3e]">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about EMI shielding..."
            disabled={isLoading}
            className="input-field flex-1 text-xs"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="shrink-0 w-9 h-9 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600
                       flex items-center justify-center transition-all hover:from-cyan-400 hover:to-blue-500
                       disabled:opacity-40 disabled:cursor-not-allowed active:scale-95"
            aria-label="Send message"
          >
            {isLoading ? (
              <Loader2 size={14} className="text-white animate-spin" />
            ) : (
              <Send size={14} className="text-white" />
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

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
