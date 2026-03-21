'use client'

import dynamic from 'next/dynamic'
import { useResultsStore, useSimulationStore } from '@/lib/store'
import type { CalculationResult, SweepResult, OptimizationResult } from '@/types'
import { Activity, Layers, Zap, Clock, TrendingUp } from 'lucide-react'
import type { Layout, Config, Data } from 'plotly.js'

// Plotly loaded client-side only to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

// ---- Shared dark Plotly layout ----

const DARK_LAYOUT: Partial<Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#9898b0', family: 'Inter, sans-serif', size: 11 },
  margin: { t: 24, r: 16, b: 48, l: 56 },
  xaxis: {
    gridcolor: '#2a2a3e',
    linecolor: '#2a2a3e',
    zerolinecolor: '#2a2a3e',
    tickfont: { color: '#9898b0', size: 10 },
    title: { text: '', font: { color: '#9898b0', size: 11 } },
  },
  yaxis: {
    gridcolor: '#2a2a3e',
    linecolor: '#2a2a3e',
    zerolinecolor: '#2a2a3e',
    tickfont: { color: '#9898b0', size: 10 },
    title: { text: '', font: { color: '#9898b0', size: 11 } },
  },
  legend: {
    bgcolor: 'transparent',
    font: { color: '#9898b0', size: 11 },
    orientation: 'h',
    yanchor: 'bottom',
    y: 1.02,
    xanchor: 'right',
    x: 1,
  },
  hoverlabel: {
    bgcolor: '#16161f',
    bordercolor: '#2a2a3e',
    font: { color: '#e8e8f0', size: 12 },
  },
}

const PLOT_CONFIG: Partial<Config> = {
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'],
  responsive: true,
}

// ---- Metric card ----

interface MetricCardProps {
  label: string
  value: number | string
  unit: string
  icon: React.ReactNode
  accent: 'cyan' | 'purple' | 'green' | 'amber'
  sub?: string
}

const ACCENT_STYLES = {
  cyan: {
    glow: 'shadow-cyan-500/20',
    text: 'text-[#00d4ff]',
    border: 'border-cyan-500/30',
    icon: 'text-[#00d4ff] bg-cyan-500/10',
  },
  purple: {
    glow: 'shadow-purple-500/20',
    text: 'text-[#8b5cf6]',
    border: 'border-purple-500/30',
    icon: 'text-[#8b5cf6] bg-purple-500/10',
  },
  green: {
    glow: 'shadow-green-500/20',
    text: 'text-[#10b981]',
    border: 'border-green-500/30',
    icon: 'text-[#10b981] bg-green-500/10',
  },
  amber: {
    glow: 'shadow-amber-500/20',
    text: 'text-[#f59e0b]',
    border: 'border-amber-500/30',
    icon: 'text-[#f59e0b] bg-amber-500/10',
  },
}

function MetricCard({ label, value, unit, icon, accent, sub }: MetricCardProps) {
  const s = ACCENT_STYLES[accent]
  const displayValue =
    typeof value === 'number'
      ? Math.abs(value) >= 1000
        ? value.toFixed(0)
        : Math.abs(value) >= 10
        ? value.toFixed(1)
        : value.toFixed(2)
      : value

  return (
    <div
      className={`card p-4 flex gap-3 items-start shadow-lg ${s.glow} border ${s.border} transition-all duration-300`}
    >
      <div className={`rounded-lg p-2 shrink-0 ${s.icon}`}>{icon}</div>
      <div className="min-w-0">
        <p className="text-xs text-[#9898b0] mb-1 truncate">{label}</p>
        <p className={`text-2xl font-bold font-mono leading-none ${s.text}`}>
          {displayValue}
          <span className="text-sm font-normal text-[#9898b0] ml-1">{unit}</span>
        </p>
        {sub && <p className="text-xs text-[#686880] mt-1 truncate">{sub}</p>}
      </div>
    </div>
  )
}

// ---- Sweep chart ----

interface SweepChartProps {
  result: SweepResult
  mode: string
}

function SweepChart({ result, mode }: SweepChartProps) {
  const xKey =
    mode === 'frequency-sweep'
      ? 'frequencies_mhz'
      : mode === 'thickness-sweep'
      ? 'thicknesses_mm'
      : mode === 'grain-sweep'
      ? 'grain_sizes_um'
      : 'cooling_rates'

  const xData = (result[xKey as keyof SweepResult] as number[] | undefined) ?? []

  const xLabel =
    mode === 'frequency-sweep'
      ? 'Frequency (MHz)'
      : mode === 'thickness-sweep'
      ? 'Thickness (mm)'
      : mode === 'grain-sweep'
      ? 'Grain Size (um)'
      : 'Cooling Rate (K/s)'

  const traces: Data[] = [
    {
      x: xData,
      y: result.total_se_db,
      name: 'Total SE',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#00d4ff', width: 2 },
      fill: 'tozeroy',
      fillcolor: 'rgba(0,212,255,0.06)',
    },
    {
      x: xData,
      y: result.reflection_loss_db,
      name: 'Reflection',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#8b5cf6', width: 1.5, dash: 'dot' },
    },
    {
      x: xData,
      y: result.absorption_loss_db,
      name: 'Absorption',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#10b981', width: 1.5, dash: 'dash' },
    },
  ]

  const layout: Partial<Layout> = {
    ...DARK_LAYOUT,
    xaxis: {
      ...DARK_LAYOUT.xaxis,
      title: { text: xLabel, font: { color: '#9898b0', size: 11 } },
      type: mode === 'frequency-sweep' || mode === 'cooling-sweep' ? 'log' : 'linear',
    },
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      title: { text: 'Shielding Effectiveness (dB)', font: { color: '#9898b0', size: 11 } },
    },
  }

  return (
    <div className="card p-4">
      <h3 className="text-xs font-semibold text-[#9898b0] uppercase tracking-wide mb-3 flex items-center gap-2">
        <Activity size={13} />
        Sweep Results
      </h3>
      <div className="w-full" style={{ height: 320 }}>
        <Plot
          data={traces}
          layout={layout}
          config={PLOT_CONFIG}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  )
}

// ---- Single-point result breakdown ----

interface SingleResultProps {
  result: CalculationResult
}

function SingleResult({ result }: SingleResultProps) {
  const confidenceColor =
    result.confidence_level === 'high'
      ? 'text-[#10b981]'
      : result.confidence_level === 'medium'
      ? 'text-[#f59e0b]'
      : 'text-[#ef4444]'

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <MetricCard
        label="Total Shielding Effectiveness"
        value={result.shielding_effectiveness_db}
        unit="dB"
        icon={<Layers size={16} />}
        accent="cyan"
        sub={
          result.confidence_level
            ? `Confidence: ${result.confidence_level}`
            : undefined
        }
      />
      <MetricCard
        label="Reflection Loss"
        value={result.reflection_loss_db}
        unit="dB"
        icon={<Zap size={16} />}
        accent="purple"
      />
      <MetricCard
        label="Absorption Loss"
        value={result.absorption_loss_db}
        unit="dB"
        icon={<TrendingUp size={16} />}
        accent="green"
      />
      <MetricCard
        label="Skin Depth"
        value={result.skin_depth_um}
        unit="um"
        icon={<Activity size={16} />}
        accent="amber"
        sub={`Conductivity: ${result.effective_conductivity.toExponential(2)} S/m`}
      />

      {/* Extra meta */}
      <div className="sm:col-span-2 card p-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-[#9898b0]">
          <Clock size={12} />
          <span>Computed in {result.execution_time_ms.toFixed(1)} ms</span>
        </div>
        {result.confidence !== undefined && (
          <span className={`text-xs font-medium ${confidenceColor}`}>
            {(result.confidence * 100).toFixed(0)}% confidence
          </span>
        )}
        {result.multiple_reflection_db !== 0 && (
          <span className="text-xs text-[#686880]">
            MR correction: {result.multiple_reflection_db.toFixed(2)} dB
          </span>
        )}
      </div>
    </div>
  )
}

// ---- Optimization result ----

interface OptimizeResultProps {
  result: OptimizationResult
}

function OptimizeResult({ result }: OptimizeResultProps) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <MetricCard
        label="Optimal Thickness"
        value={result.optimal_thickness_mm}
        unit="mm"
        icon={<Layers size={16} />}
        accent="cyan"
        sub="Minimum to meet target SE"
      />
      <MetricCard
        label="Achieved SE"
        value={result.achieved_se_db}
        unit="dB"
        icon={<TrendingUp size={16} />}
        accent="green"
      />
      <MetricCard
        label="Reflection Loss"
        value={result.reflection_loss_db}
        unit="dB"
        icon={<Zap size={16} />}
        accent="purple"
      />
      <MetricCard
        label="Absorption Loss"
        value={result.absorption_loss_db}
        unit="dB"
        icon={<Activity size={16} />}
        accent="amber"
      />
      <div className="sm:col-span-2 card p-3 flex items-center gap-2 text-xs text-[#9898b0]">
        <Clock size={12} />
        <span>Optimized in {result.execution_time_ms.toFixed(1)} ms</span>
      </div>
    </div>
  )
}

// ---- Loading skeleton ----

function Skeleton() {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 animate-pulse">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="card p-4 h-24 bg-[#1a1a2e]" />
      ))}
    </div>
  )
}

// ---- Error banner ----

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="card p-4 border-[#ef4444]/40 bg-[#ef4444]/5">
      <p className="text-sm text-[#ef4444] font-medium">Calculation failed</p>
      <p className="text-xs text-[#9898b0] mt-1">{message}</p>
    </div>
  )
}

// ---- Idle state ----

function IdleState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-4">
      <div className="w-16 h-16 rounded-full bg-[#1a1a2e] border border-[#2a2a3e] flex items-center justify-center">
        <Activity size={24} className="text-[#3a3a52]" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-[#9898b0]">No results yet</p>
        <p className="text-xs text-[#686880] mt-1">
          Configure parameters and press Calculate
        </p>
      </div>
    </div>
  )
}

// ---- Main component ----

interface ResultsPanelProps {
  optimizationResult?: OptimizationResult | null
}

export default function ResultsPanel({ optimizationResult }: ResultsPanelProps) {
  const { calculationResult, sweepResult, isLoading, error } = useResultsStore()
  const { analysisMode } = useSimulationStore()

  if (isLoading) return <Skeleton />
  if (error) return <ErrorBanner message={error} />

  const hasSingle = calculationResult !== null
  const hasSweep = sweepResult !== null
  const hasOptimize = optimizationResult != null

  if (!hasSingle && !hasSweep && !hasOptimize) return <IdleState />

  return (
    <div className="flex flex-col gap-4">
      {/* Single point */}
      {hasSingle && analysisMode === 'single' && (
        <SingleResult result={calculationResult!} />
      )}

      {/* Sweep chart + summary cards */}
      {hasSweep && (
        <>
          <SweepChart result={sweepResult!} mode={analysisMode} />

          {/* Peak-value summary cards */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <MetricCard
              label="Peak SE"
              value={Math.max(...sweepResult!.total_se_db)}
              unit="dB"
              icon={<Layers size={16} />}
              accent="cyan"
            />
            <MetricCard
              label="Min SE"
              value={Math.min(...sweepResult!.total_se_db)}
              unit="dB"
              icon={<Activity size={16} />}
              accent="amber"
            />
            <MetricCard
              label="Avg SE"
              value={
                sweepResult!.total_se_db.reduce((s, v) => s + v, 0) /
                sweepResult!.total_se_db.length
              }
              unit="dB"
              icon={<TrendingUp size={16} />}
              accent="green"
            />
          </div>
        </>
      )}

      {/* Optimize */}
      {hasOptimize && <OptimizeResult result={optimizationResult!} />}
    </div>
  )
}
