'use client'

import { useSimulationStore } from '@/lib/store'

// ---- Types ----

type AnalysisMode =
  | 'single'
  | 'frequency-sweep'
  | 'thickness-sweep'
  | 'grain-sweep'
  | 'cooling-sweep'
  | 'optimize'

const ANALYSIS_MODES: { value: AnalysisMode; label: string; description: string }[] = [
  {
    value: 'single',
    label: 'Single Point',
    description: 'Calculate SE at one frequency and thickness',
  },
  {
    value: 'frequency-sweep',
    label: 'Frequency Sweep',
    description: 'SE vs frequency over a configurable range',
  },
  {
    value: 'thickness-sweep',
    label: 'Thickness Sweep',
    description: 'SE vs shield thickness at fixed frequency',
  },
  {
    value: 'grain-sweep',
    label: 'Grain Size Sweep',
    description: 'SE vs grain size (microstructure effect)',
  },
  {
    value: 'cooling-sweep',
    label: 'Cooling Rate Sweep',
    description: 'SE vs manufacturing cooling rate',
  },
  {
    value: 'optimize',
    label: 'Optimize Thickness',
    description: 'Find minimum thickness for target SE',
  },
]

// ---- Reusable labeled input ----

interface LabeledInputProps {
  id: string
  label: string
  unit: string
  value: number | string
  onChange: (v: string) => void
  type?: string
  min?: number
  max?: number
  step?: number
  placeholder?: string
  hint?: string
  optional?: boolean
}

function LabeledInput({
  id,
  label,
  unit,
  value,
  onChange,
  type = 'number',
  min,
  max,
  step = 1,
  placeholder,
  hint,
  optional = false,
}: LabeledInputProps) {
  return (
    <div>
      <label htmlFor={id} className="label-text flex items-center gap-2">
        {label}
        {optional && (
          <span className="text-xs text-[#686880] font-normal">(optional)</span>
        )}
      </label>
      <div className="relative">
        <input
          id={id}
          type={type}
          min={min}
          max={max}
          step={step}
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          className="input-field pr-14 font-mono"
        />
        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-[#686880] font-mono pointer-events-none">
          {unit}
        </span>
      </div>
      {hint && <p className="mt-1 text-xs text-[#686880]">{hint}</p>}
    </div>
  )
}

// ---- Main component ----

interface ShieldParametersProps {
  /** Target SE in dB — only shown in optimize mode */
  targetSE: number
  onTargetSEChange: (v: number) => void
  /** Sweep range values lifted to parent so Calculate button can read them */
  sweepParams: SweepParams
  onSweepParamsChange: (p: Partial<SweepParams>) => void
}

export interface SweepParams {
  freqStart: number
  freqEnd: number
  thicknessStart: number
  thicknessEnd: number
  grainStart: number
  grainEnd: number
  numPoints: number
}

export default function ShieldParameters({
  targetSE,
  onTargetSEChange,
  sweepParams,
  onSweepParamsChange,
}: ShieldParametersProps) {
  const {
    frequency_mhz,
    thickness_mm,
    grain_size_um,
    analysisMode,
    setFrequency,
    setThickness,
    setGrainSize,
    setAnalysisMode,
  } = useSimulationStore()

  const mode = analysisMode as AnalysisMode
  const isSweep = mode !== 'single' && mode !== 'optimize'

  return (
    <div className="flex flex-col gap-6">
      {/* ---- Analysis mode selector ---- */}
      <div>
        <p className="label-text mb-2">Analysis Mode</p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {ANALYSIS_MODES.map((m) => (
            <button
              key={m.value}
              onClick={() => setAnalysisMode(m.value)}
              title={m.description}
              className={`px-3 py-2 rounded-lg text-xs font-medium border transition-all duration-150 text-left
                ${
                  mode === m.value
                    ? 'bg-gradient-to-r from-cyan-600/20 to-blue-600/20 border-cyan-500 text-[#00d4ff]'
                    : 'bg-[#12121a] border-[#2a2a3e] text-[#9898b0] hover:border-[#3a3a52] hover:text-[#e8e8f0]'
                }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-[#686880]">
          {ANALYSIS_MODES.find((m) => m.value === mode)?.description}
        </p>
      </div>

      <div className="border-t border-[#2a2a3e]" />

      {/* ---- Base parameters (always visible) ---- */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {/* Frequency — hidden when sweeping frequency, show as fixed otherwise */}
        {mode !== 'frequency-sweep' && (
          <LabeledInput
            id="frequency"
            label="Frequency"
            unit="MHz"
            value={frequency_mhz}
            onChange={(v) => setFrequency(parseFloat(v) || 0)}
            min={0.001}
            max={300000}
            step={100}
            hint={`${(frequency_mhz / 1000).toFixed(2)} GHz`}
          />
        )}

        {/* Thickness — hidden when sweeping thickness */}
        {mode !== 'thickness-sweep' && (
          <LabeledInput
            id="thickness"
            label="Thickness"
            unit="mm"
            value={thickness_mm}
            onChange={(v) => setThickness(parseFloat(v) || 0)}
            min={0.001}
            max={500}
            step={0.1}
          />
        )}

        {/* Grain size — hidden during grain sweep, shown as optional otherwise */}
        {mode !== 'grain-sweep' && mode !== 'cooling-sweep' && (
          <LabeledInput
            id="grain-size"
            label="Grain Size"
            unit="um"
            value={grain_size_um ?? ''}
            onChange={(v) =>
              setGrainSize(v === '' ? null : parseFloat(v) || null)
            }
            min={0.001}
            max={10000}
            step={1}
            placeholder="auto"
            optional
            hint="Microstructure correction; leave blank for bulk properties"
          />
        )}
      </div>

      {/* ---- Sweep range parameters ---- */}
      {isSweep && (
        <>
          <div className="border-t border-[#2a2a3e]" />
          <div>
            <p className="label-text mb-3">Sweep Range</p>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {mode === 'frequency-sweep' && (
                <>
                  <LabeledInput
                    id="freq-start"
                    label="Start Frequency"
                    unit="MHz"
                    value={sweepParams.freqStart}
                    onChange={(v) =>
                      onSweepParamsChange({ freqStart: parseFloat(v) || 1 })
                    }
                    min={0.001}
                    step={100}
                  />
                  <LabeledInput
                    id="freq-end"
                    label="End Frequency"
                    unit="MHz"
                    value={sweepParams.freqEnd}
                    onChange={(v) =>
                      onSweepParamsChange({ freqEnd: parseFloat(v) || 1000 })
                    }
                    min={0.001}
                    step={100}
                  />
                </>
              )}

              {mode === 'thickness-sweep' && (
                <>
                  <LabeledInput
                    id="thick-start"
                    label="Min Thickness"
                    unit="mm"
                    value={sweepParams.thicknessStart}
                    onChange={(v) =>
                      onSweepParamsChange({ thicknessStart: parseFloat(v) || 0.1 })
                    }
                    min={0.001}
                    step={0.1}
                  />
                  <LabeledInput
                    id="thick-end"
                    label="Max Thickness"
                    unit="mm"
                    value={sweepParams.thicknessEnd}
                    onChange={(v) =>
                      onSweepParamsChange({ thicknessEnd: parseFloat(v) || 10 })
                    }
                    min={0.001}
                    step={0.1}
                  />
                </>
              )}

              {mode === 'grain-sweep' && (
                <>
                  <LabeledInput
                    id="grain-start"
                    label="Min Grain Size"
                    unit="um"
                    value={sweepParams.grainStart}
                    onChange={(v) =>
                      onSweepParamsChange({ grainStart: parseFloat(v) || 1 })
                    }
                    min={0.001}
                    step={1}
                  />
                  <LabeledInput
                    id="grain-end"
                    label="Max Grain Size"
                    unit="um"
                    value={sweepParams.grainEnd}
                    onChange={(v) =>
                      onSweepParamsChange({ grainEnd: parseFloat(v) || 1000 })
                    }
                    min={0.001}
                    step={10}
                  />
                </>
              )}

              {/* cooling-sweep has no user-visible range — backend uses defaults */}
              {mode === 'cooling-sweep' && (
                <p className="col-span-2 text-xs text-[#9898b0]">
                  Cooling rate range is determined automatically (0.01 to 1000 K/s).
                </p>
              )}

              {/* Point count for all sweeps */}
              {mode !== 'cooling-sweep' && (
                <LabeledInput
                  id="num-points"
                  label="Number of Points"
                  unit="pts"
                  value={sweepParams.numPoints}
                  onChange={(v) =>
                    onSweepParamsChange({
                      numPoints: Math.max(2, parseInt(v) || 50),
                    })
                  }
                  min={2}
                  max={500}
                  step={10}
                />
              )}
            </div>
          </div>
        </>
      )}

      {/* ---- Optimize mode: target SE ---- */}
      {mode === 'optimize' && (
        <>
          <div className="border-t border-[#2a2a3e]" />
          <LabeledInput
            id="target-se"
            label="Target Shielding Effectiveness"
            unit="dB"
            value={targetSE}
            onChange={(v) => onTargetSEChange(parseFloat(v) || 30)}
            min={1}
            max={300}
            step={5}
            hint="Optimizer will find the minimum thickness to achieve this target SE"
          />
        </>
      )}
    </div>
  )
}
