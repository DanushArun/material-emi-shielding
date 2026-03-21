'use client'

import { useWorkbenchStore } from '@/lib/store'
import { useState } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/Button'

export function PropertyEditor() {
  const { activeNode } = useWorkbenchStore()

  return (
    <div className="flex flex-col gap-6">
      {activeNode === 'enclosure' && <div className="text-xs text-text-muted">Select Geometry or Materials below.</div>}
      {activeNode === 'geometry' && <GeometryProperties />}
      {activeNode === 'materials' && <MaterialProperties />}
      {activeNode === 'cables' && <CableProperties />}
      {activeNode === 'hazards' && <HazardProperties />}
      {activeNode === 'analysis' && <AnalysisProperties />}
      {activeNode === 'sweep' && <SweepProperties />}
      {activeNode === 'results' && <ResultProperties />}
    </div>
  )
}

function CableProperties() {
  const { cableLength, setCableLength, wireSeparation, setWireSeparation } = useWorkbenchStore()
  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-xs font-semibold text-text-primary uppercase tracking-widest border-b border-border-panel pb-2">Cable Harness Config</h3>
      <div className="bg-bg-subpanel border border-border-panel p-2 rounded text-[10px] text-text-secondary italic mb-2">
        Define multi-conductor transmission line parameters for crosstalk calculation.
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-xs text-text-secondary">Cable Run Length (m)</label>
        <input 
          type="number" 
          value={cableLength}
          onChange={e => setCableLength(parseFloat(e.target.value))}
          className="input-engineering" 
          step="0.1"
        />
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-xs text-text-secondary">Wire Separation (m)</label>
        <input 
          type="number" 
          value={wireSeparation}
          onChange={e => setWireSeparation(parseFloat(e.target.value))}
          className="input-engineering" 
          step="0.01"
        />
      </div>
    </div>
  )
}

function HazardProperties() {
  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-xs font-semibold text-text-primary uppercase tracking-widest border-b border-border-panel pb-2">Environmental Hazards</h3>
      <div className="flex flex-col gap-2">
        <label className="text-xs text-text-secondary">Hazard Type</label>
        <select className="input-engineering cursor-pointer">
          <option>EMP (Electromagnetic Pulse)</option>
          <option>Lightning Indirect Effects</option>
          <option>HIRF (High Intensity Radiated Field)</option>
        </select>
      </div>
      <div className="text-xs text-text-muted mt-2 border border-dashed border-border-panel p-2 rounded">
        Feature under development. For now, use Analysis Setup for frequency sweeps.
      </div>
    </div>
  )
}

function GeometryProperties() {
  const { thickness, setThickness, grainSize, setGrainSize } = useWorkbenchStore()
  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-xs font-semibold text-text-primary uppercase tracking-widest border-b border-border-panel pb-2">Physical Dimensions</h3>
      <div className="flex flex-col gap-2">
        <label className="text-xs text-text-secondary">Shield Thickness (mm)</label>
        <input 
          type="number" 
          value={thickness}
          onChange={e => setThickness(parseFloat(e.target.value))}
          className="input-engineering" 
          step="0.1"
        />
      </div>
      <div className="flex flex-col gap-2">
        <label className="text-xs text-text-secondary">Grain Size (µm) <span className="text-text-muted italic">- Optional</span></label>
        <input 
          type="number" 
          value={grainSize || ''}
          onChange={e => setGrainSize(e.target.value ? parseFloat(e.target.value) : undefined)}
          className="input-engineering" 
        />
      </div>
    </div>
  )
}

function MaterialProperties() {
  const { composition, addElement, removeElement } = useWorkbenchStore()
  const [newSymbol, setNewSymbol] = useState('')
  const [newPercent, setNewPercent] = useState('')

  const handleAdd = () => {
    if (newSymbol && newPercent) {
      addElement(newSymbol, parseFloat(newPercent))
      setNewSymbol('')
      setNewPercent('')
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-xs font-semibold text-text-primary uppercase tracking-widest border-b border-border-panel pb-2">Composite Definition</h3>
      
      <div className="bg-bg-canvas border border-border-panel rounded text-xs flex flex-col">
        <div className="grid grid-cols-3 bg-bg-subpanel border-b border-border-panel px-2 py-1 font-semibold text-text-secondary">
          <span>Element</span>
          <span>Mass %</span>
          <span className="text-right">Action</span>
        </div>
        {Object.entries(composition).map(([symbol, pct]) => (
          <div key={symbol} className="grid grid-cols-3 px-2 py-1.5 border-b border-border-panel last:border-0 items-center">
            <span className="font-mono text-accent-selection font-bold">{symbol}</span>
            <span className="text-text-primary">{pct}</span>
            <button onClick={() => removeElement(symbol)} className="text-text-muted hover:text-accent-danger flex justify-end">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>

      <div className="flex gap-2 items-end">
        <div className="flex flex-col gap-1 flex-1">
          <label className="text-[10px] text-text-secondary">Symbol</label>
          <input value={newSymbol} onChange={e => setNewSymbol(e.target.value)} className="input-engineering" />
        </div>
        <div className="flex flex-col gap-1 flex-1">
          <label className="text-[10px] text-text-secondary">%</label>
          <input value={newPercent} onChange={e => setNewPercent(e.target.value)} type="number" className="input-engineering" />
        </div>
        <Button size="sm" onClick={handleAdd} className="h-[26px]">
          <Plus className="w-3 h-3" />
        </Button>
      </div>
    </div>
  )
}

function AnalysisProperties() {
  const { frequency, setFrequency } = useWorkbenchStore()
  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-xs font-semibold text-text-primary uppercase tracking-widest border-b border-border-panel pb-2">Single Point Solver</h3>
      <div className="flex flex-col gap-2">
        <label className="text-xs text-text-secondary">Test Frequency (MHz)</label>
        <input 
          type="number" 
          value={frequency}
          onChange={e => setFrequency(parseFloat(e.target.value))}
          className="input-engineering" 
        />
      </div>
    </div>
  )
}

function SweepProperties() {
  const { 
    sweepMode, setSweepMode, 
    sweepStart, setSweepStart, sweepEnd, setSweepEnd, sweepPoints, setSweepPoints,
    heatmapStartThickness, setHeatmapStartThickness, heatmapEndThickness, setHeatmapEndThickness
  } = useWorkbenchStore()

  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-xs font-semibold text-text-primary uppercase tracking-widest border-b border-border-panel pb-2">3. Parametric Analysis</h3>
      
      <div className="bg-bg-subpanel border border-border-panel p-2 rounded text-[10px] text-text-secondary italic mb-2">
        Generate large datasets by sweeping physical parameters across a defined range.
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-xs text-text-secondary">Analysis Type</label>
        <select 
          value={sweepMode}
          onChange={e => setSweepMode(e.target.value as 'frequency' | 'thickness' | 'heatmap')}
          className="input-engineering cursor-pointer"
        >
          <option value="frequency">1D: Frequency Sweep (MHz)</option>
          <option value="thickness">1D: Thickness Sweep (mm)</option>
          <option value="heatmap">2D: Full Spectrum Heatmap</option>
        </select>
      </div>

      <div className="flex gap-2">
        <div className="flex flex-col gap-1 flex-1">
          <label className="text-[10px] text-text-secondary">
            {sweepMode === 'thickness' ? 'Start (mm)' : 'Start Freq (MHz)'}
          </label>
          <input value={sweepStart} onChange={e => setSweepStart(parseFloat(e.target.value))} type="number" className="input-engineering" />
        </div>
        <div className="flex flex-col gap-1 flex-1">
          <label className="text-[10px] text-text-secondary">
            {sweepMode === 'thickness' ? 'End (mm)' : 'End Freq (MHz)'}
          </label>
          <input value={sweepEnd} onChange={e => setSweepEnd(parseFloat(e.target.value))} type="number" className="input-engineering" />
        </div>
      </div>

      {sweepMode === 'heatmap' && (
        <div className="flex gap-2 pt-2 border-t border-border-panel">
          <div className="flex flex-col gap-1 flex-1">
            <label className="text-[10px] text-text-secondary">Start Thick. (mm)</label>
            <input value={heatmapStartThickness} onChange={e => setHeatmapStartThickness(parseFloat(e.target.value))} type="number" className="input-engineering" />
          </div>
          <div className="flex flex-col gap-1 flex-1">
            <label className="text-[10px] text-text-secondary">End Thick. (mm)</label>
            <input value={heatmapEndThickness} onChange={e => setHeatmapEndThickness(parseFloat(e.target.value))} type="number" className="input-engineering" />
          </div>
        </div>
      )}

      <div className="flex flex-col gap-1">
        <label className="text-[10px] text-text-secondary">Resolution (Points)</label>
        <input value={sweepPoints} onChange={e => setSweepPoints(parseFloat(e.target.value))} type="number" className="input-engineering" />
        {sweepMode === 'heatmap' && <span className="text-[9px] text-accent-warning">Warning: 2D grids calculate N×N points. High resolution may take longer.</span>}
      </div>
    </div>
  )
}

function ResultProperties() {
  const { singleResult, sweepResult } = useWorkbenchStore()
  
  if (!singleResult && !sweepResult) return <div className="text-xs text-text-muted">No results available. Run a solver first.</div>

  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-xs font-semibold text-text-primary uppercase tracking-widest border-b border-border-panel pb-2">Solution Data</h3>
      {singleResult && (
        <div className="flex flex-col gap-2">
          <div className="flex justify-between border-b border-border-panel py-1">
            <span className="text-text-secondary text-xs">Total SE</span>
            <span className="text-accent-success font-mono font-bold text-xs">{singleResult.shielding_effectiveness_db.toFixed(2)} dB</span>
          </div>
          <div className="flex justify-between border-b border-border-panel py-1">
            <span className="text-text-secondary text-xs">Skin Depth</span>
            <span className="text-text-primary font-mono text-xs">{singleResult.skin_depth_um.toFixed(2)} µm</span>
          </div>
          <div className="flex justify-between border-b border-border-panel py-1">
            <span className="text-text-secondary text-xs">Solve Time</span>
            <span className="text-text-muted font-mono text-xs">{singleResult.execution_time_ms} ms</span>
          </div>
        </div>
      )}
      {sweepResult && (
        <div className="flex flex-col gap-2">
           <div className="flex justify-between border-b border-border-panel py-1">
            <span className="text-text-secondary text-xs">Data Points</span>
            <span className="text-text-primary font-mono text-xs">{sweepResult.total_se_db.length}</span>
          </div>
           <div className="flex justify-between border-b border-border-panel py-1">
            <span className="text-text-secondary text-xs">Max SE (Peak)</span>
            <span className="text-accent-success font-mono font-bold text-xs">{Math.max(...sweepResult.total_se_db).toFixed(2)} dB</span>
          </div>
        </div>
      )}
    </div>
  )
}
