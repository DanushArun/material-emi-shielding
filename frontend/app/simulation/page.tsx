'use client'

import { WorkbenchLayout } from '@/components/layout/WorkbenchLayout'
import { ProjectTree } from '@/components/simulation/ProjectTree'
import { PropertyEditor } from '@/components/simulation/PropertyEditor'
import { ViewportCanvas } from '@/components/simulation/ViewportCanvas'
import { ScientificChart } from '@/components/simulation/ScientificChart'
import { AIAssistantPanel } from '@/components/simulation/AIAssistantPanel'
import { useWorkbenchStore } from '@/lib/store'
import { Button } from '@/components/ui/Button'
import { Play, Activity, Download } from 'lucide-react'
import { calculateSE, frequencySweep, thicknessSweep } from '@/lib/api'

export default function SimulationPage() {
  const { 
    activeNode, setActiveNode, 
    isCalculating, setIsCalculating,
    composition, thickness, frequency, grainSize,
    sweepMode, sweepStart, sweepEnd, sweepPoints,
    setSingleResult, setSweepResult
  } = useWorkbenchStore()

  const handleSolve = async () => {
    setIsCalculating(true)
    try {
      if (activeNode === 'sweep' || activeNode === 'results') {
        // Run Sweep
        let res
        if (sweepMode === 'frequency') {
          res = await frequencySweep({
            composition, thickness_mm: thickness, grain_size_um: grainSize,
            freq_start_mhz: sweepStart, freq_end_mhz: sweepEnd, num_points: sweepPoints
          })
        } else {
          res = await thicknessSweep({
            composition, frequency_mhz: frequency, grain_size_um: grainSize,
            thickness_start_mm: sweepStart, thickness_end_mm: sweepEnd, num_points: sweepPoints
          })
        }
        setSweepResult(res)
        setActiveNode('results')
      } else {
        // Run Single Point
        const res = await calculateSE({
          composition, thickness_mm: thickness, frequency_mhz: frequency, grain_size_um: grainSize
        })
        setSingleResult(res)
        setActiveNode('results')
      }
    } catch (err) {
      console.error(err)
    } finally {
      setIsCalculating(false)
    }
  }

  const ribbon = (
    <div className="w-full flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" className="gap-2">File</Button>
        <Button variant="ghost" size="sm" className="gap-2">Edit</Button>
        <Button variant="ghost" size="sm" className="gap-2">View</Button>
        <div className="w-px h-4 bg-border-panel mx-2" />
        <Button variant="secondary" size="sm" className="gap-2" onClick={() => setActiveNode('sweep')}>
          <Activity className="w-3 h-3" /> Configure Sweep
        </Button>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="secondary" size="sm" className="gap-2 text-text-muted hover:text-text-primary">
          <Download className="w-3 h-3" /> Export Data
        </Button>
        <Button variant="primary" size="sm" className="gap-2 bg-accent-success hover:bg-[#2ea043]" onClick={handleSolve} disabled={isCalculating}>
          <Play className="w-3 h-3 fill-current" />
          {isCalculating ? 'Solving...' : 'Solve'}
        </Button>
      </div>
    </div>
  )

  const viewport = activeNode === 'results' || activeNode === 'sweep' ? <ScientificChart /> : <ViewportCanvas />

  return (
    <WorkbenchLayout
      ribbon={ribbon}
      tree={<ProjectTree />}
      viewport={viewport}
      properties={<PropertyEditor />}
      consolePanel={<AIAssistantPanel />}
    />
  )
}
