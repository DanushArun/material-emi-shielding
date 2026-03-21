'use client'

import { useWorkbenchStore, WorkbenchState } from '@/lib/store'
import { Button } from '@/components/ui/Button'
import { BookOpen, Plane, Car, Smartphone, ShieldAlert, LucideIcon } from 'lucide-react'
import { useState } from 'react'

export function PresetsMenu() {
  const { loadPreset } = useWorkbenchStore()
  const [isOpen, setIsOpen] = useState(false)

  type Preset = { 
    id: string, 
    label: string, 
    icon: LucideIcon, 
    description: string, 
    data: Partial<WorkbenchState> 
  }
  const presets: Preset[] = [
    {
      id: 'aerospace',
      label: 'Aerospace: Ku-Band Composite',
      icon: Plane,
      description: 'Multi-layer MXene/Ni/Cu shield for satellite avionics (12-18 GHz).',
      data: {
        composition: { 'Ti3C2Tx': 60, 'Ni': 30, 'Cu': 10 },
        thickness: 0.5,
        sweepMode: 'frequency',
        sweepStart: 12000,
        sweepEnd: 18000,
        sweepPoints: 200,
        activeNode: 'sweep'
      }
    },
    {
      id: 'automotive',
      label: 'EV: High-Voltage Crosstalk',
      icon: Car,
      description: 'Crosstalk analysis between HV traction cable and low-voltage CAN bus.',
      data: {
        cableLength: 3.5,
        wireSeparation: 0.02,
        sweepMode: 'frequency',
        sweepStart: 0.1,
        sweepEnd: 100,
        sweepPoints: 150,
        activeNode: 'cables'
      }
    },
    {
      id: 'mobile',
      label: 'Consumer: 5G Smartphone Film',
      icon: Smartphone,
      description: 'Ultra-thin sputtered Cu/Ni film for internal interference mitigation.',
      data: {
        composition: { 'Cu': 85, 'Ni': 15 },
        thickness: 0.05,
        sweepMode: 'heatmap',
        sweepStart: 2000,
        sweepEnd: 6000,
        heatmapStartThickness: 0.01,
        heatmapEndThickness: 0.1,
        sweepPoints: 30,
        activeNode: 'sweep'
      }
    },
    {
      id: 'mil',
      label: 'Defense: HEMP Hardening',
      icon: ShieldAlert,
      description: 'MIL-STD-461G high-altitude EMP protection for ruggedized enclosures.',
      data: {
        composition: { 'Fe': 95, 'Ni': 5 },
        thickness: 2.0,
        grainSize: 15,
        sweepMode: 'frequency',
        sweepStart: 1,
        sweepEnd: 1000,
        sweepPoints: 500,
        activeNode: 'sweep'
      }
    }
  ]

  const handleSelect = (p: Preset) => {
    loadPreset(p.data)
    setIsOpen(false)
  }

  return (
    <div className="relative">
      <Button 
        variant="secondary" 
        size="sm" 
        className="gap-2 border-accent-selection/30 hover:border-accent-selection"
        onClick={() => setIsOpen(!isOpen)}
      >
        <BookOpen className="w-3.5 h-3.5 text-accent-selection" />
        Industry Templates
      </Button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full left-0 mt-1 w-80 bg-bg-panel border border-border-panel rounded shadow-2xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
            <div className="px-3 py-2 bg-bg-subpanel border-b border-border-panel">
              <span className="text-[10px] font-bold uppercase tracking-widest text-text-secondary">Load Engineering Benchmark</span>
            </div>
            <div className="flex flex-col">
              {presets.map((p: Preset) => (
                <button
                  key={p.id}
                  onClick={() => handleSelect(p)}
                  className="flex flex-col gap-1 p-3 text-left hover:bg-bg-subpanel transition-colors border-b border-border-panel last:border-0 group"
                >
                  <div className="flex items-center gap-2">
                    <p.icon className="w-3.5 h-3.5 text-accent-selection group-hover:scale-110 transition-transform" />
                    <span className="text-xs font-semibold text-text-primary">{p.label}</span>
                  </div>
                  <p className="text-[10px] text-text-secondary leading-normal">{p.description}</p>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
