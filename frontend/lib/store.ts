import { create } from 'zustand'
import { CalculationResult, SweepResult, HeatmapResult } from '@/types'

export type TreeNodeId = 'geometry' | 'materials' | 'analysis' | 'results' | 'sweep' | 'enclosure' | 'cables' | 'hazards'

interface WorkbenchState {
  activeNode: TreeNodeId
  setActiveNode: (node: TreeNodeId) => void

  // Material & Basic Geometry
  composition: Record<string, number>
  addElement: (symbol: string, percentage: number) => void
  removeElement: (symbol: string) => void

  thickness: number
  setThickness: (val: number) => void

  // Cable Harness specific
  cableLength: number
  setCableLength: (val: number) => void
  wireSeparation: number
  setWireSeparation: (val: number) => void

  // Analysis Setup
  frequency: number
  setFrequency: (val: number) => void
  
  grainSize: number | undefined
  setGrainSize: (val: number | undefined) => void

  sweepMode: 'frequency' | 'thickness' | 'heatmap'
  setSweepMode: (mode: 'frequency' | 'thickness' | 'heatmap') => void
  sweepPoints: number
  setSweepPoints: (val: number) => void
  sweepStart: number
  setSweepStart: (val: number) => void
  sweepEnd: number
  setSweepEnd: (val: number) => void

  heatmapStartThickness: number
  setHeatmapStartThickness: (val: number) => void
  heatmapEndThickness: number
  setHeatmapEndThickness: (val: number) => void

  singleResult: CalculationResult | null
  setSingleResult: (res: CalculationResult | null) => void
  sweepResult: SweepResult | null
  setSweepResult: (res: SweepResult | null) => void
  heatmapResult: HeatmapResult | null
  setHeatmapResult: (res: HeatmapResult | null) => void
  cableResult: { frequencies_mhz: number[], next_db: number[], fext_db: number[] } | null
  setCableResult: (res: { frequencies_mhz: number[], next_db: number[], fext_db: number[] } | null) => void

  isCalculating: boolean
  setIsCalculating: (val: boolean) => void
}

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  activeNode: 'materials',
  setActiveNode: (node) => set({ activeNode: node }),

  composition: { 'Cu': 100 },
  addElement: (symbol, percentage) => set((state) => ({
    composition: { ...state.composition, [symbol]: percentage }
  })),
  removeElement: (symbol) => set((state) => {
    const newComp = { ...state.composition }
    delete newComp[symbol]
    return { composition: newComp }
  }),

  thickness: 1.0,
  setThickness: (val) => set({ thickness: val }),

  cableLength: 2.0,
  setCableLength: (val) => set({ cableLength: val }),
  wireSeparation: 0.05,
  setWireSeparation: (val) => set({ wireSeparation: val }),

  frequency: 1000,
  setFrequency: (val) => set({ frequency: val }),

  grainSize: undefined,
  setGrainSize: (val) => set({ grainSize: val }),

  sweepMode: 'frequency',
  setSweepMode: (mode) => set({ sweepMode: mode }),
  sweepPoints: 100,
  setSweepPoints: (val) => set({ sweepPoints: val }),
  sweepStart: 100,
  setSweepStart: (val) => set({ sweepStart: val }),
  sweepEnd: 10000,
  setSweepEnd: (val) => set({ sweepEnd: val }),

  heatmapStartThickness: 0.1,
  setHeatmapStartThickness: (val) => set({ heatmapStartThickness: val }),
  heatmapEndThickness: 5.0,
  setHeatmapEndThickness: (val) => set({ heatmapEndThickness: val }),

  singleResult: null,
  setSingleResult: (res) => set({ singleResult: res }),
  sweepResult: null,
  setSweepResult: (res) => set({ sweepResult: res }),
  heatmapResult: null,
  setHeatmapResult: (res) => set({ heatmapResult: res }),
  cableResult: null,
  setCableResult: (res) => set({ cableResult: res }),

  isCalculating: false,
  setIsCalculating: (val) => set({ isCalculating: val }),
}))
