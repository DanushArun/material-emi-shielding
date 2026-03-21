import { create } from 'zustand'
import { CalculationResult, SweepResult } from '@/types'

export type TreeNodeId = 'geometry' | 'materials' | 'analysis' | 'results' | 'sweep'

interface WorkbenchState {
  activeNode: TreeNodeId
  setActiveNode: (node: TreeNodeId) => void

  composition: Record<string, number>
  addElement: (symbol: string, percentage: number) => void
  removeElement: (symbol: string) => void

  thickness: number
  setThickness: (val: number) => void

  frequency: number
  setFrequency: (val: number) => void
  
  grainSize: number | undefined
  setGrainSize: (val: number | undefined) => void

  sweepMode: 'frequency' | 'thickness'
  setSweepMode: (mode: 'frequency' | 'thickness') => void
  sweepPoints: number
  setSweepPoints: (val: number) => void
  sweepStart: number
  setSweepStart: (val: number) => void
  sweepEnd: number
  setSweepEnd: (val: number) => void

  singleResult: CalculationResult | null
  setSingleResult: (res: CalculationResult | null) => void
  sweepResult: SweepResult | null
  setSweepResult: (res: SweepResult | null) => void

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

  singleResult: null,
  setSingleResult: (res) => set({ singleResult: res }),
  sweepResult: null,
  setSweepResult: (res) => set({ sweepResult: res }),

  isCalculating: false,
  setIsCalculating: (val) => set({ isCalculating: val }),
}))
