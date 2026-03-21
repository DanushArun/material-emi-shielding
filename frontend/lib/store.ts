import { create } from 'zustand'
import type {
  ShieldLayerConfig, CalculationResult, SweepResult,
  HistoryEntry, ChatMessage,
} from '@/types'

// Composition store
interface CompositionState {
  composition: Record<string, number>
  mode: 'direct' | 'molecular'
  setComposition: (comp: Record<string, number>) => void
  addElement: (symbol: string, percentage: number) => void
  removeElement: (symbol: string) => void
  updateElement: (symbol: string, percentage: number) => void
  clearComposition: () => void
  setMode: (mode: 'direct' | 'molecular') => void
}

export const useCompositionStore = create<CompositionState>((set) => ({
  composition: {},
  mode: 'direct',
  setComposition: (comp) => set({ composition: comp }),
  addElement: (symbol, percentage) =>
    set((s) => ({ composition: { ...s.composition, [symbol]: percentage } })),
  removeElement: (symbol) =>
    set((s) => {
      const next = { ...s.composition }
      delete next[symbol]
      return { composition: next }
    }),
  updateElement: (symbol, percentage) =>
    set((s) => ({ composition: { ...s.composition, [symbol]: percentage } })),
  clearComposition: () => set({ composition: {} }),
  setMode: (mode) => set({ mode }),
}))

// Simulation parameters store
interface SimulationState {
  frequency_mhz: number
  thickness_mm: number
  grain_size_um: number | null
  temperature_k: number
  analysisMode: 'single' | 'frequency-sweep' | 'thickness-sweep' | 'grain-sweep' | 'cooling-sweep' | 'optimize'
  setFrequency: (f: number) => void
  setThickness: (t: number) => void
  setGrainSize: (g: number | null) => void
  setTemperature: (t: number) => void
  setAnalysisMode: (m: SimulationState['analysisMode']) => void
}

export const useSimulationStore = create<SimulationState>((set) => ({
  frequency_mhz: 1000,
  thickness_mm: 1.0,
  grain_size_um: null,
  temperature_k: 293.15,
  analysisMode: 'single',
  setFrequency: (f) => set({ frequency_mhz: f }),
  setThickness: (t) => set({ thickness_mm: t }),
  setGrainSize: (g) => set({ grain_size_um: g }),
  setTemperature: (t) => set({ temperature_k: t }),
  setAnalysisMode: (m) => set({ analysisMode: m }),
}))

// Results store
interface ResultsState {
  calculationResult: CalculationResult | null
  sweepResult: SweepResult | null
  isLoading: boolean
  error: string | null
  setCalculationResult: (r: CalculationResult | null) => void
  setSweepResult: (r: SweepResult | null) => void
  setLoading: (l: boolean) => void
  setError: (e: string | null) => void
}

export const useResultsStore = create<ResultsState>((set) => ({
  calculationResult: null,
  sweepResult: null,
  isLoading: false,
  error: null,
  setCalculationResult: (r) => set({ calculationResult: r }),
  setSweepResult: (r) => set({ sweepResult: r }),
  setLoading: (l) => set({ isLoading: l }),
  setError: (e) => set({ error: e }),
}))

// History store
interface HistoryState {
  entries: HistoryEntry[]
  addEntry: (entry: HistoryEntry) => void
  clearHistory: () => void
}

export const useHistoryStore = create<HistoryState>((set) => ({
  entries: [],
  addEntry: (entry) =>
    set((s) => ({ entries: [entry, ...s.entries].slice(0, 100) })),
  clearHistory: () => set({ entries: [] }),
}))

// Chat store
interface ChatState {
  messages: ChatMessage[]
  isOpen: boolean
  isLoading: boolean
  addMessage: (msg: ChatMessage) => void
  setOpen: (open: boolean) => void
  setLoading: (l: boolean) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isOpen: false,
  isLoading: false,
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setOpen: (open) => set({ isOpen: open }),
  setLoading: (l) => set({ isLoading: l }),
  clearMessages: () => set({ messages: [] }),
}))
