// Material and element types
export interface ElementData {
  symbol: string
  name: string
  atomic_number: number
  atomic_weight: number
  electrical_conductivity: number
  relative_permeability: number
  density: number
  category?: string
}

export interface AlloyData {
  key: string
  name: string
  composition: Record<string, number>
  density: number
  electrical_conductivity: number
  relative_permeability: number
  relative_permittivity: number
  note?: string
}

export interface CompositeProperties {
  conductivity: number
  permeability: number
  permittivity: number
  density: number
  composition: Record<string, number>
}

// Calculation types
export interface CalculationRequest {
  composition: Record<string, number>
  frequency_mhz: number
  thickness_mm: number
  grain_size_um?: number
}

export interface CalculationResult {
  shielding_effectiveness_db: number
  reflection_loss_db: number
  absorption_loss_db: number
  multiple_reflection_db: number
  skin_depth_um: number
  effective_conductivity: number
  execution_time_ms: number
  confidence?: number
  confidence_level?: string
}

// Sweep types
export interface FrequencySweepRequest {
  composition: Record<string, number>
  thickness_mm: number
  freq_start_mhz: number
  freq_end_mhz: number
  num_points: number
  grain_size_um?: number
}

export interface SweepResult {
  frequencies_mhz?: number[]
  thicknesses_mm?: number[]
  grain_sizes_um?: number[]
  cooling_rates?: number[]
  total_se_db: number[]
  reflection_loss_db: number[]
  absorption_loss_db: number[]
  skin_depth_um?: number[]
  effective_conductivities?: number[]
  execution_time_ms: number
}

export interface OptimizationResult {
  optimal_thickness_mm: number
  achieved_se_db: number
  reflection_loss_db: number
  absorption_loss_db: number
  execution_time_ms: number
}

// Shield layer for multilayer designer
export interface ShieldLayerConfig {
  id: string
  name: string
  composition: Record<string, number>
  thickness_mm: number
  conductivity?: number
  permeability?: number
  permittivity?: number
}

// Chat message
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

// Calculation history entry
export interface HistoryEntry {
  id: string
  timestamp: Date
  composition: Record<string, number>
  compositionSummary: string
  thickness_mm: number
  frequency_mhz: number
  total_se: number
  mode: string
}
