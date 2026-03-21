export interface HeatmapRequest {
  composition: Record<string, number>
  freq_start_mhz: number
  freq_end_mhz: number
  thickness_start_mm: number
  thickness_end_mm: number
  num_points?: number
}

export interface HeatmapResult {
  frequencies_mhz: number[]
  thicknesses_mm: number[]
  se_matrix_db: number[][]
}