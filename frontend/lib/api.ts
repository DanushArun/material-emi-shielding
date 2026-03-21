import axios from 'axios'
import type {
  ElementData, AlloyData, CompositeProperties,
  CalculationRequest, CalculationResult,
  FrequencySweepRequest, SweepResult, OptimizationResult,
  HeatmapRequest, HeatmapResult
} from '@/types'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Materials
export async function fetchElements(): Promise<ElementData[]> {
  const { data } = await api.get('/api/v1/materials/elements')
  return data
}

export async function fetchElement(symbol: string): Promise<ElementData> {
  const { data } = await api.get(`/api/v1/materials/elements/${symbol}`)
  return data
}

export async function fetchAlloys(): Promise<AlloyData[]> {
  const { data } = await api.get('/api/v1/materials/alloys')
  return data
}

export async function calculateCompositeProperties(
  elements: Record<string, number>
): Promise<CompositeProperties> {
  const { data } = await api.post('/api/v1/materials/composite-properties', { elements })
  return data
}

// Physics calculations
export async function calculateSE(req: CalculationRequest): Promise<CalculationResult> {
  const { data } = await api.post('/api/v1/physics/calculate', req)
  return data
}

// Analysis sweeps
export async function generateHeatmap(req: HeatmapRequest): Promise<HeatmapResult> {
  const { data } = await api.post('/api/v1/heatmap/generate', req)
  return data
}

export async function frequencySweep(req: FrequencySweepRequest): Promise<SweepResult> {
  const { data } = await api.post('/api/v1/analysis/frequency-sweep', req)
  return data
}

export async function calculateCrosstalk(req: {
  cable_length_m: number
  wire_separation_m: number
  freq_start_mhz: number
  freq_end_mhz: number
  num_points: number
}): Promise<{ frequencies_mhz: number[], next_db: number[], fext_db: number[] }> {
  const { data } = await api.post('/api/v1/cables/crosstalk', req)
  return data
}

export async function thicknessSweep(params: {
  composition: Record<string, number>
  frequency_mhz: number
  thickness_start_mm: number
  thickness_end_mm: number
  num_points: number
  grain_size_um?: number
}): Promise<SweepResult> {
  const { data } = await api.post('/api/v1/analysis/thickness-sweep', params)
  return data
}

export async function grainSizeSweep(params: {
  composition: Record<string, number>
  frequency_mhz: number
  thickness_mm: number
  grain_start_um: number
  grain_end_um: number
  num_points: number
}): Promise<SweepResult> {
  const { data } = await api.post('/api/v1/analysis/grain-size-sweep', params)
  return data
}

export async function coolingRateSweep(params: {
  composition: Record<string, number>
  frequency_mhz: number
  thickness_mm: number
  cooling_rate_min?: number
  cooling_rate_max?: number
  num_points?: number
}): Promise<SweepResult> {
  const { data } = await api.post('/api/v1/analysis/cooling-rate-sweep', params)
  return data
}

export async function optimizeThickness(params: {
  composition: Record<string, number>
  frequency_mhz: number
  target_se_db: number
  thickness_max_mm?: number
  grain_size_um?: number
}): Promise<OptimizationResult> {
  const { data } = await api.post('/api/v1/analysis/optimize-thickness', params)
  return data
}

// AI Chat
export async function sendChatMessage(
  message: string,
  history: { role: string; content: string }[],
  context?: Record<string, unknown>,
): Promise<{ response: string; suggestions: string[] }> {
  const { data } = await api.post('/api/v1/chat/message', { message, history, context })
  return data
}

// Health check
export async function checkHealth(): Promise<{ status: string; version: string }> {
  const { data } = await api.get('/health')
  return data
}
