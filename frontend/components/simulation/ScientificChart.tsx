'use client'

import dynamic from 'next/dynamic'
import { useWorkbenchStore } from '@/lib/store'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false, loading: () => <div className="animate-pulse bg-bg-panel w-full h-full rounded border border-border-panel flex items-center justify-center text-text-muted text-xs">Loading Plotly Engine...</div> })

export function ScientificChart() {
  const { sweepResult, heatmapResult, cableResult, sweepMode } = useWorkbenchStore()

  if (!sweepResult && !heatmapResult && !cableResult) {
    return (
      <div className="w-full h-full flex items-center justify-center border border-dashed border-border-panel bg-bg-panel/50 rounded flex-col gap-2">
        <span className="text-text-muted text-sm font-medium">No simulation data available</span>
        <span className="text-text-secondary text-xs">Run a parametric sweep, heatmap, or cable crosstalk solver to generate visualization.</span>
      </div>
    )
  }

  // CABLE CROSSTALK PLOT
  if (cableResult) {
    return (
      <div className="w-full h-full bg-bg-canvas border border-border-panel rounded overflow-hidden relative">
        <div className="absolute top-4 left-6 z-10 bg-bg-panel/80 backdrop-blur-sm border border-border-panel px-3 py-1.5 rounded text-[10px] font-mono text-text-primary shadow-lg">
          Cable Harness Analysis: Near-End (NEXT) & Far-End (FEXT) Crosstalk
        </div>
        <Plot
          data={[
            {
              x: cableResult.frequencies_mhz,
              y: cableResult.next_db,
              type: 'scatter',
              mode: 'lines',
              name: 'NEXT (Near-End)',
              line: { color: '#da3633', width: 2 },
            },
            {
              x: cableResult.frequencies_mhz,
              y: cableResult.fext_db,
              type: 'scatter',
              mode: 'lines',
              name: 'FEXT (Far-End)',
              line: { color: '#005fb8', width: 2, dash: 'dash' },
            }
          ]}
          layout={{
            autosize: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#8b8b99', family: 'Inter' },
            xaxis: { title: 'Frequency (MHz)', gridcolor: '#2d2d33', zerolinecolor: '#2d2d33' },
            yaxis: { title: 'Crosstalk (dB)', gridcolor: '#2d2d33', zerolinecolor: '#2d2d33' },
            margin: { t: 40, r: 20, b: 40, l: 60 },
            legend: { orientation: 'h', y: 1.1 },
            hovermode: 'x unified'
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    )
  }

  // HEATMAP PLOT
  if (heatmapResult && sweepMode === 'heatmap') {
    return (
      <div className="w-full h-full bg-bg-canvas border border-border-panel rounded overflow-hidden relative">
        <div className="absolute top-4 left-6 z-10 bg-bg-panel/80 backdrop-blur-sm border border-border-panel px-3 py-1.5 rounded text-[10px] font-mono text-text-primary shadow-lg">
          Spectral Contour Map: Total SE (dB) vs. Frequency & Thickness
        </div>
        <Plot
          data={[
            {
              z: heatmapResult.se_matrix_db,
              x: heatmapResult.frequencies_mhz,
              y: heatmapResult.thicknesses_mm,
              type: 'contour',
              colorscale: 'Viridis',
              contours: {
                showlabels: true,
                labelfont: { family: 'Inter', size: 10, color: '#fff' }
              },
              colorbar: {
                title: { text: 'SE (dB)', font: { color: '#8b8b99', family: 'Inter' } },
                tickfont: { color: '#8b8b99', family: 'Inter' }
              }
            }
          ]}
          layout={{
            autosize: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#8b8b99', family: 'Inter' },
            xaxis: { title: 'Frequency (MHz)', gridcolor: '#2d2d33', zerolinecolor: '#2d2d33' },
            yaxis: { title: 'Thickness (mm)', gridcolor: '#2d2d33', zerolinecolor: '#2d2d33' },
            margin: { t: 40, r: 20, b: 40, l: 60 },
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    )
  }

  // STANDARD SWEEP PLOT
  if (!sweepResult) return null;

  const xData = sweepMode === 'frequency' ? sweepResult.frequencies_mhz : sweepResult.thicknesses_mm
  const xLabel = sweepMode === 'frequency' ? 'Frequency (MHz)' : 'Thickness (mm)'

  return (
    <div className="w-full h-full bg-bg-canvas border border-border-panel rounded overflow-hidden">
      <Plot
        data={[
          {
            x: xData,
            y: sweepResult.total_se_db,
            type: 'scatter',
            mode: 'lines',
            name: 'Total SE',
            line: { color: '#005fb8', width: 2 },
          },
          {
            x: xData,
            y: sweepResult.absorption_loss_db,
            type: 'scatter',
            mode: 'lines',
            name: 'Absorption Loss',
            line: { color: '#238636', width: 1.5, dash: 'dot' },
          },
          {
            x: xData,
            y: sweepResult.reflection_loss_db,
            type: 'scatter',
            mode: 'lines',
            name: 'Reflection Loss',
            line: { color: '#d29922', width: 1.5, dash: 'dash' },
          }
        ]}
        layout={{
          autosize: true,
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#8b8b99', family: 'Inter' },
          xaxis: { title: xLabel, gridcolor: '#2d2d33', zerolinecolor: '#2d2d33' },
          yaxis: { title: 'Shielding Effectiveness (dB)', gridcolor: '#2d2d33', zerolinecolor: '#2d2d33' },
          margin: { t: 40, r: 20, b: 40, l: 60 },
          legend: { orientation: 'h', y: 1.1 },
          hovermode: 'x unified'
        }}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
