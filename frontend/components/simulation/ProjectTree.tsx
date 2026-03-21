'use client'

import { useWorkbenchStore, TreeNodeId } from '@/lib/store'
import { FolderGit2, Box, Layers, Play, LineChart, Activity, Cpu, Cable, Zap } from 'lucide-react'
import { cn } from '@/lib/utils'

export function ProjectTree() {
  const { activeNode, setActiveNode } = useWorkbenchStore()

  const nodes: { id: TreeNodeId, label: string, icon: React.ElementType, children?: { id: TreeNodeId, label: string, icon: React.ElementType }[] }[] = [
    {
      id: 'enclosure', label: '1. Enclosure Shielding', icon: Cpu,
      children: [
        { id: 'geometry', label: '1.1 Shield Geometry', icon: Box },
        { id: 'materials', label: '1.2 Materials & Composites', icon: Layers }
      ]
    },
    {
      id: 'cables', label: '2. Cable Harness & SI', icon: Cable
    },
    {
      id: 'hazards', label: '3. Environment Hazards', icon: Zap
    },
    {
      id: 'analysis', label: '4. Analysis Setup', icon: Play,
      children: [
        { id: 'sweep', label: '4.1 Parametric Sweep', icon: Activity }
      ]
    },
    {
      id: 'results', label: '5. Results & Post-Processing', icon: LineChart
    }
  ]

  type NodeDef = { id: TreeNodeId, label: string, icon: React.ElementType, children?: NodeDef[] }
  const renderNode = (node: NodeDef, depth = 0) => {
    const isActive = activeNode === node.id
    return (
      <div key={node.id} className="flex flex-col">
        <button
          onClick={() => setActiveNode(node.id)}
          className={cn(
            "flex items-center gap-2 px-2 py-1.5 w-full text-left transition-colors",
            isActive ? "bg-accent-selection text-white" : "text-text-secondary hover:bg-bg-subpanel hover:text-text-primary"
          )}
          style={{ paddingLeft: `${(depth * 12) + 8}px` }}
        >
          <node.icon className={cn("w-3.5 h-3.5", isActive ? "text-white" : "text-text-muted")} />
          <span className="text-xs font-medium tracking-wide">{node.label}</span>
        </button>
        {node.children && node.children.map((child: NodeDef) => renderNode(child, depth + 1))}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 px-2 py-1.5 text-text-primary font-medium border-b border-border-panel mb-1 pb-2">
        <FolderGit2 className="w-4 h-4 text-accent-selection" />
        <span className="text-xs">Project_1</span>
      </div>
      {nodes.map(node => renderNode(node))}
    </div>
  )
}
