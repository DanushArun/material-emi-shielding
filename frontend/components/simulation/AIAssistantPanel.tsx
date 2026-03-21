'use client'

import { useState } from 'react'
import { sendChatMessage } from '@/lib/api'
import { Terminal } from 'lucide-react'

// Simple mock store for chat since we removed it from WorkbenchStore to keep it clean, 
// but we can add it back or just use local state for this console.
// For true persistence, we should add chatHistory to useWorkbenchStore.

export function AIAssistantPanel() {
  const [messages, setMessages] = useState<{role: 'user'|'assistant', content: string}[]>([
    { role: 'assistant', content: 'TMM Engine AI Assistant initialized. Ready for parameter queries.' }
  ])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)

  const handleSend = async () => {
    if (!input.trim()) return
    const newMessages = [...messages, { role: 'user' as const, content: input }]
    setMessages(newMessages)
    setInput('')
    setIsTyping(true)

    try {
      const { response } = await sendChatMessage(input, messages)
      setMessages([...newMessages, { role: 'assistant', content: response }])
    } catch {
      setMessages([...newMessages, { role: 'assistant', content: 'Connection to AI solver failed.' }])
    } finally {
      setIsTyping(false)
    }
  }

  return (
    <div className="flex flex-col h-full bg-bg-canvas font-mono text-[11px]">
      <div className="flex items-center gap-2 px-3 py-1 bg-bg-panel border-b border-border-panel text-text-secondary">
        <Terminal className="w-3 h-3" />
        <span>Console / Assistant</span>
      </div>
      <div className="flex-1 overflow-auto p-3 flex flex-col gap-2">
        {messages.map((m, i) => (
          <div key={i} className="flex gap-2">
            <span className={m.role === 'user' ? 'text-accent-warning' : 'text-accent-selection'}>
              {m.role === 'user' ? '>' : 'SYS'}
            </span>
            <span className={m.role === 'user' ? 'text-text-primary' : 'text-text-secondary'}>
              {m.content}
            </span>
          </div>
        ))}
        {isTyping && (
          <div className="flex gap-2">
            <span className="text-accent-selection">SYS</span>
            <span className="text-text-secondary animate-pulse">Processing...</span>
          </div>
        )}
      </div>
      <div className="flex items-center border-t border-border-panel bg-bg-subpanel">
        <span className="pl-3 text-accent-warning">{'>'}</span>
        <input 
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          className="flex-1 bg-transparent border-none outline-none px-2 py-1.5 text-text-primary"
          placeholder="Enter command or query..."
          disabled={isTyping}
        />
      </div>
    </div>
  )
}
