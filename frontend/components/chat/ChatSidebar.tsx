'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Bot, ChevronRight, Loader2, Send, User } from 'lucide-react'

import { sendChatMessage } from '@/lib/api'
import { useChatStore, useCompositionStore, useResultsStore, useSimulationStore } from '@/lib/store'
import type { ChatMessage } from '@/types'

// ---------------------------------------------------------------------------
// Welcome message shown before the first user message
// ---------------------------------------------------------------------------

const WELCOME_MESSAGE =
  "I'm your EMI shielding design assistant. Describe your shielding requirements and I'll recommend materials and configurations."

// Starter suggestions shown on the empty state
const STARTER_SUGGESTIONS = [
  'What material gives the best SE at 1 GHz?',
  'How does thickness affect absorption loss?',
  'Explain the Schelkunoff shielding theory',
]

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatSidebarProps {
  /** Called when the user clicks the close / collapse button */
  onClose?: () => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ChatSidebar({ onClose }: ChatSidebarProps) {
  const { messages, isLoading, addMessage, setLoading } = useChatStore()
  const [input, setInput] = useState('')
  const [lastSuggestions, setLastSuggestions] = useState<string[]>([])

  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Pull current simulation state for context-aware AI responses
  const { composition } = useCompositionStore()
  const { frequency_mhz, thickness_mm, grain_size_um, analysisMode } = useSimulationStore()
  const { calculationResult, sweepResult } = useResultsStore()

  // Auto-scroll whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // ---------------------------------------------------------------------------
  // Send logic
  // ---------------------------------------------------------------------------

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || isLoading) return

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
        timestamp: new Date(),
      }
      addMessage(userMsg)
      setInput('')
      setLastSuggestions([])
      setLoading(true)

      try {
        const history = messages.slice(-10).map((m) => ({
          role: m.role,
          content: m.content,
        }))

        const context: Record<string, unknown> = {
          composition,
          frequency_mhz,
          thickness_mm,
          grain_size_um,
          analysisMode,
          latestResult: calculationResult ?? sweepResult ?? null,
        }

        const data = await sendChatMessage(trimmed, history, context)

        const assistantMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
        }
        addMessage(assistantMsg)
        setLastSuggestions(data.suggestions ?? [])
      } catch {
        const errMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content:
            'Sorry, the AI assistant is unavailable right now. Please check that the backend service is running and that a valid GEMINI_API_KEY is configured.',
          timestamp: new Date(),
        }
        addMessage(errMsg)
        setLastSuggestions([])
      } finally {
        setLoading(false)
      }
    },
    [
      isLoading,
      addMessage,
      setLoading,
      messages,
      composition,
      frequency_mhz,
      thickness_mm,
      grain_size_um,
      analysisMode,
      calculationResult,
      sweepResult,
    ],
  )

  const handleInputSend = useCallback(() => {
    sendMessage(input)
  }, [input, sendMessage])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleInputSend()
    }
  }

  const handleSuggestionClick = useCallback(
    (suggestion: string) => {
      sendMessage(suggestion)
    },
    [sendMessage],
  )

  const handleStarterClick = (suggestion: string) => {
    setInput(suggestion)
    inputRef.current?.focus()
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex flex-col h-full bg-[#12121a]">

      {/* ---- Header ---- */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a3e] shrink-0">
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center
                       bg-gradient-to-br from-cyan-500/20 to-purple-600/20
                       border border-cyan-500/30"
          >
            <Bot size={14} className="text-[#00d4ff]" />
          </div>
          <div>
            <p className="text-sm font-semibold text-[#e8e8f0] leading-none">AI Assistant</p>
            <p className="text-xs text-[#686880] mt-0.5">EMI shielding expert</p>
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-[#9898b0] hover:text-[#e8e8f0] hover:bg-[#1a1a2e] transition-colors"
            aria-label="Close AI chat"
          >
            <ChevronRight size={16} />
          </button>
        )}
      </div>

      {/* ---- Message list ---- */}
      <div className="flex-1 overflow-y-auto min-h-0 p-3 space-y-3">

        {/* Welcome / empty state */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 py-8 text-center">
            <div
              className="w-14 h-14 rounded-full flex items-center justify-center
                         bg-gradient-to-br from-cyan-500/10 to-purple-600/10
                         border border-[#2a2a3e]"
            >
              <Bot size={22} className="text-[#9898b0]" />
            </div>

            <p className="text-sm text-[#9898b0] leading-relaxed max-w-[220px]">
              {WELCOME_MESSAGE}
            </p>

            {/* Starter suggestion pills */}
            <div className="flex flex-col gap-1.5 w-full">
              {STARTER_SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleStarterClick(s)}
                  className="text-left text-xs px-3 py-2 rounded-lg
                             bg-[#0a0a0f] border border-[#2a2a3e]
                             text-[#9898b0] hover:text-[#e8e8f0] hover:border-[#3a3a52]
                             transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Conversation turns */}
        {messages.map((msg, index) => {
          const isLast = index === messages.length - 1
          return (
            <div key={msg.id}>
              <div
                className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                {/* Avatar */}
                <div
                  className={`w-7 h-7 rounded-lg shrink-0 flex items-center justify-center ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-cyan-500 to-blue-600'
                      : 'bg-gradient-to-br from-cyan-500/20 to-purple-600/20 border border-cyan-500/30'
                  }`}
                >
                  {msg.role === 'user' ? (
                    <User size={13} className="text-white" />
                  ) : (
                    <Bot size={13} className="text-[#00d4ff]" />
                  )}
                </div>

                {/* Bubble */}
                <div
                  className={`max-w-[85%] px-3 py-2 rounded-xl text-xs leading-relaxed whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-cyan-500/15 to-blue-600/15 border border-cyan-500/30 text-[#e8e8f0] rounded-tr-sm'
                      : 'bg-[#0a0a0f] border border-[#2a2a3e] text-[#e8e8f0] rounded-tl-sm'
                  }`}
                >
                  {msg.content}
                </div>
              </div>

              {/* Follow-up suggestion buttons below the last assistant message */}
              {msg.role === 'assistant' && isLast && !isLoading && lastSuggestions.length > 0 && (
                <div className="mt-2 ml-9 flex flex-col gap-1">
                  {lastSuggestions.map((s) => (
                    <button
                      key={s}
                      onClick={() => handleSuggestionClick(s)}
                      className="text-left text-xs px-3 py-1.5 rounded-lg
                                 bg-[#0a0a0f] border border-[#2a2a3e]
                                 text-[#00d4ff] hover:text-white hover:bg-cyan-500/10 hover:border-cyan-500/40
                                 transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )
        })}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-2">
            <div
              className="w-7 h-7 rounded-lg shrink-0 flex items-center justify-center
                         bg-gradient-to-br from-cyan-500/20 to-purple-600/20
                         border border-cyan-500/30"
            >
              <Bot size={13} className="text-[#00d4ff]" />
            </div>
            <div className="px-3 py-2 rounded-xl bg-[#0a0a0f] border border-[#2a2a3e] rounded-tl-sm">
              <div className="flex gap-1 items-center h-4">
                {[0, 150, 300].map((delay) => (
                  <span
                    key={delay}
                    className="w-1.5 h-1.5 rounded-full bg-[#9898b0] animate-bounce"
                    style={{ animationDelay: `${delay}ms` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={bottomRef} />
      </div>

      {/* ---- Input area ---- */}
      <div className="shrink-0 px-3 py-3 border-t border-[#2a2a3e]">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about EMI shielding..."
            disabled={isLoading}
            className="input-field flex-1 text-xs"
            aria-label="Chat input"
          />
          <button
            onClick={handleInputSend}
            disabled={!input.trim() || isLoading}
            aria-label="Send message"
            className="shrink-0 w-9 h-9 rounded-lg flex items-center justify-center
                       bg-gradient-to-r from-cyan-500 to-blue-600
                       hover:from-cyan-400 hover:to-blue-500
                       disabled:opacity-40 disabled:cursor-not-allowed
                       active:scale-95 transition-all"
          >
            {isLoading ? (
              <Loader2 size={14} className="text-white animate-spin" />
            ) : (
              <Send size={14} className="text-white" />
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
