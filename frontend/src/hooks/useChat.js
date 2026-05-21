import { useState, useCallback, useRef } from 'react'
import { streamChat } from '../utils/api'

const WELCOME_MESSAGE = {
  role: 'assistant',
  content:
    "Welcome to **akaradje** 👋\n\nI'm your AI assistant powered by DeepSeek. I can help you with:\n\n- 🔍 **Web search** — find information online\n- 🐍 **Python execution** — run code snippets\n- 📄 **File analysis** — read and analyze uploaded files\n- 🌐 **URL fetching** — extract content from web pages\n- 💻 **Shell commands** — execute system commands\n- 🧮 **Calculations** — solve math problems\n\nType a message below to get started!",
  thinking: '',
  toolCalls: [],
  meta: null,
}

export function useChat() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE])
  const [isStreaming, setIsStreaming] = useState(false)
  const [status, setStatus] = useState('Ready')
  const [totalTokens, setTotalTokens] = useState(0)
  const [totalCost, setTotalCost] = useState(0)
  const [artifactIds, setArtifactIds] = useState([])
  const [plan, setPlan] = useState(null)
  const abortRef = useRef(null)

  const sendMessage = useCallback(
    async (text, { effort, complexity, fileIds, showThinking, projectId } = {}) => {
      if (!text.trim() || isStreaming) return

      const userMsg = { role: 'user', content: text }
      const assistantMsg = {
        role: 'assistant',
        content: '',
        thinking: '',
        toolCalls: [],
        meta: null,
      }

      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setIsStreaming(true)
      setStatus('Thinking...')

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const response = await streamChat(text, {
          effort,
          complexity,
          fileIds,
          showThinking,
          projectId,
        })

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: 'Request failed' }))
          throw new Error(err.detail || `HTTP ${response.status}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const dataStr = line.slice(6).trim()
            if (!dataStr || dataStr === '[DONE]') continue

            let event
            try {
              event = JSON.parse(dataStr)
            } catch {
              continue
            }

            setMessages((prev) => {
              const updated = [...prev]
              const last = { ...updated[updated.length - 1] }
              updated[updated.length - 1] = last

              switch (event.type) {
                case 'progress':
                  setStatus(event.message || 'Processing...')
                  break

                case 'reasoning':
                  last.thinking = (last.thinking || '') + (event.content || '')
                  break

                case 'content':
                  last.content = (last.content || '') + (event.content || '')
                  setStatus('Streaming...')
                  break

                case 'tool_call':
                  last.toolCalls = [
                    ...(last.toolCalls || []),
                    {
                      id: event.id || crypto.randomUUID(),
                      name: event.name,
                      args: event.arguments || event.args || {},
                      result: null,
                      status: 'calling',
                      durationMs: null,
                    },
                  ]
                  setStatus(`Using tool: ${event.name}...`)
                  break

                case 'tool_result': {
                  last.toolCalls = (last.toolCalls || []).map((tc) =>
                    tc.id === event.id || tc.name === event.name
                      ? {
                          ...tc,
                          result: event.result || event.content,
                          status: 'done',
                          durationMs: event.duration_ms || event.durationMs || null,
                        }
                      : tc
                  )
                  setStatus('Processing...')
                  break
                }

                case 'done':
                  last.meta = {
                    complexity: event.complexity || null,
                    totalTokens: event.total_tokens || event.tokens || 0,
                    cost: event.cost || 0,
                    model: event.model || null,
                  }
                  break

                case 'plan_update':
                  setPlan(event.plan || null)
                  break

                case 'artifact':
                  setArtifactIds((prev) => {
                    if (prev.includes(event.id)) return prev
                    return [...prev, event.id]
                  })
                  break

                case 'error':
                  last.content =
                    (last.content || '') +
                    `\n\n⚠️ **Error**: ${event.message || event.error || 'Unknown error'}`
                  break

                default:
                  break
              }

              return updated
            })
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setMessages((prev) => {
            const updated = [...prev]
            const last = { ...updated[updated.length - 1] }
            last.content = `⚠️ **Error**: ${err.message}`
            updated[updated.length - 1] = last
            return updated
          })
        }
      } finally {
        setIsStreaming(false)
        setStatus('Ready')
        abortRef.current = null

        // Update totals from the last message meta
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          if (last?.meta) {
            setTotalTokens((t) => t + (last.meta.totalTokens || 0))
            setTotalCost((c) => c + (last.meta.cost || 0))
          }
          return prev
        })
      }
    },
    [isStreaming]
  )

  const stopStreaming = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
  }, [])

  const clearChat = useCallback(() => {
    setMessages([WELCOME_MESSAGE])
    setTotalTokens(0)
    setTotalCost(0)
    setArtifactIds([])
    setPlan(null)
  }, [])

  return {
    messages,
    isStreaming,
    status,
    totalTokens,
    totalCost,
    artifactIds,
    plan,
    sendMessage,
    stopStreaming,
    clearChat,
  }
}
