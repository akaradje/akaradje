import { useEffect, useRef } from 'react'
import ChatMessage from './ChatMessage'
import PlanPanel from './PlanPanel'

export default function ChatArea({ messages, isStreaming, showThinking, plan }) {
  const bottomRef = useRef(null)
  const containerRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto px-4 py-6"
    >
      <div className="max-w-3xl mx-auto space-y-4">
        {/* Plan visualization — shown above messages when a plan exists */}
        {plan && plan.steps && plan.steps.length > 0 && (
          <PlanPanel plan={plan} />
        )}

        {messages.map((msg, i) => (
          <ChatMessage
            key={i}
            role={msg.role}
            content={msg.content}
            thinking={msg.thinking}
            toolCalls={msg.toolCalls}
            meta={msg.meta}
            isStreaming={isStreaming && i === messages.length - 1}
            showThinking={showThinking}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
