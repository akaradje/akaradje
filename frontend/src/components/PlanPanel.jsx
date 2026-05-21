export default function PlanPanel({ plan }) {
  if (!plan || !plan.steps || plan.steps.length === 0) return null

  const { steps, total_steps, completed, is_complete, current_step } = plan

  return (
    <div className="backdrop-blur-md bg-glass border border-accent/20 rounded-xl overflow-hidden shadow-lg shadow-accent/5 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-glass-border bg-accent-glow/30">
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2" />
            <rect x="9" y="3" width="6" height="4" rx="1" />
            <path d="M9 14l2 2 4-4" />
          </svg>
          <span className="text-sm font-semibold text-text-primary">Execution Plan</span>
        </div>
        <span className="text-xs text-text-tertiary">
          {completed}/{total_steps} steps
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-glass-border">
        <div
          className="h-full bg-gradient-to-r from-accent to-cyan-400 transition-all duration-500 ease-out"
          style={{ width: `${total_steps > 0 ? (completed / total_steps) * 100 : 0}%` }}
        />
      </div>

      {/* Steps list */}
      <div className="px-4 py-3 space-y-2">
        {steps.map((step) => {
          const isRunning = step.status === 'running'
          const isDone = step.status === 'done'
          const isFailed = step.status === 'failed'
          const isPending = step.status === 'pending'

          return (
            <div
              key={step.index}
              className={`flex items-start gap-3 px-3 py-2.5 rounded-lg border transition-all duration-300 ${
                isRunning
                  ? 'border-accent/40 bg-accent-glow/20'
                  : isDone
                    ? 'border-success/20 bg-success/5 opacity-80'
                    : isFailed
                      ? 'border-error/20 bg-error/5'
                      : 'border-glass-border bg-glass/50'
              }`}
            >
              {/* Status icon */}
              <div className="shrink-0 mt-0.5">
                {isDone && (
                  <span className="flex items-center justify-center w-5 h-5 rounded-full bg-success/20 border border-success/40">
                    <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-success">
                      <path d="M3 8l4 3 6-6" />
                    </svg>
                  </span>
                )}
                {isRunning && (
                  <span className="flex items-center justify-center w-5 h-5 rounded-full bg-accent/20 border border-accent/40">
                    <span className="w-2 h-2 rounded-full bg-accent animate-ping" />
                  </span>
                )}
                {isFailed && (
                  <span className="flex items-center justify-center w-5 h-5 rounded-full bg-error/20 border border-error/40">
                    <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-error">
                      <path d="M4 4l8 8M12 4l-8 8" />
                    </svg>
                  </span>
                )}
                {isPending && (
                  <span className="flex items-center justify-center w-5 h-5 rounded-full bg-glass border border-glass-border">
                    <span className="w-1.5 h-1.5 rounded-full bg-text-tertiary" />
                  </span>
                )}
              </div>

              {/* Step content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded ${
                      isRunning
                        ? 'bg-accent/20 text-accent-light'
                        : isDone
                          ? 'bg-success/10 text-success'
                          : 'bg-glass text-text-tertiary'
                    }`}
                  >
                    {step.index}
                  </span>
                  <span
                    className={`text-sm leading-snug ${
                      isDone ? 'text-text-tertiary line-through decoration-text-tertiary/30' : 'text-text-primary'
                    }`}
                  >
                    {step.description}
                  </span>
                </div>

                {/* Tool hints */}
                {step.tool_hints && step.tool_hints.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-1.5 ml-7">
                    {step.tool_hints.map((hint) => (
                      <span
                        key={hint}
                        className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                          isRunning
                            ? 'bg-accent/10 text-accent border border-accent/20'
                            : 'bg-black/20 text-text-tertiary'
                        }`}
                      >
                        {hint}
                      </span>
                    ))}
                  </div>
                )}

                {/* Expected output */}
                {step.expected_output && isRunning && (
                  <p className="text-xs text-text-tertiary mt-1 ml-7 italic">
                    → {step.expected_output}
                  </p>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Footer — overall status */}
      <div className="px-4 py-2 border-t border-glass-border bg-glass/30">
        <div className="flex items-center gap-2 text-xs">
          {is_complete ? (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-success" />
              <span className="text-success">Plan complete</span>
            </>
          ) : current_step ? (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
              <span className="text-text-secondary">
                Step {current_step} of {total_steps}
              </span>
            </>
          ) : (
            <>
              <span className="w-1.5 h-1.5 rounded-full bg-text-tertiary" />
              <span className="text-text-tertiary">Plan created</span>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
