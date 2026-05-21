import { useState, useEffect, useCallback } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import ChatInput from './components/ChatInput'
import StatusBar from './components/StatusBar'
import ArtifactPanel from './components/ArtifactPanel'
import { useChat } from './hooks/useChat'
import { useFileUpload } from './hooks/useFileUpload'
import { useProjects } from './hooks/useProjects'

export default function App() {
  const [effort, setEffort] = useState('medium')
  const [complexity, setComplexity] = useState('')
  const [showThinking, setShowThinking] = useState(true)
  const [panelOpen, setPanelOpen] = useState(false)

  const {
    messages,
    isStreaming,
    status,
    totalTokens,
    totalCost,
    artifactIds,
    plan,
    sendMessage,
    clearChat,
  } = useChat()
  const { files, uploadFile, removeFile, clearFiles, fileIds } = useFileUpload()
  const {
    projects,
    activeProjectId,
    activeProject,
    loading: projectLoading,
    switchProject,
    createProject,
    updateProject,
    deleteProject,
    addFile: addProjectFile,
    removeFile: removeProjectFile,
  } = useProjects()

  // Auto-open the artifact panel when new artifacts arrive
  useEffect(() => {
    if (artifactIds.length > 0) {
      setPanelOpen(true)
    }
  }, [artifactIds])

  const handleSend = (text) => {
    sendMessage(text, {
      effort,
      complexity,
      fileIds,
      showThinking,
      projectId: activeProjectId,
    })
    clearFiles()
  }

  const handleClosePanel = useCallback(() => {
    setPanelOpen(false)
  }, [])

  return (
    <div className="flex flex-col h-screen bg-bg-primary text-text-primary overflow-hidden">
      <Header
        effort={effort}
        setEffort={setEffort}
        complexity={complexity}
        setComplexity={setComplexity}
        showThinking={showThinking}
        setShowThinking={setShowThinking}
        onClearChat={clearChat}
        activeProjectName={activeProject?.name}
        projectCount={projects.length}
      />
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar — project selector + sessions */}
        <Sidebar
          projects={projects}
          activeProjectId={activeProjectId}
          activeProject={activeProject}
          loading={projectLoading}
          onSwitchProject={switchProject}
          onCreateProject={createProject}
          onUpdateProject={updateProject}
          onDeleteProject={deleteProject}
          onAddFile={addProjectFile}
          onRemoveFile={removeProjectFile}
        />

        {/* Main area — chat + optional artifact panel */}
        <div className="flex flex-1 overflow-hidden">
          <div className="flex flex-col min-w-0 flex-1">
            <ChatArea
              messages={messages}
              isStreaming={isStreaming}
              showThinking={showThinking}
              plan={plan}
            />
            <ChatInput
              onSend={handleSend}
              isStreaming={isStreaming}
              files={files}
              onUpload={uploadFile}
              onRemoveFile={removeFile}
            />
          </div>

          {/* Artifact Panel — slides in from right */}
          {panelOpen && artifactIds.length > 0 && (
            <div className="w-[45%] min-w-[380px] max-w-[800px] shrink-0">
              <ArtifactPanel
                artifactIds={artifactIds}
                onClose={handleClosePanel}
              />
            </div>
          )}
        </div>
      </div>
      <StatusBar
        status={status}
        message={status}
        totalTokens={totalTokens}
        totalCost={totalCost}
      />
    </div>
  )
}
