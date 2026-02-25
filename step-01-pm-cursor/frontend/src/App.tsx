import React, { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Link, useParams } from 'react-router-dom'
import TiptapEditor from './components/TiptapEditor'
import { api } from './services/api'

interface Project {
  id: number
  name: string
  description: string
  created_at: string
}

interface UserStory {
  id: number
  title: string
  description: string
  as_a?: string
  i_want?: string
  so_that?: string
  priority: string
  status: string
}

interface Specification {
  id: number
  title: string
  content: string
  status: string
  generated_by_ai?: string
}

// Project List Component
const ProjectList: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([])
  const [showCreate, setShowCreate] = useState(false)
  const [newProject, setNewProject] = useState({ name: '', description: '' })

  useEffect(() => {
    loadProjects()
  }, [])

  const loadProjects = async () => {
    const data = await api.getProjects()
    setProjects(data)
  }

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    await api.createProject(newProject)
    setNewProject({ name: '', description: '' })
    setShowCreate(false)
    loadProjects()
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Projects</h1>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          + New Project
        </button>
      </div>

      {showCreate && (
        <form onSubmit={handleCreate} className="mb-6 p-4 bg-gray-50 rounded-lg">
          <input
            type="text"
            placeholder="Project name"
            value={newProject.name}
            onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
            className="w-full p-2 border rounded mb-2"
            required
          />
          <textarea
            placeholder="Description"
            value={newProject.description}
            onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
            className="w-full p-2 border rounded mb-2"
            rows={3}
          />
          <button type="submit" className="px-4 py-2 bg-green-600 text-white rounded">
            Create
          </button>
        </form>
      )}

      <div className="grid gap-4">
        {projects.map((project) => (
          <Link
            key={project.id}
            to={`/projects/${project.id}`}
            className="block p-4 border rounded-lg hover:bg-gray-50 transition"
          >
            <h2 className="text-xl font-semibold">{project.name}</h2>
            <p className="text-gray-600 mt-1">{project.description || 'No description'}</p>
            <p className="text-sm text-gray-400 mt-2">
              Created: {new Date(project.created_at).toLocaleDateString()}
            </p>
          </Link>
        ))}
      </div>
    </div>
  )
}

// Project Detail Component
const ProjectDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const [project, setProject] = useState<Project | null>(null)
  const [stories, setStories] = useState<UserStory[]>([])
  const [specs, setSpecs] = useState<Specification[]>([])
  const [activeTab, setActiveTab] = useState<'stories' | 'specs' | 'ai'>('stories')
  const [aiPrompt, setAiPrompt] = useState('')
  const [aiStyle, setAiStyle] = useState('technical')
  const [generating, setGenerating] = useState(false)

  useEffect(() => {
    if (id) {
      loadProject()
      loadStories()
      loadSpecs()
    }
  }, [id])

  const loadProject = async () => {
    const data = await api.getProject(Number(id))
    setProject(data)
  }

  const loadStories = async () => {
    const data = await api.getUserStories(Number(id))
    setStories(data)
  }

  const loadSpecs = async () => {
    const data = await api.getSpecifications(Number(id))
    setSpecs(data)
  }

  const handleGenerateSpec = async () => {
    if (!aiPrompt.trim()) return
    setGenerating(true)
    try {
      await api.generateSpec(Number(id), aiPrompt, aiStyle)
      loadSpecs()
      setAiPrompt('')
    } catch (error) {
      console.error('Failed to generate:', error)
    }
    setGenerating(false)
  }

  const handleGenerateStories = async () => {
    if (!aiPrompt.trim()) return
    setGenerating(true)
    try {
      await api.generateUserStories(Number(id), aiPrompt)
      loadStories()
      setAiPrompt('')
    } catch (error) {
      console.error('Failed to generate:', error)
    }
    setGenerating(false)
  }

  const priorityColors: Record<string, string> = {
    high: 'bg-red-100 text-red-800',
    medium: 'bg-yellow-100 text-yellow-800',
    low: 'bg-green-100 text-green-800',
    critical: 'bg-red-200 text-red-900'
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <Link to="/" className="text-blue-600 hover:underline mb-4 inline-block">
        ← Back to Projects
      </Link>
      
      <h1 className="text-3xl font-bold mb-2">{project?.name}</h1>
      <p className="text-gray-600 mb-6">{project?.description}</p>

      {/* Tabs */}
      <div className="flex gap-4 border-b mb-6">
        <button
          onClick={() => setActiveTab('stories')}
          className={`pb-2 px-4 ${activeTab === 'stories' ? 'border-b-2 border-blue-600 font-semibold' : ''}`}
        >
          User Stories ({stories.length})
        </button>
        <button
          onClick={() => setActiveTab('specs')}
          className={`pb-2 px-4 ${activeTab === 'specs' ? 'border-b-2 border-blue-600 font-semibold' : ''}`}
        >
          Specifications ({specs.length})
        </button>
        <button
          onClick={() => setActiveTab('ai')}
          className={`pb-2 px-4 ${activeTab === 'ai' ? 'border-b-2 border-blue-600 font-semibold' : ''}`}
        >
          AI Generator
        </button>
      </div>

      {/* User Stories Tab */}
      {activeTab === 'stories' && (
        <div className="grid gap-4">
          {stories.length === 0 ? (
            <p className="text-gray-500">No user stories yet. Use AI to generate some!</p>
          ) : (
            stories.map((story) => (
              <div key={story.id} className="border rounded-lg p-4">
                <div className="flex justify-between items-start">
                  <h3 className="text-lg font-semibold">{story.title}</h3>
                  <span className={`px-2 py-1 rounded text-sm ${priorityColors[story.priority]}`}>
                    {story.priority}
                  </span>
                </div>
                <p className="text-gray-600 mt-2">{story.description}</p>
                {story.as_a && (
                  <p className="text-sm text-gray-500 mt-2">
                    <strong>As a:</strong> {story.as_a}
                  </p>
                )}
                {story.i_want && (
                  <p className="text-sm text-gray-500">
                    <strong>I want:</strong> {story.i_want}
                  </p>
                )}
                {story.so_that && (
                  <p className="text-sm text-gray-500">
                    <strong>So that:</strong> {story.so_that}
                  </p>
                )}
                {story.acceptance_criteria && (
                  <div className="mt-3 p-3 bg-gray-50 rounded">
                    <strong className="text-sm">Acceptance Criteria:</strong>
                    <pre className="text-sm mt-1 whitespace-pre-wrap">{story.acceptance_criteria}</pre>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Specifications Tab */}
      {activeTab === 'specs' && (
        <div className="grid gap-6">
          {specs.length === 0 ? (
            <p className="text-gray-500">No specifications yet. Use AI to generate one!</p>
          ) : (
            specs.map((spec) => (
              <div key={spec.id} className="border rounded-lg overflow-hidden">
                <div className="bg-gray-50 p-4 border-b flex justify-between items-center">
                  <h3 className="text-lg font-semibold">{spec.title}</h3>
                  <span className="text-sm text-gray-500">
                    {spec.generated_by_ai && `🤖 ${spec.generated_by_ai}`}
                  </span>
                </div>
                <TiptapEditor initialContent={spec.content} editable={false} />
              </div>
            ))
          )}
        </div>
      )}

      {/* AI Generator Tab */}
      {activeTab === 'ai' && (
        <div className="max-w-2xl">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 className="font-semibold mb-2">🤖 AI-Powered Generation</h3>
            <p className="text-sm text-gray-600">
              Describe your feature or requirement, and AI will generate detailed specifications and user stories.
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Style</label>
              <select
                value={aiStyle}
                onChange={(e) => setAiStyle(e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="technical">Technical</option>
                <option value="user-facing">User-Facing</option>
                <option value="executive">Executive</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                What would you like to build?
              </label>
              <textarea
                value={aiPrompt}
                onChange={(e) => setAiPrompt(e.target.value)}
                placeholder="E.g., A user authentication system with email/password login, OAuth support, and password reset functionality..."
                className="w-full p-3 border rounded-lg"
                rows={5}
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleGenerateSpec}
                disabled={generating || !aiPrompt.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {generating ? 'Generating...' : 'Generate Specification'}
              </button>
              <button
                onClick={handleGenerateStories}
                disabled={generating || !aiPrompt.trim()}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
              >
                {generating ? 'Generating...' : 'Generate User Stories'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Main App
const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-white">
        <header className="bg-gray-800 text-white p-4">
          <div className="max-w-6xl mx-auto">
            <Link to="/" className="text-xl font-bold">PM Cursor</Link>
          </div>
        </header>
        
        <Routes>
          <Route path="/" element={<ProjectList />} />
          <Route path="/projects/:id" element={<ProjectDetail />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

export default App
