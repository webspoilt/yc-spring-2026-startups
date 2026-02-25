import axios, { AxiosInstance } from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

class APIService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json'
      }
    })
  }

  // Projects
  async getProjects() {
    const response = await this.client.get('/projects')
    return response.data
  }

  async getProject(id: number) {
    const response = await this.client.get(`/projects/${id}`)
    return response.data
  }

  async createProject(data: { name: string; description?: string }) {
    const response = await this.client.post('/projects', data)
    return response.data
  }

  async updateProject(id: number, data: { name?: string; description?: string }) {
    const response = await this.client.patch(`/projects/${id}`, data)
    return response.data
  }

  async deleteProject(id: number) {
    await this.client.delete(`/projects/${id}`)
  }

  // User Stories
  async getUserStories(projectId: number) {
    const response = await this.client.get(`/projects/${projectId}/user-stories`)
    return response.data
  }

  async createUserStory(projectId: number, data: {
    title: string
    description: string
    as_a?: string
    i_want?: string
    so_that?: string
    priority?: string
  }) {
    const response = await this.client.post(`/projects/${projectId}/user-stories`, data)
    return response.data
  }

  async updateUserStory(id: number, data: Partial<{
    title: string
    description: string
    as_a: string
    i_want: string
    so_that: string
    priority: string
    status: string
    acceptance_criteria: string
  }>) {
    const response = await this.client.patch(`/user-stories/${id}`, data)
    return response.data
  }

  // Specifications
  async getSpecifications(projectId: number) {
    const response = await this.client.get(`/projects/${projectId}/specifications`)
    return response.data
  }

  async getSpecification(id: number) {
    const response = await this.client.get(`/specifications/${id}`)
    return response.data
  }

  async createSpecification(projectId: number, data: {
    title: string
    content: string
    status?: string
  }) {
    const response = await this.client.post(`/projects/${projectId}/specifications`, data)
    return response.data
  }

  // AI Generation
  async generateSpec(projectId: number, prompt: string, style?: string) {
    const response = await this.client.post('/generate-spec', {
      project_id: projectId,
      prompt,
      style: style || 'technical'
    })
    return response.data
  }

  async generateUserStories(projectId: number, featureDescription: string, count?: number) {
    const response = await this.client.post('/generate-user-stories', {
      project_id: projectId,
      feature_description: featureDescription,
      count: count || 3
    })
    return response.data
  }

  async getAICapabilities() {
    const response = await this.client.get('/ai/capabilities')
    return response.data
  }
}

export const api = new APIService()
export default api
