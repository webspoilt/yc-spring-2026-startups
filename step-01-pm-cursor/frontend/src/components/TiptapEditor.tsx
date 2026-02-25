import React, { useCallback, useState } from 'react'
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Placeholder from '@tiptap/extension-placeholder'
import TaskList from '@tiptap/extension-task-list'
import TaskItem from '@tiptap/extension-task-list'

interface TiptapEditorProps {
  initialContent?: string
  onChange?: (content: string) => void
  editable?: boolean
  placeholder?: string
}

const TiptapEditor: React.FC<TiptapEditorProps> = ({
  initialContent = '',
  onChange,
  editable = true,
  placeholder = 'Start writing your specification...'
}) => {
  const [isSaving, setIsSaving] = useState(false)

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3]
        }
      }),
      Placeholder.configure({
        placeholder
      }),
      TaskList,
      TaskItem.configure({
        nested: true
      })
    ],
    content: initialContent ? JSON.parse(initialContent) : '',
    editable,
    onUpdate: ({ editor }) => {
      if (onChange) {
        const json = editor.getJSON()
        onChange(JSON.stringify(json))
      }
    }
  })

  const handleSave = useCallback(() => {
    if (!editor) return
    setIsSaving(true)
    
    const json = editor.getJSON()
    const html = editor.getHTML()
    
    // Simulate save - in real app, call API
    setTimeout(() => {
      setIsSaving(false)
      console.log('Saved:', { json, html })
    }, 500)
  }, [editor])

  if (!editor) {
    return <div className="animate-pulse">Loading editor...</div>
  }

  return (
    <div className="border border-gray-300 rounded-lg overflow-hidden">
      {/* Toolbar */}
      {editable && (
        <div className="bg-gray-50 border-b border-gray-300 p-2 flex flex-wrap gap-1">
          <button
            onClick={() => editor.chain().focus().toggleBold().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('bold') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            <strong>B</strong>
          </button>
          <button
            onClick={() => editor.chain().focus().toggleItalic().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('italic') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            <em>I</em>
          </button>
          <button
            onClick={() => editor.chain().focus().toggleStrike().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('strike') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            <s>S</s>
          </button>
          
          <div className="w-px bg-gray-300 mx-1" />
          
          <button
            onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('heading', { level: 1 }) ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            H1
          </button>
          <button
            onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('heading', { level: 2 }) ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            H2
          </button>
          <button
            onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('heading', { level: 3 }) ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            H3
          </button>
          
          <div className="w-px bg-gray-300 mx-1" />
          
          <button
            onClick={() => editor.chain().focus().toggleBulletList().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('bulletList') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            • List
          </button>
          <button
            onClick={() => editor.chain().focus().toggleOrderedList().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('orderedList') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            1. List
          </button>
          <button
            onClick={() => editor.chain().focus().toggleTaskList().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('taskList') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            ☑ Tasks
          </button>
          
          <div className="w-px bg-gray-300 mx-1" />
          
          <button
            onClick={() => editor.chain().focus().toggleBlockquote().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('blockquote') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            " Quote
          </button>
          <button
            onClick={() => editor.chain().focus().toggleCodeBlock().run()}
            className={`px-3 py-1 rounded ${
              editor.isActive('codeBlock') ? 'bg-gray-200' : 'hover:bg-gray-200'
            }`}
          >
            {'</>'} Code
          </button>
          
          <div className="flex-grow" />
          
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-4 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {isSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
      )}
      
      {/* Editor Content */}
      <div className="p-4 min-h-[400px] prose max-w-none">
        <EditorContent editor={editor} />
      </div>
    </div>
  )
}

export default TiptapEditor
