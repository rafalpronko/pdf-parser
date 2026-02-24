import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Fetch available documents on mount
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch('/api/v1/documents');
      const data = await response.json();
      setDocuments(data || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Check if it's a PDF
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Tylko pliki PDF są obsługiwane');
      return;
    }

    setUploading(true);
    setUploadProgress(`Wgrywanie ${file.name}...`);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/v1/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      await response.json();
      setUploadProgress(`✅ ${file.name} wgrano pomyślnie!`);
      
      // Refresh documents list
      await fetchDocuments();

      // Clear progress after 3 seconds
      setTimeout(() => {
        setUploadProgress('');
      }, 3000);
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadProgress(`❌ Błąd podczas wgrywania ${file.name}`);
      setTimeout(() => {
        setUploadProgress('');
      }, 3000);
    } finally {
      setUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/api/v1/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: input,
          top_k: 16,
          temperature: 0.0,
          include_sources: true
        }),
      });

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        processingTime: data.processing_time,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'Przepraszam, wystąpił błąd podczas przetwarzania zapytania.',
        error: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="sidebar">
        <h2>📚 Dokumenty</h2>
        
        <div className="upload-section">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
            id="file-upload"
          />
          <button
            className="upload-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? '⏳ Wgrywanie...' : '📤 Wgraj PDF'}
          </button>
          {uploadProgress && (
            <div className="upload-progress">{uploadProgress}</div>
          )}
        </div>

        <div className="documents-list">
          {documents.length === 0 ? (
            <p className="no-docs">Brak dokumentów</p>
          ) : (
            documents.map((doc) => (
              <div key={doc.doc_id} className="document-item">
                <div className="doc-icon">📄</div>
                <div className="doc-info">
                  <div className="doc-name">{doc.filename}</div>
                  <div className="doc-meta">
                    {doc.num_pages} stron • {doc.num_chunks} chunków
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="main-content">
        <header className="header">
          <h1>💬 RAG Chat</h1>
          <p>Zadaj pytanie o swoje dokumenty</p>
        </header>

        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>👋 Witaj!</h2>
                <p>Zadaj pytanie o dokumenty w bazie wiedzy.</p>
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-avatar">
                  {message.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-text">{message.content}</div>
                  
                  {message.sources && message.sources.length > 0 && (
                    <div className="sources">
                      <div className="sources-header">📎 Źródła:</div>
                      {message.sources.map((source, idx) => (
                        <div key={idx} className="source-item">
                          <div className="source-header">
                            <span className="source-filename">{source.filename}</span>
                            <span className="source-page">Strona {source.page}</span>
                            <span className="source-score">
                              {(source.relevance_score * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="source-content">
                            {source.chunk_content.substring(0, 150)}...
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {message.processingTime && (
                    <div className="processing-time">
                      ⏱️ {message.processingTime.toFixed(2)}s
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Zadaj pytanie..."
              disabled={loading}
              className="message-input"
            />
            <button 
              type="submit" 
              disabled={loading || !input.trim()}
              className="send-button"
            >
              {loading ? '⏳' : '📤'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
