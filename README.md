# Aegis - AI-Powered Cybersecurity Framework Assistant

## Project Overview

Aegis is an intelligent cybersecurity framework assistant that helps organizations implement ISO 27001 and ISO 27002 standards. The system leverages advanced AI to guide users through the complex process of designing and implementing a robust cybersecurity framework tailored to their specific business needs.

## Key Technical Features

### AI and Machine Learning Components

- **Large Language Model Integration**: Utilizes Groq's LLM API (specifically llama-3.3-70b-Versatile) for generating contextually relevant responses based on ISO standards.
- **Vector Database for Semantic Search**: Implements FAISS (Facebook AI Similarity Search) for efficient similarity search across ISO documentation.
- **Embedding Generation Pipeline**: Uses Hugging Face's sentence transformers (all-MiniLM-L6-v2) to create document embeddings for semantic retrieval.
- **Retrieval-Augmented Generation (RAG)**: Combines vector search with LLM capabilities to provide accurate, standards-based guidance.

### Software Architecture

- **FastAPI Backend**: RESTful API service with proper error handling and middleware configuration.
- **Streamlit Frontend**: Interactive web interface for user engagement.
- **Redis Caching Layer**: Optimizes performance by caching conversation data.
- **SQLAlchemy ORM**: Provides database abstraction and management.
- **Containerization-Ready**: Structured for easy Docker deployment.

### Engineering Best Practices

- **Modular Service Architecture**: Clean separation of concerns with dedicated services for database, embeddings, LLM, and Redis operations.
- **Comprehensive Error Handling**: Backoff strategies for API calls and custom exception classes.
- **Token Management**: Efficient token counting and management to optimize API usage.
- **Environment Configuration**: Flexible configuration system with dotenv integration.
- **Type Annotations**: Extensive use of Python type hints for better code quality.

## Technical Implementation Details

### Conversation Flow

1. The system initiates a guided conversation to understand the user's business context.
2. It asks targeted questions about the organization's activities, size, and existing security measures.
3. After gathering sufficient information, it provides tailored ISO 27001/27002 recommendations.
4. Users can continue the conversation to explore specific aspects of the recommendations.

### Data Processing Pipeline

1. **Document Ingestion**: PDF documents containing ISO standards are loaded and processed.
2. **Text Chunking**: Documents are split into manageable chunks using RecursiveCharacterTextSplitter.
3. **Embedding Generation**: Text chunks are converted to vector embeddings.
4. **Vector Storage**: Embeddings are stored in a FAISS index for efficient retrieval.
5. **Contextual Retrieval**: User queries trigger semantic search to find relevant ISO guidance.

### System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│  FastAPI    │────▶│  Database   │
│  Frontend   │◀────│  Backend    │◀────│  Service    │
└─────────────┘     └─────────────┘     └─────────────┘
                          │  ▲
                          │  │
                          ▼  │
                    ┌─────────────┐     ┌─────────────┐
                    │    LLM      │────▶│   Vector    │
                    │   Service   │◀────│    Store    │
                    └─────────────┘     └─────────────┘
                          │  ▲
                          │  │
                          ▼  │
                    ┌─────────────┐
                    │    Redis    │
                    │    Cache    │
                    └─────────────┘
```

## Technical Skills Demonstrated

- **AI/ML Engineering**: Implementation of embedding models, vector databases, and LLM integration.
- **Full-Stack Development**: Backend API development with FastAPI and frontend with Streamlit.
- **Database Design**: SQL database schema design and Redis caching implementation.
- **Software Architecture**: Clean, modular design with separation of concerns.
- **DevOps Readiness**: Environment configuration, dependency management, and deployment preparation.

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Redis instance
- Groq API key

### Environment Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with the following variables:
   ```
   DATABASE_URL=postgresql://user:password@localhost/aegis
   REDIS_URL=redis://localhost:6379/0
   GROQ_API_KEY=your_groq_api_key
   API_URL=http://localhost:8000
   MODEL_NAME=llama-3.3-70b-Versatile
   ```

### Running the Application

1. Initialize the database: `python init_db.py`
2. Start the API server: `uvicorn src.api.main:app --reload`
3. Start the Streamlit frontend: `streamlit run src.streamlit_app:main`

## Future Development Roadmap

- Integration with additional security standards beyond ISO 27001/27002
- Enhanced visualization of security framework components
- Multi-user support with role-based access control
- Automated compliance reporting generation
- Integration with security scanning tools for real-time assessment

---

This project demonstrates advanced AI application development, combining natural language processing, vector databases, and modern web technologies to solve complex business problems in the cybersecurity domain.
