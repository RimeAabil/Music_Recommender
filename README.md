# AI-Powered Song Recommender

An intelligent music discovery application that uses semantic vector search and natural language processing to recommend songs based on your mood, emotions, and lyrical themes. Simply describe what you're feeling or the vibe you want, and let AI find the perfect songs for you.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ Overview

Traditional music recommendation systems rely on listening history, genre tags, or collaborative filtering. This project takes a different approach: **semantic understanding of lyrics**. By leveraging state-of-the-art natural language processing and vector embeddings, the app understands the meaning and emotion behind your search queries and matches them with songs that truly resonate with what you're looking for.

Whether you're searching for "a heartbreak song that makes me cry" or "an upbeat motivational anthem," the AI understands your intent and delivers meaningful recommendations.

## Key Features

### Semantic Search Technology
- **Natural Language Queries**: Search using everyday language and descriptive phrases
- **Context-Aware Matching**: Understands mood, emotion, and thematic elements
- **Vector Embeddings**: Uses HuggingFace transformer models for accurate semantic representation
- **ChromaDB Integration**: Efficient vector similarity search across thousands of songs

### Interactive User Interface
- **Clean Modern Design**: Gradient-themed interface with intuitive navigation
- **Multiple View Modes**: Toggle between card view and table view for results
- **Real-Time Search**: Fast semantic search with progress indicators
- **Example Queries**: Pre-built search suggestions to inspire exploration

### Comprehensive Results
- **Similarity Scoring**: Each recommendation comes with a quantitative similarity score
- **Lyrics Preview**: Snippet view of relevant lyrics sections
- **Full Lyrics Access**: Expandable sections to read complete song lyrics
- **Visual Analytics**: Interactive charts showing similarity distributions
- **Artist Analysis**: Statistics on unique artists and recommendation patterns

### Performance Optimization
- **Smart Caching**: Database and metadata cached for instant subsequent searches
- **Efficient Embeddings**: Optimized HuggingFace model for speed and accuracy
- **Progress Feedback**: Real-time updates during search operations
- **Scalable Architecture**: Handles databases with thousands of songs

##  Use Cases

### Personal Music Discovery
- Find songs that match your current emotional state
- Discover new music similar to your favorite lyrics
- Create playlists around specific themes or moods

### Music Research
- Analyze lyrical themes across different artists
- Study sentiment patterns in song lyrics
- Compare similar songs from different genres or eras

### Content Creation
- Find perfect background music for videos or podcasts
- Curate themed playlists for events
- Discover songs with specific lyrical content

### Educational Applications
- Study sentiment analysis in music
- Explore NLP and vector search technologies
- Understand modern recommendation systems

## Technology Stack

### Core Technologies
- **Streamlit**: Modern web application framework for Python
- **LangChain**: Framework for building LLM-powered applications
- **ChromaDB**: Open-source embedding database for vector search
- **HuggingFace Transformers**: State-of-the-art NLP models

### Key Libraries
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **Sentence Transformers**: Text embedding generation
- **Collections**: Data structure utilities for analysis

### AI/ML Components
- **all-MiniLM-L6-v2**: Efficient sentence embedding model
- **Vector Similarity Search**: Cosine similarity for semantic matching
- **Document Embeddings**: Converts lyrics into high-dimensional vectors

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB+ RAM recommended
- Internet connection (for initial model download)

### Quick Setup

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/song-recommender.git
cd song-recommender
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare Your Data**
```bash
# Place your songs CSV file in the project directory
# Expected format: columns for 'text' (lyrics), 'song' (title), 'artist'
```

5. **Build Vector Database** (First-time setup)
```python
python setup_database.py
```

### Required Packages

```txt
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.14.0
langchain>=0.1.0
langchain-huggingface>=0.0.1
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

## 🎮 Usage

### Starting the Application

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

### Basic Workflow

1. **Enter Your Query**: Type a description of the song you want
   - Example: "A sad song about missing someone special"
   
2. **Click Search**: The AI processes your query and searches the database

3. **Explore Results**: View top 5 recommendations with similarity scores

4. **Read Lyrics**: Expand any song to view full lyrics

5. **Analyze Patterns**: Check the similarity chart and statistics

### Example Queries

**Emotional Themes:**
- "A song about heartbreak and lost love"
- "Upbeat happy song to boost my mood"
- "Nostalgic song about childhood memories"

**Situational Searches:**
- "Motivational anthem for overcoming challenges"
- "Romantic ballad for a special moment"
- "Calming acoustic song for relaxation"

**Genre-Specific:**
- "Rock song with powerful guitar solos"
- "Electronic dance music with high energy"
- "Folk song with storytelling lyrics"

## 📊 How It Works

### 1. Data Processing Pipeline

**Data Ingestion**:
- Loads song lyrics from CSV files
- Extracts metadata (title, artist, lyrics)
- Creates structured document objects

**Text Preprocessing**:
- Cleans and normalizes lyrics text
- Prepares data for embedding generation

### 2. Vector Database Creation

**Embedding Generation**:
- Converts lyrics into 384-dimensional vectors
- Uses HuggingFace's all-MiniLM-L6-v2 model
- Captures semantic meaning of lyrics

**Database Storage**:
- Stores vectors in ChromaDB
- Maintains metadata associations
- Enables efficient similarity search

### 3. Search & Retrieval

**Query Processing**:
- User query converted to vector embedding
- Similarity computed against all songs
- Top-K most similar songs retrieved

**Ranking & Scoring**:
- Cosine similarity for relevance
- Distance metrics for ranking
- Normalized scores for interpretation

### 4. Result Presentation

**Display Logic**:
- Formatted cards with key information
- Interactive visualizations
- Expandable detailed views

## Configuration Options

### Database Settings

```python
# Modify in vector_search_functions.py

PERSIST_DIRECTORY = "./chroma_songs_db"  # Database location
COLLECTION_NAME = "lyrics"                # Collection identifier
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # HuggingFace model
```

### Search Parameters

```python
# Adjust in app.py or functions

TOP_K = 5              # Number of recommendations
SNIPPET_LENGTH = 200   # Preview text length
CACHE_TTL = 3600      # Cache duration (seconds)
```

### UI Customization

Modify the custom CSS in `app.py` to change:
- Color schemes and gradients
- Card styling and layouts
- Font sizes and spacing
- Animation effects

## 📈 Performance Considerations

### Optimization Tips

**For Large Databases (50,000+ songs)**:
- Initial search may take 30-60 seconds
- Subsequent searches are cached and instant
- Consider database partitioning for very large datasets

**Memory Usage**:
- Embedding model: ~100MB
- ChromaDB: Scales with dataset size
- Recommended: 4GB RAM for 100k songs

**Speed Improvements**:
- Use GPU acceleration if available
- Reduce TOP_K for faster results
- Pre-compute embeddings offline

### Scalability

Current implementation efficiently handles:
- Up to 100,000 songs
- Real-time semantic search
- Multiple concurrent users (with caching)

For larger datasets, consider:
- Distributed vector databases (Pinecone, Weaviate)
- Approximate nearest neighbor search (FAISS)
- Database sharding strategies

## Learning Outcomes

This project demonstrates proficiency in:

- **Natural Language Processing**: Text embeddings and semantic search
- **Vector Databases**: ChromaDB implementation and optimization
- **Web Development**: Streamlit for rapid prototyping
- **Data Engineering**: ETL pipelines and data management
- **Machine Learning**: Transformer models and similarity metrics
- **UI/UX Design**: Interactive and responsive interfaces

## Future Enhancements

### Planned Features
- [ ] Multi-language support for global music
- [ ] Audio feature integration (tempo, key, mood)
- [ ] Collaborative playlist generation
- [ ] User preference learning over time
- [ ] Export recommendations to Spotify/Apple Music
- [ ] Advanced filters (genre, year, artist)
- [ ] Sentiment analysis visualization
- [ ] Real-time lyrics streaming

### Technical Improvements
- [ ] GPU acceleration for embeddings
- [ ] Hybrid search (semantic + keyword)
- [ ] A/B testing framework
- [ ] API development for external integration
- [ ] Mobile-responsive design
- [ ] Multi-modal search (lyrics + audio)

## Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
- Report bugs or issues
- Suggest new features
- Improve documentation
- Enhance UI/UX design
- Optimize performance
-  Add test coverage

### Contribution Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **HuggingFace**: For providing excellent embedding models
- **ChromaDB**: For the efficient vector database
- **Streamlit**: For the amazing web framework
- **LangChain**: For simplifying LLM application development
- **Open Source Community**: For inspiration and support
**Built with ❤️ and AI**  
**Last Updated**: December 2025  
**Version**: 1.0.0
