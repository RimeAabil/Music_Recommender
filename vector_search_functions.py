import pandas as pd
from langchain.schema import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

def load_songs_data(csv_path='sentiment_analysis.csv'):
    """Load songs data from CSV"""
    try:
        songs_sentiment_df = pd.read_csv(csv_path)
        return songs_sentiment_df
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found")
        return None

def create_documents_with_metadata(songs_sentiment_df):
    """Create document objects with metadata from dataframe"""
    lyrics_song_artist = []
    
    for _, row in songs_sentiment_df.iterrows():
        lyrics = row['text']
        title = row['song']
        artist = row['artist']
        
        doc = Document(
            page_content=lyrics,
            metadata={
                'title': title,
                'artist': artist
            }
        )
        lyrics_song_artist.append(doc)
    
    return lyrics_song_artist

def create_vector_database(documents, persist_directory="./chroma_songs_db", collection_name="lyrics"):
    """Create and persist ChromaDB vector database"""
    # Use HuggingFace embedding model
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create and persist Chroma DB
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # Save the DB to disk
    db.persist()
    return db, embedding_function

def load_existing_database(persist_directory="./chroma_songs_db", collection_name="lyrics"):
    """Load existing ChromaDB database"""
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name
    )
    
    return db, embedding_function

def get_top_songs(query, db, top_k=3):
    """Get top song recommendations using similarity search"""
    results = db.similarity_search(query, k=top_k)
    
    songs = []
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get('title', 'Unknown Title')
        artist = doc.metadata.get('artist', 'Unknown Artist')
        snippet = doc.page_content[:200].strip().replace('\n', ' ') + "..."
        
        songs.append({
            'rank': i,
            'title': title,
            'artist': artist,
            'lyrics_snippet': snippet,
            'full_lyrics': doc.page_content,
            'similarity_score': 0.9 - (i * 0.1)  # Mock score
        })
    
    return songs

def setup_complete_database(csv_path='sentiment_analysis.csv', 
                           persist_directory="./chroma_songs_db", 
                           collection_name="lyrics"):
    """Complete setup: load data, create documents, and build database"""
    
    # Load data
    print("Loading songs data...")
    songs_df = load_songs_data(csv_path)
    if songs_df is None:
        return None, None
    
    print(f"Loaded {len(songs_df)} songs")
    
    # Create documents with metadata
    print("Creating documents with metadata...")
    documents = create_documents_with_metadata(songs_df)
    
    # Create database
    print("Creating vector database...")
    db, embedding_function = create_vector_database(
        documents, persist_directory, collection_name
    )
    
    print("Database created successfully!")
    return db, embedding_function