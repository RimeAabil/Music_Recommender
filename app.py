import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import time

# Import functions from our vector search module
from vector_search_functions import (
    load_songs_data,
    load_existing_database,
    get_top_songs
)

# Page configuration
st.set_page_config(
    page_title="Song Recommender",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .song-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .song-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .song-artist {
        font-size: 1.1rem;
        color: #667eea;
        margin-bottom: 0.8rem;
    }
    .lyrics-snippet {
        font-style: italic;
        color: #666;
        line-height: 1.6;
        border-left: 3px solid #eee;
        padding-left: 1rem;
        margin-top: 1rem;
    }
    .search-stats {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vector_database():
    """Load the ChromaDB vector database with caching for performance"""
    try:
        st.info("Loading ChromaDB database...")
        
        # Use the function from vector_search_functions.py
        db, embedding_function = load_existing_database()
        
        # Test if database has data
        try:
            collection = db._collection
            count = collection.count()
            st.success(f"Database loaded successfully with {count:,} songs")
            
            if count > 50000:
                st.warning(f"Large database detected ({count:,} songs). Search may take longer.")
                
        except Exception:
            st.info("Database loaded (count unavailable)")
            
        return db, embedding_function
        
    except FileNotFoundError:
        st.error("ChromaDB directory './chroma_songs_db' not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return None, None

@st.cache_data
def load_songs_metadata():
    """Load songs metadata if available using the imported function"""
    try: 
        df = load_songs_data('sentiment_analysis.csv')
        if df is not None:
            st.success(f"Loaded metadata for {len(df)} songs")
        return df
    except Exception as e:
        st.warning(f"Songs metadata file not found: {str(e)}")
        return None

def get_song_recommendations(query, db):
    """Get song recommendations with simple error handling"""
    if not db:
        st.error("Database is not loaded")
        return []
    
    try:
        st.info(f"Searching for: '{query}'")
        st.info("This may take 30-60 seconds for large databases...")
        
        # Use the get_top_songs function with fixed top_k=5
        recommendations = get_top_songs(query, db, top_k=5)
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recommendations!")
        else:
            st.info("Search completed - no recommendations found")
            
        return recommendations
        
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        st.error("Try refreshing the page or restarting the app")
        return []

def create_similarity_chart(recommendations):
    """Create a similarity score visualization"""
    if not recommendations:
        return None
    
    df = pd.DataFrame(recommendations)
    
    fig = px.bar(
        df, 
        x='similarity_score', 
        y=[f"{row['title'][:30]}..." if len(row['title']) > 30 else row['title'] for row in recommendations],
        orientation='h',
        title="Similarity Scores",
        color='similarity_score',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Similarity Score",
        yaxis_title="Songs",
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig

def analyze_recommendations(recommendations):
    """Analyze and display statistics about recommendations"""
    if not recommendations:
        return
    
    # Artist frequency
    artists = [rec['artist'] for rec in recommendations]
    artist_counts = Counter(artists)
    
    # Average similarity
    avg_similarity = sum(rec['similarity_score'] for rec in recommendations) / len(recommendations)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Songs", len(recommendations))
    
    with col2:
        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
    
    with col3:
        st.metric("Unique Artists", len(artist_counts))

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎵 AI Song Recommender</h1>
            <p>Discover songs that match your mood using semantic vector search</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load database
    with st.spinner("Loading vector database..."):
        db, embedding_function = load_vector_database()
    
    if not db:
        st.error("Failed to load database. Please check:")
        st.error("1. Database files exist in './chroma_songs_db'")
        st.error("2. All required packages are installed")
        st.stop()
    
    # Load additional data
    songs_df = load_songs_metadata()
    
    # Display database info
    if songs_df is not None:
        st.info(f"📊 Database contains {len(songs_df)} songs")
    
    # Example queries section
    st.subheader("💡 Example Queries")
    example_queries = [
        "A song about heartbreak and lost love",
        "Upbeat dance music to make me happy", 
        "Sad emotional ballad about missing someone",
        "Motivational song about overcoming challenges",
        "Romantic love song for a special moment",
        "Nostalgic song about childhood memories",
        "Rock song with powerful guitar solos",
        "Gentle acoustic song for relaxation"
    ]
    
    # Display example queries as buttons
    cols = st.columns(4)
    for i, query in enumerate(example_queries):
        with cols[i % 4]:
            if st.button(query[:25] + "...", key=f"ex{i+1}"):
                st.session_state.selected_query = query
    
    # Get selected query from session state
    selected_query = st.session_state.get('selected_query', '')
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "🔍 Describe the type of song you're looking for:",
            value=selected_query,
            height=100,
            placeholder="e.g., 'A song that talks about love and heartbreaks' or 'Upbeat song to boost my mood'",
            help="Describe the mood, theme, or feeling you want in a song"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # More spacing
        search_button = st.button("🎯 Find Songs", type="primary")
        if st.button("🔄 Clear"):
            st.session_state.selected_query = ''
            st.rerun()
    
    # Search functionality
    if search_button and query.strip():
        start_time = time.time()
        
        with st.spinner("🔍 Searching for similar songs..."):
            recommendations = get_song_recommendations(query, db)
            
        search_time = time.time() - start_time
        
        # Display search stats
        st.markdown(f"""
            <div class="search-stats">
                <strong>Search completed in {search_time:.2f} seconds</strong><br>
                Query: "{query}"<br>
                Found {len(recommendations)} recommendations
            </div>
        """, unsafe_allow_html=True)
        
        if recommendations:
            # Display analysis
            analyze_recommendations(recommendations)
            
            # Similarity chart
            fig = create_similarity_chart(recommendations)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display recommendations
            st.header("🎼 Recommended Songs")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Card View", "Table View"])
            
            with tab1:
                for rec in recommendations:
                    st.markdown(f"""
                        <div class="song-card">
                            <div class="song-title">#{rec['rank']} {rec['title']}</div>
                            <div class="song-artist">by {rec['artist']}</div>
                            <strong>Similarity Score:</strong> {rec['similarity_score']:.3f}
                            <div class="lyrics-snippet">"{rec['lyrics_snippet']}"</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Expandable full lyrics
                    with st.expander(f"View full lyrics - {rec['title']}"):
                        st.text(rec['full_lyrics'])
            
            with tab2:
                # Table view
                df_results = pd.DataFrame(recommendations)
                df_display = df_results[['rank', 'title', 'artist', 'similarity_score']].copy()
                df_display.columns = ['Rank', 'Title', 'Artist', 'Similarity Score']
                st.dataframe(df_display, use_container_width=True)
                
        else:
            st.warning("No songs found matching your query.")
            st.info("Try:")
            st.info("• More descriptive terms")
            st.info("• Different keywords")
            st.info("• One of the example queries above")
    
    elif search_button and not query.strip():
        st.info("Please enter a description of the song you're looking for.")
    
    # Additional information at the bottom
    with st.expander("ℹ️ About This App"):
        st.markdown("""
            This app uses vector search to find songs that match your description semantically.
            Returns the top 5 most similar songs based on your query.
        """)
    
    # Database management
    with st.expander("🛠️ Database Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Database"):
                st.cache_resource.clear()
                st.success("Cache cleared - please refresh the page")
        
        with col2:
            if st.button("📊 Show Database Stats"):
                if db:
                    try:
                        collection = db._collection
                        count = collection.count()
                        st.success(f"Total songs: {count:,}")
                        if songs_df is not None:
                            st.info(f"Metadata: {len(songs_df):,} songs")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()