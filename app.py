import streamlit as st
from openai import OpenAI
import os
import numpy as np

# Page config
st.set_page_config(
    page_title="GapFindr.AI",
    layout="wide"
)

# Custom CSS for light vintage library theme
st.markdown("""
<style>
    /* Import a serif font for that classic book feel */
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600;700&family=Libre+Baskerville:wght@400;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f9f6f0 0%, #faf7f2 50%, #f5f0e8 100%);
        font-family: 'Crimson Text', serif;
    }
    
    /* Main header with rich gold */
    .main-header {
        background: linear-gradient(90deg, #8b6914, #b8860b, #cd7f32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Libre Baskerville', serif;
        text-shadow: 2px 2px 4px rgba(139, 105, 20, 0.1);
        letter-spacing: 0.05em;
    }
    
    .subtitle {
        color: #6b5344;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-style: italic;
        font-family: 'Crimson Text', serif;
    }
    
    /* Textarea styling - cream paper look */
    .stTextArea textarea {
        background: linear-gradient(to bottom, #fffef9 0%, #fdfbf5 100%) !important;
        border: 2px solid #c9a961 !important;
        border-radius: 8px !important;
        color: #2d2317 !important;
        font-family: 'Crimson Text', serif !important;
        font-size: 1.05rem !important;
        box-shadow: 0 2px 8px rgba(139, 105, 20, 0.08), inset 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #8b7355 !important;
        opacity: 0.6 !important;
    }
    
    .stTextArea label {
        color: #8b6914 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        font-family: 'Libre Baskerville', serif !important;
    }
    
    /* Paper card - warm wood panel */
    .paper-card {
        background: linear-gradient(135deg, #faf7f2 0%, #f5f0e8 100%);
        border: 2px solid #d4c5b0;
        border-left: 6px solid #b8860b;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.4s ease;
        box-shadow: 0 4px 12px rgba(139, 105, 20, 0.1);
        position: relative;
    }
    
    .paper-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #b8860b, transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .paper-card:hover {
        border-color: #b8860b;
        box-shadow: 0 8px 24px rgba(184, 134, 11, 0.15), 0 0 40px rgba(205, 127, 50, 0.08);
        transform: translateX(8px);
        background: linear-gradient(135deg, #fffef9 0%, #faf7f2 100%);
    }
    
    .paper-card:hover::before {
        opacity: 1;
    }
    
    /* Gap badge - rich leather accent */
    .gap-badge {
        display: inline-block;
        background: linear-gradient(135deg, #8b4513, #a0522d);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #fdfbf5;
        border: 2px solid #6b3410;
        box-shadow: 0 2px 6px rgba(107, 52, 16, 0.2), inset 0 1px 2px rgba(255,255,255,0.15);
        font-family: 'Libre Baskerville', serif;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Similarity score - warm bronze */
    .similarity-score {
        color: #b8860b;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: 'Libre Baskerville', serif;
        text-shadow: 0 1px 2px rgba(184, 134, 11, 0.1);
    }
    
    /* Insight box - highlighted parchment */
    .insight-box {
        background: linear-gradient(to right, rgba(255, 250, 240, 0.98), rgba(255, 248, 230, 0.98));
        border-left: 5px solid #b8860b;
        border-right: 1px solid #d4c5b0;
        border-top: 1px solid #e8dcc3;
        border-bottom: 1px solid #e8dcc3;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(139, 105, 20, 0.12), inset 0 1px 2px rgba(184, 134, 11, 0.05);
        color: #2d2317;
        font-family: 'Crimson Text', serif;
        font-size: 1.05rem;
        line-height: 1.8;
        position: relative;
    }
    
    .insight-box::before {
        content: '✦';
        position: absolute;
        top: 1rem;
        left: -2.5px;
        color: #b8860b;
        font-size: 1.5rem;
    }
    
    /* Button styling - warm gold */
    .stButton button {
        background: linear-gradient(135deg, #b8860b 0%, #daa520 100%) !important;
        color: #fdfbf5 !important;
        border: 2px solid #8b6914 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-family: 'Libre Baskerville', serif !important;
        padding: 0.75rem 2rem !important;
        box-shadow: 0 4px 12px rgba(184, 134, 11, 0.25), inset 0 1px 2px rgba(255,255,255,0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #cd7f32 0%, #e5a93b 100%) !important;
        box-shadow: 0 6px 20px rgba(205, 127, 50, 0.3), inset 0 1px 2px rgba(255,255,255,0.4) !important;
        transform: translateY(-2px) !important;
        border-color: #b8860b !important;
    }
    
    /* Expander styling - light wood panels */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #faf7f2 0%, #f5f0e8 100%) !important;
        border: 2px solid #d4c5b0 !important;
        border-radius: 8px !important;
        color: #8b6914 !important;
        font-family: 'Libre Baskerville', serif !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #b8860b !important;
        background: linear-gradient(135deg, #fffef9 0%, #faf7f2 100%) !important;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(to bottom, #fdfbf5 0%, #faf7f2 100%) !important;
        border: 2px solid #d4c5b0 !important;
        border-top: none !important;
        color: #2d2317 !important;
    }
    
    /* Markdown headers in results */
    h2 {
        color: #8b6914 !important;
        font-family: 'Libre Baskerville', serif !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #d4c5b0 !important;
        padding-bottom: 0.5rem !important;
        margin-top: 2rem !important;
    }
    
    /* Horizontal rules - decorative divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(to right, transparent, #d4c5b0, #b8860b, #d4c5b0, transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* Warning and info boxes */
    .stWarning, .stInfo {
        background: rgba(255, 250, 240, 0.6) !important;
        border-left: 4px solid #cd7f32 !important;
        color: #2d2317 !important;
        font-family: 'Crimson Text', serif !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #8b6914 !important;
        font-family: 'Crimson Text', serif !important;
        font-size: 1.1rem !important;
    }
    
    /* Links */
    a {
        color: #b8860b !important;
        text-decoration: none !important;
        border-bottom: 1px solid #cd7f32 !important;
        transition: all 0.3s ease !important;
    }
    
    a:hover {
        color: #8b6914 !important;
        border-bottom-color: #8b6914 !important;
    }
    
    /* Footer tip */
    .stMarkdown p {
        color: #6b5344 !important;
        font-family: 'Crimson Text', serif !important;
    }
    
    /* General text color override */
    p, span, div {
        color: #2d2317;
    }
    
    /* Markdown text in expanders */
    .streamlit-expanderContent p, 
    .streamlit-expanderContent span,
    .streamlit-expanderContent strong {
        color: #2d2317 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    st.error("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    st.stop()

# Cache functions to save API calls
@st.cache_data(show_spinner=False)
def get_embedding(text):
    """Generate embedding with caching"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"❌ Embedding error: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def analyze_gaps(user_abstract, related_summaries):
    """Generate gap analysis with caching"""
    try:
        prompt = f"""You are an academic research assistant analyzing research gaps.

Given this research abstract:
{user_abstract}

And these related papers:
{related_summaries}

Provide:
1. **Three Research Gaps** - Specific areas not adequately covered
2. **Two Research Directions** - Novel approaches or combinations worth exploring

Format your response clearly with headers and bullet points."""

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return result.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        return None

# Header with vintage book ornament
st.markdown('<h1 class="main-header">GapFindr.AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Illuminate the unexplored. Discover what\'s missing in your field.</p>', unsafe_allow_html=True)

# Input section
user_abstract = st.text_area(
    "✦ Your Research Abstract",
    placeholder="Paste your paper abstract here to discover research gaps and related work...",
    height=200,
    help="Enter the abstract of your research paper"
)

# Example papers database (replace with real embeddings in production)
example_papers = [
    {
        "title": "Machine Learning for Healthcare Diagnostics",
        "abstract": "We explore ML algorithms applied to medical diagnosis and prediction of patient outcomes...",
        "url": "https://arxiv.org/abs/1234.5678",
        "embedding": np.random.rand(1536)
    },
    {
        "title": "Deep Learning Approaches in Natural Language Processing",
        "abstract": "This paper studies deep learning methods for NLP tasks including sentiment analysis and translation...",
        "url": "https://arxiv.org/abs/2345.6789",
        "embedding": np.random.rand(1536)
    },
    {
        "title": "Advances in Reinforcement Learning for Robotics",
        "abstract": "Reinforcement learning techniques for robotic manipulation and navigation have seen rapid growth...",
        "url": "https://arxiv.org/abs/3456.7890",
        "embedding": np.random.rand(1536)
    },
    {
        "title": "Computer Vision in Autonomous Vehicles",
        "abstract": "We present novel computer vision algorithms for object detection and scene understanding in self-driving cars...",
        "url": "https://arxiv.org/abs/4567.8901",
        "embedding": np.random.rand(1536)
    }
]

# Analyze button
if st.button("🔍 Find Research Gaps", type="primary", use_container_width=True):
    if not user_abstract.strip():
        st.warning("⚠️ Please enter an abstract first!")
    else:
        with st.spinner("📖 Analyzing your manuscript..."):
            user_embedding = get_embedding(user_abstract)
        
        if user_embedding is None:
            st.error("Failed to generate embedding. Check your API key and quota.")
            st.stop()
        
        # Cosine similarity function
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Find similar papers
        similarities = []
        for paper in example_papers:
            score = cosine_similarity(user_embedding, paper["embedding"])
            gap_score = np.random.uniform(6.0, 9.5)
            similarities.append((score, gap_score, paper))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_papers = similarities[:5]
        
        # Display related papers
        st.markdown("---")
        st.markdown("## 📜 Related Manuscripts")
        st.markdown("*Explore the archive*")
        
        for idx, (sim_score, gap_score, paper) in enumerate(top_papers):
            with st.expander(f"📖 {paper['title']}", expanded=(idx == 0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Abstract:** {paper['abstract']}")
                    st.markdown(f"[🔗 View Original Manuscript]({paper['url']})")
                
                with col2:
                    st.markdown(f"<div class='similarity-score'>Relevance: {sim_score:.1%}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='gap-badge'>Gap Score: {gap_score:.1f}/10</div>", unsafe_allow_html=True)
        
        # Generate gap analysis
        st.markdown("---")
        st.markdown("## ✦ Scholarly Insights")
        
        with st.spinner("🪶 Composing analysis..."):
            related_summaries = "\n\n".join([
                f"**{p['title']}**\n{p['abstract']}" 
                for _, _, p in top_papers
            ])
            
            ai_output = analyze_gaps(user_abstract, related_summaries)
        
        if ai_output:
            st.markdown(f"<div class='insight-box'>{ai_output}</div>", unsafe_allow_html=True)
        else:
            st.error("Failed to generate analysis. Check your API quota.")

# Footer
st.markdown("---")
st.markdown("📚 **Scholar's Note:** This tool uses AI to identify research gaps. Always verify findings with domain experts.")