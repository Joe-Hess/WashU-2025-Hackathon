import streamlit as st
from openai import OpenAI
import os
import numpy as np

# Page config
st.set_page_config(
    page_title="GapFindr.AI",
    layout="wide"
)

# Custom CSS for dark theme with neon accents
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .paper-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .paper-card:hover {
        border-color: #06b6d4;
        box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
        transform: translateX(4px);
    }
    
    .gap-badge {
        display: inline-block;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 600;
        color: white;
    }
    
    .similarity-score {
        color: #06b6d4;
        font-weight: 600;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(139, 92, 246, 0.1));
        border-left: 4px solid #06b6d4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
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

# Header
st.markdown('<h1 class="main-header">GapFindr.AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Illuminate the unexplored. Discover what\'s missing in your field.</p>', unsafe_allow_html=True)

# Input section
user_abstract = st.text_area(
    "✨ Your Research Abstract",
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
        "embedding": np.random.rand(1536)  # text-embedding-3-small outputs 1536 dimensions
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
        with st.spinner("🤖 Generating embedding for your abstract..."):
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
            gap_score = np.random.uniform(6.0, 9.5)  # Mock gap score
            similarities.append((score, gap_score, paper))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_papers = similarities[:5]
        
        # Display related papers
        st.markdown("---")
        st.markdown("## 🎯 Related Papers")
        st.markdown("*Hover to reveal gaps*")
        
        for idx, (sim_score, gap_score, paper) in enumerate(top_papers):
            with st.expander(f"📄 {paper['title']}", expanded=(idx == 0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Abstract:** {paper['abstract']}")
                    st.markdown(f"[🔗 View Paper]({paper['url']})")
                
                with col2:
                    st.markdown(f"<div class='similarity-score'>Similarity: {sim_score:.1%}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='gap-badge'>Gap Score: {gap_score:.1f}/10</div>", unsafe_allow_html=True)
        
        # Generate gap analysis
        st.markdown("---")
        st.markdown("## ✨ AI-Generated Insights")
        
        with st.spinner("🧠 Analyzing research gaps..."):
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
st.markdown("💡 **Tip:** This tool uses AI to identify research gaps. Always verify findings with domain experts.")