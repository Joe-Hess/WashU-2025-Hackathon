import streamlit as st
from openai import OpenAI
import os
import numpy as np
import requests
import json
import time
import random

# Page config
st.set_page_config(
    page_title="GapFindr.AI",
    layout="wide"
)

# Load external CSS
def load_css(file_name):
    """Load CSS from external file"""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Using default styles.")

load_css('styles.css')

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
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
        st.error(f"Embedding error: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def get_batch_embeddings(texts):
    """Generate embeddings for multiple texts efficiently"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Batch embedding error: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def search_semantic_scholar(query, limit=10):
    """Search Semantic Scholar API for similar papers"""
    
    # Extract key terms from query (use first 200 chars)
    short_query = query[:200] if len(query) > 200 else query
    
    try:
        time.sleep(0.5)  # Rate limiting
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": short_query,
            "limit": limit,
            "fields": "title,abstract,url,authors,year,venue,paperId"
        }
        
        headers = {
            "User-Agent": "GapFindr.AI Research Tool (Educational Hackathon Project)"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=20)
        
        if response.status_code == 429:
            st.error("**Rate Limited**: Semantic Scholar API is temporarily unavailable. Please wait 2-3 minutes and try again.")
            st.info("**Tip**: Enable Demo Mode to continue working")
            return None
        
        if response.status_code == 400:
            st.error("**Invalid Query**: Try rephrasing your abstract or making it more specific.")
            return None
            
        if response.status_code != 200:
            st.error(f"**API Error** (Status {response.status_code})")
            return None
        
        data = response.json()
        
        if not data.get("data"):
            st.warning("No papers found in Semantic Scholar. Try a more general query or enable Demo Mode.")
            return None
        
        papers = []
        for paper in data.get("data", [])[:limit]:
            if paper.get("abstract") and len(paper["abstract"]) > 50:
                papers.append({
                    "title": paper.get("title", "No title"),
                    "abstract": paper.get("abstract", "No abstract available"),
                    "url": paper.get("url", ""),
                    "authors": [author.get("name", "") for author in paper.get("authors", [])[:5]],
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "paperId": paper.get("paperId", ""),
                    "embedding": None
                })
        
        return papers
        
    except requests.exceptions.Timeout:
        st.error("**Timeout**: The API took too long to respond. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("**Connection Error**: Please check your internet connection.")
        return None
    except Exception as e:
        st.error(f"**Unexpected Error**: {str(e)}")
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
        st.error(f"Analysis error: {str(e)}")
        return None

# FALLBACK: Demo papers for when API fails
DEMO_PAPERS = [
    {
        "title": "Machine Learning for Healthcare Diagnostics",
        "abstract": "We explore ML algorithms applied to medical diagnosis and prediction of patient outcomes. Our work demonstrates significant improvements in diagnostic accuracy using deep learning approaches on medical imaging data.",
        "url": "https://arxiv.org/abs/1234.5678",
        "authors": ["John Smith", "Jane Doe", "Alice Johnson"],
        "year": "2023",
        "venue": "Nature Medicine",
        "embedding": None
    },
    {
        "title": "Deep Learning Approaches in Natural Language Processing",
        "abstract": "This paper studies deep learning methods for NLP tasks including sentiment analysis and translation. We introduce novel transformer architectures that achieve state-of-the-art results across multiple benchmarks.",
        "url": "https://arxiv.org/abs/2345.6789",
        "authors": ["Bob Wilson", "Carol Martinez"],
        "year": "2024",
        "venue": "ACL",
        "embedding": None
    },
    {
        "title": "Advances in Reinforcement Learning for Robotics",
        "abstract": "Reinforcement learning techniques for robotic manipulation and navigation have seen rapid growth. We present a novel framework for sample-efficient learning in complex environments with sparse rewards.",
        "url": "https://arxiv.org/abs/3456.7890",
        "authors": ["David Lee", "Emma Chen", "Frank Kumar"],
        "year": "2023",
        "venue": "ICRA",
        "embedding": None
    },
    {
        "title": "Computer Vision for Autonomous Vehicles",
        "abstract": "We present novel computer vision algorithms for object detection and scene understanding in self-driving cars. Our approach combines multi-modal sensing with deep learning for robust perception in challenging conditions.",
        "url": "https://arxiv.org/abs/4567.8901",
        "authors": ["Grace Park", "Henry Zhang"],
        "year": "2024",
        "venue": "CVPR",
        "embedding": None
    },
    {
        "title": "Neural Architecture Search for Efficient Models",
        "abstract": "This work explores automated neural architecture search methods to design efficient deep learning models. We propose a novel search strategy that balances accuracy and computational cost for edge deployment.",
        "url": "https://arxiv.org/abs/5678.9012",
        "authors": ["Isabel Rodriguez", "Jack Thompson"],
        "year": "2023",
        "venue": "NeurIPS",
        "embedding": None
    },
]

# Header
st.markdown('<h1 class="main-header">GapFindr.AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Illuminate the unexplored. Discover what\'s missing in your field.</p>', unsafe_allow_html=True)

# Input section
user_abstract = st.text_area(
    "✦ Your Research Abstract",
    placeholder="Paste your paper abstract here to discover research gaps and related work...",
    height=200,
    help="Enter the abstract of your research paper"
)

# Demo mode toggle
use_demo_mode = st.checkbox("🎯 Use Demo Mode (recommended for hackathon if API fails)", value=False)

# Analyze button
if st.button("🔍 Find Research Gaps", type="primary", use_container_width=True):
    if not user_abstract.strip():
        st.warning("⚠️ Please enter an abstract first!")
    else:
        # Generate embedding for user's abstract
        with st.spinner("📖 Analyzing your manuscript..."):
            user_embedding = get_embedding(user_abstract)
            
            if user_embedding is None:
                st.error("Failed to generate embedding for your abstract. Check your OpenAI API key and quota.")
                st.stop()
        
        # Search for papers or use demo
        similar_papers = None
        if not use_demo_mode:
            with st.spinner("🔍 Searching academic databases..."):
                similar_papers = search_semantic_scholar(user_abstract, limit=8)
        
        # Fallback to demo papers if needed
        if similar_papers is None or use_demo_mode:
            if not use_demo_mode:
                st.info("**Using demo papers** for your analysis. Enable 'Demo Mode' checkbox to skip API calls.")
            similar_papers = DEMO_PAPERS.copy()
        
        # Generate embeddings for papers in batch
        with st.spinner("Computing similarity scores..."):
            paper_abstracts = [p["abstract"] for p in similar_papers]
            embeddings = get_batch_embeddings(paper_abstracts)
            
            if embeddings:
                for i, paper in enumerate(similar_papers):
                    paper["embedding"] = embeddings[i]
            else:
                st.error("Failed to generate embeddings for papers.")
                st.stop()
        
        # Cosine similarity function
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Calculate similarities
        similarities = []
        for paper in similar_papers:
            if paper.get("embedding"):
                score = cosine_similarity(user_embedding, paper["embedding"])
                gap_score = max(1.0, min(10.0, 10.0 - (score * 9)))
                similarities.append((score, gap_score, paper))
        
        if not similarities:
            st.error("Could not compute similarities. Please try again.")
            st.stop()
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_papers = similarities[:5]
        
        # Display related papers
        st.markdown("---")
        st.markdown("## Related Manuscripts")
        st.markdown(f"*Found {len(top_papers)} relevant papers*")
        
        for idx, (sim_score, gap_score, paper) in enumerate(top_papers):
            with st.expander(f"{paper['title']}", expanded=(idx == 0)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display paper metadata
                    if paper.get('authors'):
                        authors_str = ", ".join(paper['authors'][:3])
                        if len(paper['authors']) > 3:
                            authors_str += " et al."
                        st.markdown(f"**Authors:** {authors_str}")
                    
                    if paper.get('year') or paper.get('venue'):
                        venue_info = []
                        if paper.get('year'):
                            venue_info.append(str(paper['year']))
                        if paper.get('venue'):
                            venue_info.append(paper['venue'])
                        st.markdown(f"**Published:** {' • '.join(venue_info)}")
                    
                    st.markdown(f"**Abstract:** {paper['abstract']}")
                    
                    if paper.get('url'):
                        st.markdown(f"[🔗 View Original Manuscript]({paper['url']})")
                
                with col2:
                    st.markdown(f"<div class='similarity-score'>Relevance: {sim_score:.1%}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='gap-badge'>Gap Score: {gap_score:.1f}/10</div>", unsafe_allow_html=True)
        
        # Generate gap analysis
        st.markdown("---")
        st.markdown("## ✦ Scholarly Insights")
        
        with st.spinner("Composing analysis..."):
            related_summaries = "\n\n".join([
                f"**{p['title']}** ({p.get('year', 'N/A')})\n{p['abstract'][:300]}..." 
                for _, _, p in top_papers
            ])
            
            ai_output = analyze_gaps(user_abstract, related_summaries)
        
        if ai_output:
            st.markdown(f"<div class='insight-box'>{ai_output}</div>", unsafe_allow_html=True)
        else:
            st.error("Failed to generate analysis. Check your API quota.")

# Footer
st.markdown("---")
st.markdown("**Scholar's Note:** This tool uses AI to identify research gaps. Always verify findings with domain experts.")
if not use_demo_mode:
    st.markdown("**API Notice:** Semantic Scholar has rate limits. If errors occur, enable Demo Mode or wait 2-3 minutes.")