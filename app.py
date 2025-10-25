import streamlit as st
from openai import OpenAI
import os
import numpy as np
import requests
import json
import time
import random
from bs4 import BeautifulSoup
import re

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
    short_query = query[:200] if len(query) > 200 else query
    try:
        time.sleep(0.5)
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": short_query,
            "limit": limit,
            "fields": "title,abstract,url,authors,year,venue,paperId"
        }
        headers = {"User-Agent": "GapFindr.AI Research Tool (Educational Hackathon Project)"}
        response = requests.get(url, params=params, headers=headers, timeout=20)
        
        if response.status_code == 429:
            st.error("**Rate Limited**: Semantic Scholar API is temporarily unavailable. Please wait a few minutes.")
            return None
        if response.status_code == 400:
            st.error("**Invalid Query**: Try rephrasing or being more specific.")
            return None
        if response.status_code != 200:
            st.error(f"**API Error** (Status {response.status_code})")
            return None

        data = response.json()
        if not data.get("data"):
            st.warning("No papers found. Try a broader query or enable Demo Mode.")
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
    except Exception as e:
        st.error(f"**Error**: {str(e)}")
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

# -----------------------------

# PubMed-based QuickScan Mode (no OpenAI)

# -----------------------------

def fetch_pubmed_abstracts(query, max_results=8):
    """Fetch abstracts from PubMed API"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
    res = requests.get(search_url).json()
    ids = ",".join(res.get("esearchresult", {}).get("idlist", []))
    if not ids:
        return []
    fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
    data = requests.get(fetch_url).text
    soup = BeautifulSoup(data, "xml")
    abstracts = []
    for article in soup.find_all("PubmedArticle"):
        title = article.find("ArticleTitle").text if article.find("ArticleTitle") else ""
        abstract = article.find("AbstractText").text if article.find("AbstractText") else ""
        if abstract:
            abstracts.append({"title": title, "abstract": abstract})
    return abstracts

def extract_limitations(text):
    """Find sentences that indicate limitations"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    keywords = ["limitation", "future work", "further research", "did not", "cannot", "bias", "small sample"]
    limit_sents = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
    return limit_sents

def summarize_limitations(limitations):
    """Naive summary of recurring limitation themes"""
    if not limitations:
        return "No explicit limitations found — may require deeper text analysis."
    joined = " ".join(limitations)
    return f"Common limitation themes: {joined[:400]}..."

# -----------------------------

# DEMO DATA (fallback)

# -----------------------------

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
    }
]

# -----------------------------

# MAIN APP

# -----------------------------

st.markdown('<h1 class="main-header">GapFindr.AI</h1>', unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Illuminate the unexplored. Discover what's missing in your field.</p>", unsafe_allow_html=True)


# --- QuickScan Section (no OpenAI) ---

st.markdown("### 🧠 Quick Literature Scan (No API Key Needed)")
topic_query = st.text_input("Enter a topic (e.g., Parkinson's Disease):", "")

if st.button("⚡ Quick Scan for Limitations", use_container_width=True):
    if not topic_query.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Fetching abstracts and scanning for limitations..."):
            papers = fetch_pubmed_abstracts(topic_query)
            if not papers:
                st.error("No papers found. Try a broader topic.")
            else:
                all_limits = []
                for p in papers:
                    limits = extract_limitations(p["abstract"])
                    all_limits.extend(limits)
                summary = summarize_limitations(all_limits)
                
                st.subheader("Summary of Common Limitations")
                st.write(summary)

                st.subheader("Analyzed Papers")
                for p in papers:
                    st.markdown(f"**{p['title']}**")
                    st.caption(p['abstract'])
                    limits = extract_limitations(p["abstract"])
                    if limits:
                        st.write("Limitations:")
                        for l in limits:
                            st.write(f"- {l}")
                    st.markdown("---")

st.markdown("<hr>", unsafe_allow_html=True)

# --- Full Gap Analysis Section ---

user_abstract = st.text_area(
"✦ Your Research Abstract",
placeholder="Paste your paper abstract here to discover research gaps and related work...",
height=200,
)

use_demo_mode = st.checkbox("🎯 Use Demo Mode (recommended for hackathon if API fails)", value=False)

if st.button("🔍 Find Research Gaps", type="primary", use_container_width=True):
    if not user_abstract.strip():
        st.warning("⚠️ Please enter an abstract first!")
    else:
        with st.spinner("📖 Analyzing your manuscript..."):
            user_embedding = get_embedding(user_abstract)
            if user_embedding is None:
                st.error("Embedding failed. Check your API key.")
                st.stop()

        similar_papers = None
        if not use_demo_mode:
            with st.spinner("🔍 Searching academic databases..."):
                similar_papers = search_semantic_scholar(user_abstract, limit=8)

        if similar_papers is None or use_demo_mode:
            if not use_demo_mode:
                st.info("Using demo papers. Enable Demo Mode to skip API calls.")
            similar_papers = DEMO_PAPERS.copy()

        with st.spinner("Computing similarity scores..."):
            paper_abstracts = [p["abstract"] for p in similar_papers]
            embeddings = get_batch_embeddings(paper_abstracts)
            if embeddings:
                for i, paper in enumerate(similar_papers):
                    paper["embedding"] = embeddings[i]
            else:
                st.error("Failed to generate embeddings for papers.")
                st.stop()

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = []
        for paper in similar_papers:
            if paper.get("embedding"):
                score = cosine_similarity(user_embedding, paper["embedding"])
                gap_score = max(1.0, min(10.0, 10.0 - (score * 9)))
                similarities.append((score, gap_score, paper))

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_papers = similarities[:5]

        st.markdown("---")
        st.markdown("## Related Manuscripts")
        st.markdown(f"*Found {len(top_papers)} relevant papers*")

        for idx, (sim_score, gap_score, paper) in enumerate(top_papers):
            with st.expander(f"{paper['title']}", expanded=(idx == 0)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if paper.get('authors'):
                        authors_str = ", ".join(paper['authors'][:3])
                        if len(paper['authors']) > 3:
                            authors_str += " et al."
                        st.markdown(f"**Authors:** {authors_str}")
                    if paper.get('year') or paper.get('venue'):
                        info = []
                        if paper.get('year'): info.append(str(paper['year']))
                        if paper.get('venue'): info.append(paper['venue'])
                        st.markdown(f"**Published:** {' • '.join(info)}")
                    st.markdown(f"**Abstract:** {paper['abstract']}")
                    if paper.get('url'):
                        st.markdown(f"[🔗 View Original Manuscript]({paper['url']})")
                with col2:
                    st.markdown(f"<div class='similarity-score'>Relevance: {sim_score:.1%}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='gap-badge'>Gap Score: {gap_score:.1f}/10</div>", unsafe_allow_html=True)

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

st.markdown("---")
st.markdown("**Scholar's Note:** This tool uses AI to identify research gaps. Always verify findings with domain experts.")
if not use_demo_mode:
    st.markdown("**API Notice:** Semantic Scholar has rate limits. Enable Demo Mode if errors occur.")
