import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="GapFindr.ai", layout="wide")

# Load external CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------

@st.cache_data(show_spinner=False)
def extract_keywords(text, top_k=5):
    """Extract key topic terms from user input."""
    import re
    from collections import Counter
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if len(w) > 3]
    common = Counter(words).most_common(top_k)
    return [w for w, _ in common]


@st.cache_data(show_spinner=False)
def search_pubmed(query, retmax=7):
    """Search PubMed and return abstracts, DOIs, and attempt to extract limitations."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base}esearch.fcgi?db=pubmed&term={quote(query)}&retmax={retmax}&retmode=json&sort=relevance"

    # retry logic
    for _ in range(3):
        res = requests.get(search_url)
        if res.status_code == 200:
            break
    else:
        return []

    ids = res.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    ids_str = ",".join(ids)
    fetch_url = f"{base}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
    xml = requests.get(fetch_url).text
    soup = BeautifulSoup(xml, "xml")

    articles = []
    for a in soup.find_all("PubmedArticle"):
        title = a.ArticleTitle.text if a.ArticleTitle else "No Title"
        abstract = a.Abstract.text if a.Abstract else "No Abstract"
        year_tag = a.find("PubDate")
        year = year_tag.Year.text if year_tag and year_tag.Year else "N/A"

        doi, pmc_id = None, None
        for id_elem in a.find_all("ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
            elif id_elem.get("IdType") == "pmc":
                pmc_id = id_elem.text

        limitations_text = "Limitations not found in abstract or available full text."
        if pmc_id:
            pmc_fetch_url = f"{base}efetch.fcgi?db=pmc&id={pmc_id}&retmode=xml"
            pmc_xml = requests.get(pmc_fetch_url).text
            pmc_soup = BeautifulSoup(pmc_xml, "xml")
            full_text = pmc_soup.get_text(" ", strip=True)
            if "limitation" in full_text.lower():
                sentences = [s for s in full_text.split(". ") if "limitation" in s.lower()]
                limitations_text = ". ".join(sentences[:3]) + "."

        elif doi:
            limitations_text = f"Full text may be available at: [https://doi.org/{doi}](https://doi.org/{doi})"

        articles.append({
            "title": title,
            "abstract": abstract,
            "year": year,
            "doi": doi,
            "pmc_id": pmc_id,
            "limitations": limitations_text
        })

    return articles



def generate_research_gap_analysis(user_text, articles):
    """Use Gemini to identify how user's research fits within current landscape."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Missing GEMINI_API_KEY. Please add it as an environment variable."

    genai.configure(api_key=api_key)

    prompt = f"""
    Analyze the following research abstract within its scientific context.

    USER ABSTRACT:
    {user_text}

    EXISTING STUDIES:
    {[a['abstract'] for a in articles]}

    TASKS:
    1. Explain how this abstract fits within the current research landscape.
    2. Identify recurring or notable limitations across the referenced papers.
    3. Suggest how the user's proposed study could advance or refine scientific understanding.
    4. Provide opportunities for funding agencies to support the user's research. Analyze if the proposed topic is realistically fundable. 
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# -----------------------------
# STREAMLIT APP LAYOUT
# -----------------------------
st.markdown("<h1 class='main-header'>GapFindr.ai</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Explore PubMed literature, uncover study limitations, and identify how your research can move science forward.</p>", unsafe_allow_html=True)

user_input = st.text_area("Enter your research abstract or idea:", placeholder="Paste your abstract here...", height=200)

if st.button("Analyze Research Landscape"):
    if not user_input.strip():
        st.warning("Please enter an abstract or topic to begin.")
        st.stop()

    with st.spinner("Extracting key concepts and searching PubMed..."):
        keywords = extract_keywords(user_input)
        query = " AND ".join(keywords)
        articles = search_pubmed(query)

        if not articles:
            top_keyword = keywords[0]
            articles = search_pubmed(top_keyword)

    st.markdown("<h2>Existing Research Results</h2>", unsafe_allow_html=True)
    if not articles:
        st.info("No PubMed results found for this query.")
    else:
        for i, a in enumerate(articles, 1):
            with st.expander(f"{i}. {a['title']} ({a['year']})"):
                st.markdown(f"<div class='paper-card'><strong>Abstract:</strong> {a['abstract']}</div>", unsafe_allow_html=True)
                if a['doi']:
                    st.markdown(f"**DOI:** [https://doi.org/{a['doi']}](https://doi.org/{a['doi']})")
                

    # Research gap analysis
    st.markdown("<h2>Research Gap Analysis</h2>", unsafe_allow_html=True)
    with st.spinner("Generating research gap insights..."):
        analysis = generate_research_gap_analysis(user_input, articles)
        st.markdown(f"<div class='insight-box'>{analysis}</div>", unsafe_allow_html=True)
