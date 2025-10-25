# PART 1: setup_database.py
# Run this ONCE to create your paper database

import requests
import xml.etree.ElementTree as ET
import json
from openai import OpenAI
import os
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_arxiv_papers(query, max_results=50):
    """Fetch papers from arXiv API"""
    print(f"Searching arXiv for: {query}")
    
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}"
    url = f"{base_url}{search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    
    response = requests.get(url)
    root = ET.fromstring(response.content)
    
    papers = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    for entry in root.findall("atom:entry", ns):
        title_elem = entry.find("atom:title", ns)
        summary_elem = entry.find("atom:summary", ns)
        id_elem = entry.find("atom:id", ns)
        published_elem = entry.find("atom:published", ns)
        
        if title_elem is None or summary_elem is None:
            continue
        
        title = title_elem.text.strip().replace("\n", " ")
        abstract = summary_elem.text.strip().replace("\n", " ")
        url = id_elem.text if id_elem is not None else ""
        year = published_elem.text[:4] if published_elem is not None else "N/A"
        
        authors = []
        for author in entry.findall("atom:author", ns):
            name_elem = author.find("atom:name", ns)
            if name_elem is not None:
                authors.append(name_elem.text)
        
        papers.append({
            "title": title,
            "abstract": abstract,
            "url": url,
            "authors": authors[:5],  # Limit to 5 authors
            "year": year
        })
    
    print(f"Found {len(papers)} papers")
    return papers

def embed_papers(papers, batch_size=50):
    """Generate embeddings for papers in batches"""
    print(f"Generating embeddings for {len(papers)} papers...")
    
    all_embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        abstracts = [p["abstract"] for p in batch]
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(papers)-1)//batch_size + 1}")
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=abstracts
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
        # Rate limiting
        time.sleep(0.5)
    
    # Add embeddings to papers
    for i, paper in enumerate(papers):
        paper["embedding"] = all_embeddings[i]
    
    print("Embeddings generated!")
    return papers

def create_paper_database(topic, num_papers=50):
    """Complete pipeline to create paper database"""
    print(f"\n{'='*60}")
    print(f"Creating GapFindr.AI database for: {topic}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch papers
    papers = fetch_arxiv_papers(topic, max_results=num_papers)
    
    if not papers:
        print("No papers found!")
        return
    
    # Step 2: Generate embeddings
    papers_with_embeddings = embed_papers(papers)
    
    # Step 3: Save to JSON
    filename = "paper_database.json"
    with open(filename, "w") as f:
        json.dump({
            "topic": topic,
            "num_papers": len(papers),
            "papers": papers_with_embeddings
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Database saved to {filename}")
    print(f"   Topic: {topic}")
    print(f"   Papers: {len(papers)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # CONFIGURE YOUR TOPIC HERE
    TOPIC = "large language models code generation"  # Change this!
    NUM_PAPERS = 50
    
    create_paper_database(TOPIC, NUM_PAPERS)