import streamlit as st
import requests
import openai
import os

# Replace this with your actual OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("🧠 Research Gap Finder")
st.write("Paste your research abstract below to find related papers and AI-generated research gaps.")

# Step 1: User input
user_abstract = st.text_area("Your paper abstract:")

if st.button("Analyze"):
    # Step 2: Find related papers via Semantic Scholar
    st.write("🔍 Searching for related papers...")
    response = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/search?query={user_abstract}&limit=5&fields=title,abstract,url"
    )
    data = response.json()
    papers = data.get("data", [])

    if not papers:
        st.write("No related papers found. Try a simpler abstract.")
    else:
        st.write("### Related Papers Found:")
        for p in papers:
            st.markdown(f"- [{p.get('title')}]({p.get('url')})")

        # Step 3: Prepare a list of related papers for AI
        related_summaries = "\n".join(
            [f"Title: {p['title']}\nAbstract: {p.get('abstract', 'N/A')}" for p in papers]
        )

        # Step 4: Send to GPT for analysis
        st.write("🤖 Analyzing gaps...")
        prompt = f"""
        You are an academic research assistant.
        Given this research abstract:
        {user_abstract}

        And these related papers:
        {related_summaries}

        Identify:
        1. Three potential research gaps
        2. Two possible new research directions
        """

        result = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        ai_output = result["choices"][0]["message"]["content"]
        st.write("### AI-Generated Insights:")
        st.write(ai_output)