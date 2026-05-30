import streamlit as st
import asyncio
import os
import requests
import json
from textwrap import dedent
from typing import Any, Callable, Dict, Optional
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Agno Imports
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.team.team import Team
from agno.tools import tool

# --- Streamlit Page Config ------------------------------------------
st.set_page_config(
    page_title="Cast Graph Generator",
    page_icon="🎬",
    layout="wide"
)

# --- CSS / Style ----------------------------------------------------
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border: 1px solid #ccc;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar & Config -----------------------------------------------
with st.sidebar:
    st.header("🔐 Configuration")
    
    load_dotenv()
    env_deepseek = os.getenv("DEEPSEEK_API_KEY")
    env_tmdb = os.getenv("TMDB_TOKEN")
    
    deepseek_api_key = st.text_input(
        "DeepSeek API Key", 
        value=env_deepseek if env_deepseek else "", 
        type="password",
        help="Get this from platform.deepseek.com"
    )
    tmdb_token = st.text_input(
        "TMDb Token (v4 Bearer)", 
        value=env_tmdb if env_tmdb else "", 
        type="password",
        help="Get this from themoviedb.org settings"
    )
    
    if deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
    if tmdb_token:
        os.environ["TMDB_TOKEN"] = tmdb_token
    
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. **Disambiguator**: Cleans title.
    2. **Search**: Finds TMDb ID.
    3. **Credits**: Fetches cast.
    4. **Builder**: Infers relations.
    5. **Mermaid**: Draws graph.
    """)

# --- Constants ------------------------------------------------------
TMDB_BASE = "https://api.themoviedb.org/3"

# --- Tools ----------------------------------------------------------
def get_tmdb_headers():
    token = os.environ.get("TMDB_TOKEN")
    if not token:
        raise ValueError("TMDB_TOKEN not set. Please add it in the sidebar.")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

@tool(name="tmdb_search_title")
def tmdb_search_title(title: str, include_adult: bool = False) -> dict:
    """Search TMDb for a movie/TV show by title using /search/multi."""
    url = f"{TMDB_BASE}/search/multi"
    params = {"query": title, "include_adult": str(include_adult).lower()}
    r = requests.get(url, headers=get_tmdb_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@tool(name="tmdb_fetch_credits")
def tmdb_fetch_credits(media_type: str, tmdb_id: int) -> dict:
    """Fetch TMDb credits for a movie or TV series."""
    if media_type not in {"movie", "tv"}:
        raise ValueError("media_type must be 'movie' or 'tv'")
    url = f"{TMDB_BASE}/{media_type}/{tmdb_id}/credits"
    r = requests.get(url, headers=get_tmdb_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

@tool(name="web_search_stub")
def web_search_stub(query: str) -> dict:
    """Stub web search tool. Replace with real implementation if available."""
    return {"query": query, "results": []}

# --- Agent Factory --------------------------------------------------
@st.cache_resource
def get_agent_team(api_key: str):
    """
    Creates the agent team. Cached to prevent re-initialization on every rerun.
    We pass api_key to ensure cache invalidation if key changes.
    """
    model_instance = DeepSeek(id="deepseek-chat", api_key=api_key)
    
    coordinator = Agent(
        name="Cast Graph Coordinator",
        role="Orchestrate the team and merge results into a final Mermaid graph.",
        model=model_instance,
        instructions=dedent("""
            Coordinate the team. 
            Final output must contain a ```mermaid``` block.
            Do not include conversational filler.
        """),
    )

    title_disambiguator = Agent(
        name="Title Disambiguator",
        role="Normalize input.",
        model=model_instance,
        instructions="Output compact JSON: {title, year, media_type}."
    )

    catalog_search = Agent(
        name="Catalog Search (TMDb)",
        role="Find TMDb candidate.",
        model=model_instance,
        tools=[tmdb_search_title],
        instructions="Return JSON with media id and details."
    )

    credits_fetcher = Agent(
        name="Credits Fetcher (TMDb)",
        role="Fetch cast.",
        model=model_instance,
        tools=[tmdb_fetch_credits],
        instructions="Return JSON with top 15 cast members."
    )

    relationship_builder = Agent(
        name="Relationship Builder",
        role="Infer relationships.",
        model=model_instance,
        tools=[web_search_stub],
        instructions="Build graph JSON with nodes and edges."
    )

    mermaid_formatter = Agent(
        name="Mermaid Formatter",
        role="Convert to Mermaid.",
        model=model_instance,
        instructions="Produce ONLY a Mermaid graph definition string."
    )

    return Team(
        name="Cast Relationship Graph Team",
        model=model_instance,
        members=[
            coordinator, title_disambiguator, catalog_search, 
            credits_fetcher, relationship_builder, mermaid_formatter
        ],
        instructions=["Execute steps in order."],
    )

# --- Helper: Render Mermaid -----------------------------------------
def render_mermaid(code: str, height=600):
    """Renders Mermaid diagram using HTML/JS injection."""
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center;">
        {code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    components.html(html_code, height=height, scrolling=True)

# --- Main Logic -----------------------------------------------------
def main():
    st.title("🎬 Cast Relationship Graph")
    st.markdown("Generate a relationship graph for any Movie or TV Show cast using AI agents.")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Movie / TV Title", placeholder="e.g. Breaking Bad, The Godfather")
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("Generate Graph")

    if run_btn and user_input:
        # 1. Validation
        if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("TMDB_TOKEN"):
            st.error("Please provide API Keys in the sidebar.")
            return

        # 2. Get Team
        try:
            team = get_agent_team(os.environ["DEEPSEEK_API_KEY"])
        except Exception as e:
            st.error(f"Failed to initialize agents: {e}")
            return
        
        prompt = dedent(f"""
            Task: Build a cast relationship graph.
            Input title: "{user_input}"
            Output: A short text summary followed immediately by a Mermaid `graph` diagram.
        """).strip()

        # 3. Setup UI containers
        status_box = st.status("Agents are working...", expanded=True)
        response_container = st.container()

        # 4. Async Execution Wrapper
        async def run_agents():
            full_response = ""
            try:
                # Get the generator
                response_stream = team.run(input=prompt, stream=True)
                
                placeholder = response_container.empty()
                
                for chunk in response_stream:
                    # FIX: Handle NoneType in content stream
                    raw_content = getattr(chunk, 'content', None)
                    
                    if raw_content is None:
                        # Sometimes chunk is just a string in older versions/specific mocks
                        content = chunk if isinstance(chunk, str) else ""
                    else:
                        content = raw_content
                    
                    full_response += content
                    placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                return full_response
            except Exception as e:
                st.error(f"Error executing agents: {str(e)}")
                return None

        # 5. Run Event Loop
        with status_box:
            st.write("Initializing Team...")
            # Create a new loop for Streamlit's thread model
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_text = loop.run_until_complete(run_agents())
            status_box.update(label="Complete!", state="complete", expanded=False)

        # 6. Post-process and Visualize
        if final_text and "```mermaid" in final_text:
            try:
                parts = final_text.split("```mermaid")
                # Handle cases where there might be text after the block
                if len(parts) > 1:
                    graph_code = parts[1].split("```")[0].strip()
                    
                    st.subheader("Graph Visualization")
                    render_mermaid(graph_code)
                else:
                    st.warning("Mermaid tag found, but content was empty.")
            except IndexError:
                st.warning("Mermaid block was detected but could not be parsed.")

if __name__ == "__main__":
    main()