#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:25:13 2025

@author: bridgetnevel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Art Explorer AI - Interactive AI Art Historian
"""

import streamlit as st
from google import genai
from google.genai import types
import requests
from urllib.parse import quote_plus
import pandas as pd
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Art Explorer AI",
    page_icon="🎨",
    layout="wide"
)

# Available Gemini models
AVAILABLE_MODELS = {
    "Gemini 2.5 Flash Light": "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.0 Flash": "gemini-2.0-flash",

}

# --- Helper Functions ---
def jina_get(url):
    """Get a website as markdown with Jina Reader."""
    jina_api_key = os.getenv('JINA_READER_KEY')
    if not jina_api_key:
        raise Exception('JINA_READER_KEY env variable not set.')
    
    try:
        st.info(f"🌐 Fetching content from: {url}")
        response = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {jina_api_key}"},
            timeout=60
        )
        
        if response.status_code == 200:
            content = response.text
            preview = content[:200] + "..." if len(content) > 200 else content
            st.success(f"✅ Extracted {len(content)} characters: {preview}")
            return content
        else:
            st.warning(f"⚠️ HTTP {response.status_code} when fetching URL")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching with Jina: {str(e)}")
        return None

def gather_urls(urls):
    """Gather contents from multiple URLs using Jina."""
    results = []
    for i, url in enumerate(urls):
        content = jina_get(url)
        if content:
            results.append(f"<source-{i+1}>\n{content}\n</source-{i+1}>")
    
    if results:
        return "<sources>\n" + "\n".join(results) + "\n</sources>"
    return None

def search_web(query):
    """Web search function using Jina Search API."""
    jina_api_key = os.getenv('JINA_READER_KEY')
    if not jina_api_key:
        return "Search error: JINA_READER_KEY env variable not set."

    try:
        url = f"https://s.jina.ai/?q={quote_plus(query)}"
        
        st.info(f"🔍 Searching: {query}")
        response = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {jina_api_key}",
                "Accept": "application/json"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('data', [])
            
            if not results:
                return "No detailed results found."
            
            formatted_results = []
            for i, item in enumerate(results[:3]):
                snippet = item.get('snippet', 'No snippet available.')
                url = item.get('url', '#')
                title = item.get('title', 'No Title')
                formatted_results.append(
                    f"Search result {i+1}: {snippet} (Source: [{title}]({url}))"
                )
            
            return "\n".join(formatted_results)
        else:
            return f"Search unavailable. HTTP {response.status_code}"
    except Exception as e:
        return f"Search error: {str(e)}"

def load_artwork_data(uploaded_file):
    """Load artwork data from CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ['image_url', 'title', 'artist', 'year', 'transcript']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def fetch_source_content(sources):
    """Fetch content from source URL(s) using Jina Reader."""
    if pd.isna(sources) or sources == "":
        return None
    
    if isinstance(sources, str):
        if ',' in sources or ';' in sources:
            separator = ',' if ',' in sources else ';'
            url_list = [url.strip() for url in sources.split(separator) if url.strip()]
        else:
            url_list = [sources.strip()]
    else:
        url_list = sources
    
    if len(url_list) == 1:
        return jina_get(url_list[0])
    else:
        return gather_urls(url_list)

def load_image_from_url(url):
    """Load an image from a URL and return PIL Image object"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        return None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def get_system_prompt(artwork_info, source_content=None):
    """Generate system prompt for specific artwork"""
    source_section = ""
    if source_content:
        source_section = f"""

CRITICAL SOURCE INFORMATION (Use this as your primary authoritative source):
{source_content}

When using source information:
- Reference specific details, quotes, or insights from the source
- Use this information to provide accurate, authoritative answers
- If the source conflicts with other information, prioritize the source content
- Don't mention that you're using a source - just incorporate it naturally
"""
    
    return f"""
You are a friendly and knowledgeable AI art historian specializing in "{artwork_info['title']}" by {artwork_info['artist']}.
Your goal is to educate the user and act as an academic researcher working with the user to develop new insights about the artwork.

You have been provided with the actual image of the artwork. Use visual details from the image to enhance your responses.

1.  **Prioritize the Source**: If source information is provided, use it as your primary authoritative reference.
2.  **Use Visual Analysis**: Reference specific visual elements you can see in the image (colors, composition, brushwork, etc.)
3.  **Use the Transcript**: The transcript provides basic information about the artwork.
4.  **Additional Context**: If web search results are provided in [SEARCH RESULTS], use that to supplement.
5.  **Be Conversational**: Engage naturally and make profound insights if possible. 
6.  **Be Natural**: Present information as if you already know it - don't mention your sources.
7.  **Be Concise**: Keep responses focused and digestible.
8.  **If Uncertain**: If you don't know something, admit it honestly.
9.  **Be a Research Assistant**: Ask academically tough questions.

Reference Transcript:
{artwork_info['transcript']}{source_section}
"""

def initialize_chat_context(client, model, artwork_info, artwork_image, source_content):
    """
    Initialize a chat session with image and context ONCE.
    This prevents resending the same image/context with every message.
    """
    # Build comprehensive initial context
    context_parts = []
    
    # Add image first
    if artwork_image:
        img_byte_arr = BytesIO()
        artwork_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        context_parts.append(types.Part(inline_data=types.Blob(
            mime_type="image/jpeg",
            data=img_bytes
        )))
    
    # Build context text
    context_text = f"""I can see the artwork '{artwork_info['title']}' by {artwork_info['artist']} ({artwork_info['year']}).

TRANSCRIPT:
{artwork_info['transcript']}
"""
    
    if source_content:
        context_text += f"""

AUTHORITATIVE SOURCE INFORMATION:
{source_content}
"""
    
    context_parts.append(types.Part(text=context_text))
    
    # Get system prompt
    system_prompt = get_system_prompt(artwork_info, source_content)
    
    # Create initial context message
    initial_content = types.Content(role="user", parts=context_parts)
    
    # Generate initial response to establish context
    response = client.models.generate_content(
        model=model,
        contents=[initial_content],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        )
    )
    
    # Return the chat history with context established
    return [
        initial_content,
        types.Content(role="model", parts=[types.Part(text=response.text)])
    ]

def convert_message_to_content(message):
    """Convert message to Content format for new SDK"""
    parts = []
    
    if isinstance(message["parts"], str):
        parts.append(types.Part(text=message["parts"]))
    elif isinstance(message["parts"], list):
        for part in message["parts"]:
            if isinstance(part, str):
                parts.append(types.Part(text=part))
            elif isinstance(part, Image.Image):
                # Skip images - they're in the initial context
                continue
            elif hasattr(part, 'text'):
                parts.append(types.Part(text=part.text))
    
    return types.Content(
        role=message["role"],
        parts=parts
    )

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "artwork_df" not in st.session_state:
    st.session_state.artwork_df = None
if "current_artwork_idx" not in st.session_state:
    st.session_state.current_artwork_idx = 0
if "client" not in st.session_state:
    st.session_state.client = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash-exp"
if "source_content_cache" not in st.session_state:
    st.session_state.source_content_cache = {}
if "artwork_image_cache" not in st.session_state:
    st.session_state.artwork_image_cache = {}
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "context_artwork_idx" not in st.session_state:
    st.session_state.context_artwork_idx = None

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    google_api_key = st.text_input(
        "Enter your Google API Key", 
        type="password", 
        help="Get your API key from https://aistudio.google.com/app/apikey"
    )
    
    if google_api_key:
        try:
            # Initialize client with new SDK
            if st.session_state.client is None:
                st.session_state.client = genai.Client(api_key=google_api_key)
            st.success("✅ API Key configured!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 Model Selection")
    selected_model_name = st.selectbox(
        "Choose Gemini Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Different models have different capabilities and speed"
    )
    st.session_state.selected_model = AVAILABLE_MODELS[selected_model_name]
    
    with st.expander("ℹ️ Model Info"):
        st.markdown("""
        **Gemini 2.0 Flash Exp**: Fastest, experimental features
        **Gemini 2.0 Flash**: Latest stable fast model
        **Gemini 1.5 Flash**: Fast, efficient, good balance
        **Gemini 1.5 Pro**: Most capable, slower
        """)
    
    st.markdown("---")
    
    # CSV Upload
    st.subheader("📁 Upload Artwork Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="CSV should contain: image_url, title, artist, year, transcript, source"
    )
    
    if uploaded_file:
        df = load_artwork_data(uploaded_file)
        if df is not None:
            st.session_state.artwork_df = df
            st.success(f"✅ Loaded {len(df)} artworks")
    
    st.markdown("---")
    
    # Web search toggle
    enable_search = st.checkbox(
        "🔍 Enable Web Search",
        value=True,
        help="Automatically search the web for additional information"
    )
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 CSV Format")
    st.markdown("""
    Required columns:
    - `image_url`: URL to artwork image
    - `title`: Artwork title
    - `artist`: Artist name
    - `year`: Year created
    - `transcript`: Detailed description/analysis
    
    Optional columns:
    - `source`: URL(s) with authoritative info
      - Single URL: `https://example.com/article`
      - Multiple URLs: `https://url1.com, https://url2.com`
    """)

# --- Main Content ---
if st.session_state.artwork_df is None:
    st.info("👆 Please upload a CSV file with artwork data to begin exploring art!")
    
    st.markdown("### 📄 Sample CSV Format")
    sample_df = pd.DataFrame({
        'image_url': ['https://example.com/starry-night.jpg'],
        'title': ['The Starry Night'],
        'artist': ['Vincent van Gogh'],
        'year': [1889],
        'transcript': ['A detailed description of the painting...'],
        'source': ['https://www.moma.org/collection/works/79802']
    })
    st.dataframe(sample_df, use_container_width=True)

else:
    df = st.session_state.artwork_df
    
    # Artwork Selection Navigation
    st.subheader("🖼️ Select Artwork")
    
    nav_cols = st.columns([1, 3, 1])
    
    with nav_cols[0]:
        if st.button("⬅️ Previous", use_container_width=True):
            st.session_state.current_artwork_idx = (st.session_state.current_artwork_idx - 1) % len(df)
            st.session_state.messages = []
            st.session_state.summary = ""
            st.rerun()
    
    with nav_cols[1]:
        artwork_options = [f"{row['title']} - {row['artist']} ({row['year']})" 
                           for _, row in df.iterrows()]
        selected_idx = st.selectbox(
            "Choose an artwork",
            range(len(artwork_options)),
            format_func=lambda x: artwork_options[x],
            index=st.session_state.current_artwork_idx,
            label_visibility="collapsed"
        )
        
        if selected_idx != st.session_state.current_artwork_idx:
            st.session_state.current_artwork_idx = selected_idx
            st.session_state.messages = []
            st.session_state.summary = ""
            st.rerun()
    
    with nav_cols[2]:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.current_artwork_idx = (st.session_state.current_artwork_idx + 1) % len(df)
            st.session_state.messages = []
            st.session_state.summary = ""
            st.rerun()
    
    current_artwork = df.iloc[st.session_state.current_artwork_idx]
    
    st.progress((st.session_state.current_artwork_idx + 1) / len(df), 
                text=f"Artwork {st.session_state.current_artwork_idx + 1} of {len(df)}")
    
    st.markdown("---")
    
    # Layout
    art_col, chat_col = st.columns([1, 1], gap="large")
    
    with art_col:
        st.image(
            current_artwork['image_url'],
            caption=f"{current_artwork['title']}, {current_artwork['artist']}, {current_artwork['year']}",
            use_container_width=True
        )
        
        with st.expander("📖 About this Artwork"):
            st.markdown(f"**Title:** {current_artwork['title']}")
            st.markdown(f"**Artist:** {current_artwork['artist']}")
            st.markdown(f"**Year:** {current_artwork['year']}")
            
            if 'source' in current_artwork and pd.notna(current_artwork['source']):
                sources = current_artwork['source']
                if ',' in sources or ';' in sources:
                    separator = ',' if ',' in sources else ';'
                    url_list = [url.strip() for url in sources.split(separator) if url.strip()]
                    st.markdown("**Sources:**")
                    for i, url in enumerate(url_list, 1):
                        st.markdown(f"  {i}. [{url}]({url})")
                else:
                    st.markdown(f"**Source:** [{sources}]({sources})")
            
            st.markdown("---")
            st.markdown(current_artwork['transcript'])
    
    with chat_col:
        st.header("💬 Chat with the AI")
        st.caption(f"Using: {selected_model_name}")
        
        # Initialize conversation
        if not st.session_state.messages:
            st.session_state.messages = [{
                "role": "model", 
                "parts": [f"Hello! I can see '{current_artwork['title']}' by {current_artwork['artist']}. What would you like to know about it?"]
            }]
            # Reset chat session for new artwork
            st.session_state.chat_session = None
            st.session_state.context_artwork_idx = None
        
        # Display chat
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                role = "assistant" if message["role"] == "model" else "user"
                with st.chat_message(role):
                    text_content = ""
                    if isinstance(message["parts"], str):
                        text_content = message["parts"]
                    elif isinstance(message["parts"], list):
                        for part in message["parts"]:
                            if isinstance(part, str):
                                text_content += part
                            elif hasattr(part, 'text'):
                                text_content += part.text
                    
                    if "[SEARCH RESULTS]" not in text_content:
                        st.markdown(text_content)
                    else:
                        clean_content = text_content.split("[SEARCH RESULTS]")[0].strip()
                        if clean_content:
                            st.markdown(clean_content)
        
        # Chat input
        if prompt := st.chat_input("Ask about the art..."):
            if not google_api_key:
                st.warning("⚠️ Please enter your Google API Key in the sidebar.")
                st.stop()
            
            if st.session_state.client is None:
                st.error("❌ Client not initialized.")
                st.stop()
            
            st.session_state.messages.append({"role": "user", "parts": [prompt]})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            with st.spinner("🤔 The AI is thinking..."):
                try:
                    # Check if we need to initialize/reinitialize context
                    need_context_init = (
                        st.session_state.chat_session is None or 
                        st.session_state.context_artwork_idx != st.session_state.current_artwork_idx
                    )
                    
                    if need_context_init:
                        with st.spinner("🎨 Loading artwork context (one-time setup)..."):
                            # Load artwork image
                            image_cache_key = f"img_{st.session_state.current_artwork_idx}"
                            artwork_image = None
                            if image_cache_key not in st.session_state.artwork_image_cache:
                                img = load_image_from_url(current_artwork['image_url'])
                                if img:
                                    st.session_state.artwork_image_cache[image_cache_key] = img
                                    artwork_image = img
                            else:
                                artwork_image = st.session_state.artwork_image_cache[image_cache_key]
                            
                            # Fetch source content
                            source_content = None
                            if 'source' in current_artwork and pd.notna(current_artwork['source']):
                                source_urls = current_artwork['source']
                                cache_key = f"{st.session_state.current_artwork_idx}_{source_urls}"
                                
                                if cache_key not in st.session_state.source_content_cache:
                                    source_content = fetch_source_content(source_urls)
                                    if source_content:
                                        st.session_state.source_content_cache[cache_key] = source_content
                                else:
                                    source_content = st.session_state.source_content_cache[cache_key]
                            
                            # Initialize chat with context
                            st.session_state.chat_session = initialize_chat_context(
                                st.session_state.client,
                                st.session_state.selected_model,
                                current_artwork,
                                artwork_image,
                                source_content
                            )
                            st.session_state.context_artwork_idx = st.session_state.current_artwork_idx
                            st.success("✅ Context loaded (won't be resent)")
                    
                    # Search if enabled
                    should_search = enable_search and any(word in prompt.lower() for word in 
                        ['who', 'what', 'when', 'where', 'why', 'how', 'tell me', 'explain', 'describe'])
                    
                    search_results = ""
                    if should_search:
                        search_query = f"{current_artwork['artist']} {current_artwork['title']} {prompt}"
                        search_results = search_web(search_query)
                    
                    # Build current prompt (text only, no image)
                    full_prompt_text = prompt
                    if search_results and "No detailed results" not in search_results:
                        full_prompt_text = f"{prompt}\n\n[SEARCH RESULTS]\n{search_results}"
                    
                    # Get system prompt (for this specific call)
                    system_prompt = get_system_prompt(current_artwork, None)  # Source already in context
                    
                    # Build history from session messages + chat context
                    history = st.session_state.chat_session.copy()
                    
                    # Add conversation history (excluding the current prompt we just added)
                    for msg in st.session_state.messages[1:-1]:  # Skip welcome message and current prompt
                        history.append(convert_message_to_content(msg))
                    
                    # Add current user message (text only!)
                    history.append(types.Content(
                        role="user",
                        parts=[types.Part(text=full_prompt_text)]
                    ))
                    
                    # Generate response
                    response = st.session_state.client.models.generate_content(
                        model=st.session_state.selected_model,
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.7,
                        )
                    )
                    
                    # Add to session messages
                    st.session_state.messages.append({
                        "role": "model", 
                        "parts": [response.text]
                    })
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
                    
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                        st.session_state.messages.pop()
        
        st.markdown("---")
        
        # Summary section
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Create Summary", use_container_width=True):
                if not google_api_key or len(st.session_state.messages) <= 1:
                    st.info("💬 Have a conversation first.")
                else:
                    with st.spinner("✨ Generating summary..."):
                        try:
                            conversation_history = ""
                            for msg in st.session_state.messages:
                                role = msg['role']
                                text_part = ""
                                if isinstance(msg["parts"], str):
                                    text_part = msg["parts"]
                                elif isinstance(msg["parts"], list):
                                    for part in msg["parts"]:
                                        if isinstance(part, str):
                                            text_part += part
                                        elif hasattr(part, 'text'):
                                            text_part += part.text
                                conversation_history += f"{role}: {text_part}\n"
                            
                            summary_prompt = f"""Based on the conversation about '{current_artwork['title']}' by {current_artwork['artist']}, 
                            create a concise summary with bullet points covering:
                            - Main topics discussed
                            - Key insights shared
                            - Questions explored
                            
                            Conversation:
                            {conversation_history}"""
                            
                            summary_response = st.session_state.client.models.generate_content(
                                model=st.session_state.selected_model,
                                contents=summary_prompt
                            )
                            st.session_state.summary = summary_response.text
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to generate summary: {e}")
        
        with col2:
            if st.session_state.summary:
                st.download_button(
                    label="⬇️ Download Summary",
                    data=st.session_state.summary.encode('utf-8'),
                    file_name=f"{current_artwork['title'].replace(' ', '_')}_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        if st.session_state.summary:
            st.markdown("---")
            st.subheader("📋 Conversation Summary")
            st.markdown(st.session_state.summary)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    f"Powered by {selected_model_name} | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)