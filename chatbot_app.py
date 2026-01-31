import streamlit as st
import anthropic
import openai
from google import generativeai as genai
import os
import base64
from PIL import Image
import io
import PyPDF2
import replicate

# Page config
st.set_page_config(page_title="Multi-LLM Chatbot", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model selection
    provider = st.selectbox(
        "Select Provider",
        ["Anthropic (Claude)", "OpenAI (GPT-4o)", "Google (Gemini)", "Meta (Llama 3.2)"]
    )
    
    # API Key input
    api_key = st.text_input(
        f"Enter {provider.split()[0]} API Key",
        type="password",
        help="Your API key will not be stored"
    )
    
    # System prompt
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Set system behavior",
        value="You are a helpful AI assistant.",
        height=100
    )
    
    # Temperature
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # File upload
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload images, PDFs, or code files",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg', 'pdf', 'txt', 'py', 'js', 'java', 'cpp', 'md']
    )
    
    # Clear chat
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat interface
st.title("🤖 Multi-LLM Chatbot")
st.caption(f"Currently using: {provider}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "files" in message and message["files"]:
            st.caption(f"📎 Attached: {', '.join(message['files'])}")

# Chat input
if prompt := st.chat_input("Type your message..."):
    if not api_key:
        st.error("Please enter your API key in the sidebar")
        st.stop()
    
    # Process uploaded files
    file_contents = []
    file_names = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_names.append(uploaded_file.name)
            file_type = uploaded_file.type
            
            if file_type.startswith('image'):
                # Handle images
                image = Image.open(uploaded_file)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                file_contents.append({
                    "type": "image",
                    "data": img_base64,
                    "name": uploaded_file.name
                })
            elif file_type == 'application/pdf':
                # Handle PDFs
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                file_contents.append({
                    "type": "text",
                    "data": text,
                    "name": uploaded_file.name
                })
            else:
                # Handle text/code files
                text = uploaded_file.read().decode('utf-8')
                file_contents.append({
                    "type": "text",
                    "data": text,
                    "name": uploaded_file.name
                })
    
    # Add user message to history
    user_message = {"role": "user", "content": prompt}
    if file_names:
        user_message["files"] = file_names
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        if file_names:
            st.caption(f"📎 Attached: {', '.join(file_names)}")
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if "Anthropic" in provider:
                # Claude API
                client = anthropic.Anthropic(api_key=api_key)
                
                # Build messages with file context
                messages = []
                for msg in st.session_state.messages[:-1]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add current message with files
                current_content = []
                for file in file_contents:
                    if file["type"] == "image":
                        current_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": file["data"]
                            }
                        })
                    else:
                        current_content.append({
                            "type": "text",
                            "text": f"File: {file['name']}\n\n{file['data']}"
                        })
                current_content.append({"type": "text", "text": prompt})
                messages.append({"role": "user", "content": current_content})
                
                with client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_prompt,
                    messages=messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")
                
            elif "OpenAI" in provider:
                # OpenAI API
                client = openai.OpenAI(api_key=api_key)
                
                messages = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages[:-1]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add current message with files
                current_content = []
                for file in file_contents:
                    if file["type"] == "image":
                        current_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{file['data']}"
                            }
                        })
                    else:
                        current_content.append({
                            "type": "text",
                            "text": f"File: {file['name']}\n\n{file['data']}"
                        })
                current_content.append({"type": "text", "text": prompt})
                messages.append({"role": "user", "content": current_content})
                
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=temperature,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
            elif "Google" in provider:
                # Gemini API
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro-vision' if file_contents and any(f["type"] == "image" for f in file_contents) else 'gemini-pro')
                
                # Build context from history
                context = f"System: {system_prompt}\n\n"
                for msg in st.session_state.messages[:-1]:
                    context += f"{msg['role'].title()}: {msg['content']}\n\n"
                
                # Add files
                prompt_parts = [context]
                for file in file_contents:
                    if file["type"] == "image":
                        img_data = base64.b64decode(file["data"])
                        img = Image.open(io.BytesIO(img_data))
                        prompt_parts.append(img)
                    else:
                        prompt_parts.append(f"File: {file['name']}\n\n{file['data']}\n\n")
                prompt_parts.append(f"User: {prompt}\n\nAssistant:")
                
                response = model.generate_content(prompt_parts, stream=True)
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                
            elif "Meta" in provider:
                # Llama via Replicate
                os.environ["REPLICATE_API_TOKEN"] = api_key
                
                # Build context
                context = f"System: {system_prompt}\n\n"
                for msg in st.session_state.messages[:-1]:
                    context += f"{msg['role'].title()}: {msg['content']}\n\n"
                
                # Add file contents
                for file in file_contents:
                    if file["type"] == "text":
                        context += f"File: {file['name']}\n\n{file['data']}\n\n"
                
                context += f"User: {prompt}\n\nAssistant:"
                
                for event in replicate.stream(
                    "meta/llama-2-70b-chat",
                    input={
                        "prompt": context,
                        "temperature": temperature,
                        "max_new_tokens": 4096
                    }
                ):
                    full_response += str(event)
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            full_response = f"Error occurred: {str(e)}"
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
