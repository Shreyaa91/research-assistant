import streamlit as st
import requests
import uuid
import time

# ------------------ Config ------------------
API_URL_CHAT = "http://localhost:8000/chat"
API_URL_UPLOAD = "http://localhost:8000/upload_file"

st.set_page_config(page_title="LangGraph GPT", layout="wide")
st.title("LangGraph GPT Chatbot")
st.markdown("Upload a file or paste a URL, then chat with it!")

# ------------------ Session ID ------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ------------------ Messages ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- File uploader -------------------
uploaded_file = st.file_uploader(
    "Drag and drop a file here",
    type=["txt", "pdf", "docx"],
    help="Limit 200MB per file"
)

if uploaded_file:
    with st.spinner("Uploading and processing file..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            res = requests.post(API_URL_UPLOAD, files=files, timeout=120)
            if res.status_code == 200:
                st.success(res.json().get("message", "File processed successfully!"))
            else:
                st.error(f"Upload failed: {res.text}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

# ------------------- Chat input -------------------
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    # Display user message immediately
    st.chat_message("user").markdown(user_input)

    # ------------------ Stream response ------------------
    try:
        with st.chat_message("assistant"):
            # Open streaming request
            with requests.post(
                API_URL_CHAT,
                json={"text": user_input, "session_id": st.session_state.session_id},
                stream=True,
                timeout=300
            ) as response:
                if response.status_code == 200:
                    buffer = ""
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            text_chunk = chunk.decode("utf-8")
                            buffer += text_chunk
                            # Display chunk as it comes
                            st.markdown(buffer)  # show typing cursor
                            time.sleep(0.05)
                    # Remove cursor after done
                    st.markdown(buffer)
                    # Store assistant message in session_state
                    st.session_state.messages.append(("ai", buffer))
                else:
                    error_msg = f"Error {response.status_code}: {response.text}"
                    st.session_state.messages.append(("ai", error_msg))
                    st.error(error_msg)
    except Exception as e:
        st.session_state.messages.append(("ai", f"❌ Error: {str(e)}"))
        st.error(f"❌ Error: {e}")

# ------------------- Display previous messages -------------------
for sender, msg in st.session_state.messages:
    if sender == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
