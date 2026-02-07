import streamlit as st
import os 
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import base64
import tempfile
import glob
from datetime import datetime

from utils import (
    extract_text_from_pdf,
    clean_text, 
    chunk_text, 
    create_embeddings,
    create_faiss_index,
    search_similar_chunks,
    generate_prompt
)

load_dotenv()

st.set_page_config(
    page_title='RAG PDF Q&A',
    page_icon = "",
    layout = 'wide'
)

st.title("RAG-based PDF Question Answering")
st.markdown("""
Upload a PDF document and ask questions about its content. 
This app uses Retrieval-Augmented Generation (RAG) to provide accurate answers.
""")

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'pdf_count' not in st.session_state:
    st.session_state.pdf_count = 0
if 'last_reset' not in st.session_state:
    st.session_state.last_reset = datetime.now()

MAX_QUERIES_PER_SESSION = 15     
MAX_PDFS_PER_SESSION = 3         
MAX_PDF_SIZE_MB = 5         
MAX_TOTAL_SIZE_MB = 10          
MAX_CHUNKS = 300                 

COOLDOWN_SECONDS = 2 

with st.sidebar:
    st.header("Settings")

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value = 100, 
        max_value = 300, 
        value = 300,
        step = 50
    )

    overlap = st.slider(
        "Overlap (characters)",
        min_value = 1,
        max_value = 5,
        value = 3
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3
    )

    temperature = st.slider(
        "Temperature",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.3, 
        step = 0.1
    )

    st.markdown("---")
    st.markdown("Quick Demo")
    # use_demo = st.checkbox("Load sample PDFs")
    st.caption("Select sample PDFs to try:")
    
    # if use_demo:
    #     st.success("✅ Sample PDFs selected!")
    #     st.caption("📚 ancient_egypt.pdf")
    #     st.caption("📚 coffee_history.pdf")
    #     st.caption("Click 'Process PDFs' below to load them")
    demo_egypt = st.checkbox("📚 Ancient Egypt", key="demo_egypt")
    demo_coffee = st.checkbox("📚 Coffee History", key="demo_coffee")
    
    use_demo = demo_egypt or demo_coffee

    st.markdown("### Restrictions")
    st.caption(f"• Max {MAX_PDF_SIZE_MB}MB per PDF")
    st.caption(f"• Max {MAX_TOTAL_SIZE_MB}MB total")
    st.caption(f"• Max {MAX_PDFS_PER_SESSION} PDFs per session")  # Shows 3 now
    st.caption(f"• Max {MAX_QUERIES_PER_SESSION} questions per session")  # Shows 15 now
    st.caption(f"• {COOLDOWN_SECONDS}s cooldown between questions")

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'index' not in st.session_state:
    st.session_state.index = None

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            st.error('GROQ API KEY not found.')
            st.stop()

    client = Groq(api_key = api_key)

    return embedding_model, client

try:
    embedding_model, groq_client = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Upload Document")

    st.info("Maximum file size: 5MB per PDF")
    uploaded_files = st.file_uploader(
        "Choose a PDF file", 
        type = 'pdf',
        accept_multiple_files=True
    )

    if use_demo and not uploaded_files:
        selected_samples = []
        
        # Check which samples were selected
        if demo_egypt:
            selected_samples.append("sample_data/ancient_egypt.pdf")
        if demo_coffee:
            selected_samples.append("sample_data/coffee_history.pdf")

        # if sample_pdf_paths:
        #     st.info(f"{len(sample_pdf_paths)} sample PDF(s) ready to process")
        if selected_samples:
            st.info(f"{len(selected_samples)} sample PDF(s) selected")

            class FakePDFFile:
                def __init__(self, filepath):
                    self.name = os.path.basename(filepath)
                    self.path = filepath
                    with open(filepath, 'rb') as f:
                        self.content = f.read()
                    self.size = len(self.content)

                def getbuffer(self):
                    return self.content
                
                def getvalue(self):
                    return self.content
                
            uploaded_files = [FakePDFFile(path) for path in selected_samples]
        else:
            st.error("No sample PDFs found in sample_data/ folder")

    if uploaded_files:

        total_size = 0
        oversized_files = []
        
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)  # Convert to MB
            total_size += file_size_mb
            
            if file_size_mb > MAX_PDF_SIZE_MB:
                oversized_files.append(f"{file.name} ({file_size_mb:.2f}MB)")
        
        # Show error if files too large
        if oversized_files:
            st.error(f"Files exceed {MAX_PDF_SIZE_MB}MB limit:\n" + "\n".join(oversized_files))
            st.stop()
        
        if total_size > MAX_TOTAL_SIZE_MB:
            st.error(f"Total size ({total_size:.2f}MB) exceeds {MAX_TOTAL_SIZE_MB}MB limit")
            st.stop()
        
        # Check PDF count limit
        if st.session_state.pdf_count >= MAX_PDFS_PER_SESSION:
            st.error(f"Session limit: You can only process {MAX_PDFS_PER_SESSION} PDFs per session.")
            st.stop()
        
        st.write(f"{len(uploaded_files)} file(s) uploaded (Total: {total_size:.2f}MB)") 
        # if not st.session_state.processed:
        with st.expander("📄 Preview Uploaded Files"):
            for file in uploaded_files:
                st.write(f"**{file.name}** ({file.size / 1024:.2f} KB)")

                try:
                    pdf_bytes = file.getvalue()
                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

                    pdf_display = f'''
                        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                            width="500" 
                            height="500" 
                            type="application/pdf">
                    </iframe>
                    '''
                    st.markdown(pdf_display, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not preview PDF: {str(e)}")

        if st.button("Process PDFs", type = 'primary'):
            with st.spinner("Processing PDF"):
                try:

                    all_text = ""
                    # with tempfile.NamedTemporaryFile(delete = False, suffix = '.pdf') as tmp_file:
                    #     tmp_file.write(uploaded_file.getbuffer())
                    #     tmp_file_path = tmp_file.name

                    with st.status("Extracting text from PDF", expanded = True) as status:
                        for i, uploaded_file in enumerate(uploaded_files):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                tmp_file_path = tmp_file.name

                            raw_text = extract_text_from_pdf(tmp_file_path)
                            all_text += raw_text + '\n\n'

                            os.unlink(tmp_file_path)

                            st.write(f"Processed: {uploaded_file.name}")
                        st.write(f"Total characters extracted: {len(all_text)}")

                        cleaned_text = clean_text(all_text)
                        st.write("Cleaned Text")

                        chunks = chunk_text(cleaned_text, chunk_size = chunk_size, overlap = overlap)
                        st.write(f"Created {len(chunks)} chunks")

                        embeddings = create_embeddings(chunks, embedding_model)
                        st.write(f"Generated embeddings")

                        index = create_faiss_index(embeddings)
                        st.write(f"Built search index")

                        status.update(label = 'Processing complete', state = 'complete')

                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.processed = True
                    st.session_state.pdf_count += len(uploaded_files)
                    st.session_state.file_names = [f.name for f in uploaded_files]

                    st.success("PDF processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing PDF: str{e}")
                    st.session_state.processed= False

with col2:
    st.subheader("Ask Questions: ")

    if not st.session_state.processed:
        st.info('Please upload and process a PDF first')
    else:
        query = st.text_input(
            "Enter your question:",
            placeholder= 'e.g. What is the document about? '
        )

        submit_button = st.button("Get Answer", type="primary", disabled= not query)

        if submit_button and query:

            if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
                st.error(f"Session limit reached: {MAX_QUERIES_PER_SESSION} questions per session. Please refresh to continue.")
                st.stop()

            if 'last_query_time' in st.session_state:
                time_since_last = (datetime.now() - st.session_state.last_query_time).total_seconds()
                if time_since_last < COOLDOWN_SECONDS:
                    wait_time = COOLDOWN_SECONDS - time_since_last
                    st.warning(f"Please wait {wait_time:.1f} seconds before asking another question.")
                    st.stop()

            st.session_state.last_query_time = datetime.now()
            st.session_state.query_count += 1

            with st.spinner("Searching and generating Answer"):
                try:
                    retrieved_chunks, indices, distances = search_similar_chunks(
                        query, 
                        st.session_state.chunks, 
                        embedding_model,
                        st.session_state.index,
                        k = top_k
                    )

                    context = " ".join(retrieved_chunks)
                    prompt = generate_prompt(query, context)
                    try:
                        response = groq_client.chat.completions.create(
                            model = 'llama-3.1-8b-instant',
                            messages= [
                                {
                                    'role': 'system',
                                    "content": "You are a helpful assistant that answers questions based on the provided context."
                                },
                                {
                                    'role': "user",
                                    'content': prompt
                                }
                            ],
                            temperature= temperature,
                            max_tokens= 500
                        )

                        answer = response.choices[0].message.content
                    except Exception as e:
                        error_msg = str(e).lower()
                        
                        # Check if it's a rate limit error
                        if "rate limit" in error_msg or "429" in error_msg:
                            st.error("Rate limit reached. Please wait a moment and try again.")
                        elif "timeout" in error_msg:
                            st.error("Request timed out. Please try again.")
                        else:
                            st.error(f"Error: {str(e)}")
                        
                        st.stop()

                    st.markdown("# Answer: ")
                    st.markdown(answer)

                    with st.expander("View Source chunks"):
                        for i, (chunk, idx, dist) in enumerate(zip(retrieved_chunks, indices, distances)):
                            st.markdown(f"Chunk {i+1} (Index: {idx}, Distance: {dist:.4f})")
                            st.text_area(
                                f"chunk_{i}",
                                chunk, 
                                height = 100,
                                label_visibility="collapsed"
                            )
                            st.markdown("----")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")