import os
import gradio as gr
import fitz  # PyMuPDF
from fpdf import FPDF
import httpx
import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"
MAX_MEMORY_ITEMS = 4  # Keeps last 2 Q&A pairs

# Global conversation memory
conversation_memory = []
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def clear_memory():
    global conversation_memory
    conversation_memory = []

def chunk_and_embed(text):
    sentences = text.split(". ")
    chunks, temp = [], ""
    for sent in sentences:
        temp += sent.strip() + ". "
        if len(temp) > 300:  # Chunk size
            chunks.append(temp.strip())
            temp = ""
            if len(chunks) >= 50:  # Max chunks
                break
    if temp:
        chunks.append(temp.strip())
    return chunks, embedder.encode(chunks)

def extract_text(file_objs):
    try:
        combined_text = ""
        for file_obj in file_objs:
            file_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
            with fitz.open(stream=BytesIO(file_bytes), filetype="pdf") as doc:
                for page in doc:
                    combined_text += page.get_text() + "\n\n"

        chunks, embeddings = chunk_and_embed(combined_text)
        clear_memory()
        return f"✅ Processed {len(file_objs)} PDF(s)", {
            "chunks": chunks,
            "embeddings": embeddings,
            "pdf_text": combined_text
        }
    except Exception as e:
        return f"❌ PDF error: {str(e)}", None

async def ask_llm_async(question, doc_data):
    global conversation_memory

    if not question.strip():
        clear_memory()
        return "🗑️ Memory cleared", doc_data

    if not doc_data or not doc_data["chunks"]:
        return "❌ Upload PDFs first", doc_data

    conversation_memory.append({"role": "user", "content": question})
    if len(conversation_memory) > MAX_MEMORY_ITEMS:
        conversation_memory = conversation_memory[-MAX_MEMORY_ITEMS:]

    question_vec = embedder.encode([question])[0]
    sims = cosine_similarity([question_vec], doc_data["embeddings"])[0]
    top_chunks = [doc_data["chunks"][i] for i in sims.argsort()[-3:][::-1]]

    context_str = "Document Context:\n" + "\n".join(top_chunks)
    messages = [
        {"role": "system", "content": "Answer using the documents and recent conversation."},
        {"role": "user", "content": context_str},
        *conversation_memory
    ]

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            llm_response = response.json()['choices'][0]['message']['content']

            conversation_memory.append({"role": "assistant", "content": llm_response})
            return llm_response, doc_data

        except Exception as e:
            return f"❌ API error: {str(e)}", doc_data

def create_pdf(response, filename):
    try:
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        clean_text = response.encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 10, clean_text)
        pdf.output(filename)
        return f"💾 Saved as '{filename}'"
    except Exception as e:
        return f"❌ PDF creation failed: {str(e)}"

with gr.Blocks(title="Multi-PDF Q&A", css="""
    .scroll-container {
        height: 500px;
        overflow-y: scroll;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 4px;
    }
    .scroll-container::-webkit-scrollbar {
        width: 10px;
    }
    .scroll-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
    .scroll-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
""") as demo:
    gr.Markdown("## 📚 Multi-PDF Q&A with Memory")

    doc_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                type="binary"
            )
            upload_status = gr.Textbox(label="Status", interactive=False)
            question = gr.Textbox(label="Ask question", placeholder="Type here...")
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("New Chat")
            with gr.Row():
                pdf_name = gr.Textbox(label="PDF Filename", placeholder="response.pdf", value="response.pdf")
                download_btn = gr.Button("Save Last Response")

    with gr.Column(scale=2):
        with gr.Column(variant="panel"):  # Changed from Box to Column with panel style
            answer_output = gr.Textbox(
                label="LLM Response",
                lines=10,
                interactive=False,
                elem_classes=["visible-scrollbar"]
            )
        download_status = gr.Textbox(label="Download Status")

        # CSS to force visible scrollbar
        demo.css = """
        .visible-scrollbar textarea {
            max-height: 500px !important;
            overflow-y: scroll !important;
            scrollbar-width: auto !important;
        }
        .visible-scrollbar textarea::-webkit-scrollbar {
            width: 12px !important;
        }
        .visible-scrollbar textarea::-webkit-scrollbar-thumb {
            background: #666 !important;
            border-radius: 10px !important;
            border: 3px solid #f0f0f0 !important;
        }
        """

    file_input.change(
        extract_text,
        inputs=file_input,
        outputs=[upload_status, doc_state]
    )
    ask_btn.click(
        lambda q, d: asyncio.run(ask_llm_async(q, d)),
        inputs=[question, doc_state],
        outputs=[answer_output, doc_state]
    )
    question.submit(
        lambda q, d: asyncio.run(ask_llm_async(q, d)),
        inputs=[question, doc_state],
        outputs=[answer_output, doc_state]
    )
    clear_btn.click(
        lambda: [clear_memory(), "🗑️ Memory cleared"],
        outputs=download_status
    )
    download_btn.click(
        create_pdf,
        inputs=[answer_output, pdf_name],
        outputs=download_status
    )

if __name__ == "__main__":
    demo.launch()
