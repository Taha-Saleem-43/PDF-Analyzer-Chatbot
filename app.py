import os
import gradio as gr
import fitz  # PyMuPDF
from fpdf import FPDF
import httpx
import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# --- Config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"
MAX_MEMORY_ITEMS = 4

conversation_memory = []
text_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Utils ---

def clear_conversation_memory():
    global conversation_memory
    conversation_memory = []

def extract_text_from_pdf(file_bytes):
    with fitz.open(stream=BytesIO(file_bytes), filetype="pdf") as doc:
        return [page.get_text() for page in doc]

def chunk_text(text, chunk_size=300):
    sentences = text.split(". ")
    chunks, current = [], ""
    for s in sentences:
        current += s.strip() + ". "
        if len(current) > chunk_size:
            chunks.append(current.strip())
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks

def process_uploaded_pdfs(uploaded_files):
    try:
        file_bytes_list = [file.read() if hasattr(file, "read") else file for file in uploaded_files]
        with ThreadPoolExecutor() as executor:
            texts = executor.map(extract_text_from_pdf, file_bytes_list)
        all_text = "\n\n".join(["\n".join(pages) for pages in texts])
        text_chunks = chunk_text(all_text)
        embeddings = text_embedder.encode(text_chunks, batch_size=16, show_progress_bar=False)
        clear_conversation_memory()
        return f"✅ Processed {len(uploaded_files)} PDF(s)", {
            "text_chunks": text_chunks,
            "embeddings": embeddings,
            "full_text": all_text
        }
    except Exception as e:
        return f"❌ PDF processing error: {str(e)}", None

async def query_llm(user_question, document_data):
    global conversation_memory
    if not user_question.strip():
        clear_conversation_memory()
        return "🗑️ Memory cleared", document_data
    if not document_data or not document_data["text_chunks"]:
        return "❌ Please upload PDFs first", document_data

    conversation_memory.append({"role": "user", "content": user_question})
    if len(conversation_memory) > MAX_MEMORY_ITEMS:
        conversation_memory = conversation_memory[-MAX_MEMORY_ITEMS:]

    question_embedding = text_embedder.encode([user_question])[0]
    sim_scores = cosine_similarity([question_embedding], document_data["embeddings"])[0]
    top_chunks = [document_data["text_chunks"][i] for i in sim_scores.argsort()[-3:][::-1]]
    context = "Document Context:\n" + "\n".join(top_chunks)
    messages = [
        {"role": "system", "content": "Answer using the documents and recent conversation."},
        {"role": "user", "content": context},
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
            return llm_response, document_data
        except Exception as e:
            return f"❌ API error: {str(e)}", document_data

async def generate_quiz(document_data, quiz_topic, difficulty_level, mcq_count, short_answer_count, long_answer_count):
    if not document_data or not document_data["full_text"]:
        return "❌ No PDF content available. Please upload PDFs first.", None

    prompt = f"""
    Generate a {difficulty_level} quiz about {quiz_topic} based on the document.
    Multiple-choice: {mcq_count}, Short Answer: {short_answer_count}, Long Answer: {long_answer_count}.
    For MCQ: 4 options, no answers. For short/long: no answers.

    Format:
    === Multiple Choice Questions ===
    1. [Question]?
    a) Option 1
    b) Option 2
    c) Option 3
    d) Option 4

    === Short Answer Questions ===
    1. [Question]?

    === Long Answer Questions ===
    1. [Question]?

    Document content:
    {document_data["full_text"][:5000]}
    """
    messages = [
        {"role": "system", "content": "You are a quiz generator that creates educational questions."},
        {"role": "user", "content": prompt}
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
            quiz_content = response.json()['choices'][0]['message']['content']
            return quiz_content, document_data
        except Exception as e:
            return f"❌ Quiz generation failed: {str(e)}", document_data

async def generate_quiz_answers(document_data, quiz_content):
    if isinstance(quiz_content, (tuple, list)):
        quiz_content = quiz_content[0] if quiz_content else ""
    if not quiz_content or quiz_content.strip() == "":
        return "❌ No quiz has been generated yet. Please generate a quiz before requesting answers.", None
    if not document_data or not document_data.get("full_text"):
        return "❌ No PDF content available. Please upload PDFs first.", None

    prompt = f"""
    Provide correct answers for the following quiz.

    Quiz:
    {quiz_content}

    Reference document content:
    {document_data['full_text'][:5000]}

    Format:
    === Answers ===
    1. [Answer to Q1]
    2. [Answer to Q2]
    (and so on)
    """
    messages = [
        {"role": "system", "content": "You are an educational assistant providing concise, accurate answers to quizzes based on the given document."},
        {"role": "user", "content": prompt}
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
            answer_content = response.json()['choices'][0]['message']['content']
            return answer_content, document_data
        except Exception as e:
            return f"❌ Quiz answer generation failed: {str(e)}", document_data

def create_pdf(text_content, filename, title=None):
    try:
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        if title:
            pdf.cell(200, 10, txt=title, ln=True, align='C')
            pdf.ln(10)
        for line in text_content.split('\n'):
            if line.startswith("==="):
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, txt=line, ln=True)
                pdf.set_font("Arial", size=12)
            else:
                pdf.multi_cell(0, 10, txt=line)
            pdf.ln(2)
        pdf.output(filename)
        return f"💾 Saved as '{filename}'", filename
    except Exception as e:
        return f"❌ PDF creation failed: {str(e)}", None

# --- UI ---

with gr.Blocks(title="PDF Q&A & Quiz Generator", css="""
    .visible-scrollbar textarea {
        max-height: 400px !important;
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
""") as app_interface:
    gr.Markdown(
        """
        <div style="display: flex; align-items: center; gap: 16px;">
            <img src="https://cdn-icons-png.flaticon.com/512/337/337946.png" width="48" style="vertical-align:middle" />
            <h1 style="margin: 0; font-size: 2.2rem;">PDF Q&A & Quiz Generator</h1>
        </div>
        """,
        elem_classes=["main-title"]
    )

    document_state = gr.State()
    quiz_state = gr.State()
    answers_state = gr.State()

    with gr.Tabs():
        # --- Tab 1: PDF Q&A ---
        with gr.TabItem("📄 PDF Q&A"):
            with gr.Column():
                gr.Markdown("### 📤 Upload PDFs & Ask")
                pdf_upload = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="binary"
                )
                upload_status = gr.Textbox(label="Status", interactive=False)

                user_question = gr.Textbox(
                    label="Ask about your PDFs",
                    placeholder="Type your question and press Enter..."
                )
                ask_button = gr.Button("Ask", variant="primary")
                clear_button = gr.Button("Clear Memory")
                llm_response = gr.Textbox(
                    label="Response",
                    lines=12,
                    interactive=False,
                    elem_classes=["visible-scrollbar"]
                )

        # --- Tab 2: Quiz ---
        with gr.TabItem("📝 Generate Quiz"):
            with gr.Column():
                gr.Markdown("### 📝 Quiz Settings")
                quiz_topic = gr.Textbox(
                    label="Quiz Topic",
                    placeholder="e.g., Machine Learning Basics"
                )
                difficulty_level = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="medium",
                    label="Quiz Difficulty"
                )
                with gr.Row():
                    mcq_count = gr.Number(
                        label="MCQ", value=3, minimum=0, maximum=20, step=1, precision=0
                    )
                    short_answer_count = gr.Number(
                        label="Short", value=2, minimum=0, maximum=20, step=1, precision=0
                    )
                    long_answer_count = gr.Number(
                        label="Long", value=1, minimum=0, maximum=20, step=1, precision=0
                    )
                generate_quiz_button = gr.Button("Generate Quiz", variant="primary")
                quiz_output = gr.Textbox(
                    label="Quiz", lines=10, interactive=False, elem_classes=["visible-scrollbar"]
                )
                save_quiz_button = gr.Button("Save Quiz as PDF")
                quiz_save_status = gr.Textbox(label="Quiz PDF Status")
                quiz_pdf_download = gr.File(label="Download Quiz PDF", visible=False)

                generate_answers_button = gr.Button("Generate Quiz Answers", variant="secondary")
                quiz_answers_output = gr.Textbox(
                    label="Answers", lines=8, interactive=False, elem_classes=["visible-scrollbar"]
                )
                save_answers_button = gr.Button("Save Answers as PDF")
                answers_save_status = gr.Textbox(label="Answers PDF Status")
                answers_pdf_download = gr.File(label="Download Answers PDF", visible=False)

    # --- Logic wiring ---

    pdf_upload.change(
        process_uploaded_pdfs,
        inputs=pdf_upload,
        outputs=[upload_status, document_state]
    )
    ask_button.click(
        lambda q, d: asyncio.run(query_llm(q, d)),
        inputs=[user_question, document_state],
        outputs=[llm_response, document_state]
    )
    user_question.submit(
        lambda q, d: asyncio.run(query_llm(q, d)),
        inputs=[user_question, document_state],
        outputs=[llm_response, document_state]
    )
    clear_button.click(
        lambda: [clear_conversation_memory(), "🗑️ Memory cleared"],
        outputs=None
    )

    generate_quiz_button.click(
        lambda d, topic, diff, mcq, short, long: asyncio.run(generate_quiz(d, topic, diff, mcq, short, long)),
        inputs=[document_state, quiz_topic, difficulty_level, mcq_count, short_answer_count, long_answer_count],
        outputs=[quiz_output, document_state],
        queue=True
    ).then(
        lambda quiz_content, d: (quiz_content,),
        inputs=[quiz_output, document_state],
        outputs=[quiz_state]
    )
    save_quiz_button.click(
        lambda content, topic, diff: create_pdf(content, f"quiz_{topic.replace(' ', '_')}_{diff}.pdf", f"Quiz on {topic} ({diff.capitalize()})"),
        inputs=[quiz_output, quiz_topic, difficulty_level],
        outputs=[quiz_save_status, quiz_pdf_download]
    )
    generate_answers_button.click(
        lambda d, quiz: asyncio.run(generate_quiz_answers(d, quiz)),
        inputs=[document_state, quiz_state],
        outputs=[quiz_answers_output, document_state],
        queue=True
    ).then(
        lambda answers_content, d: (answers_content,),
        inputs=[quiz_answers_output, document_state],
        outputs=[answers_state]
    )
    save_answers_button.click(
        lambda content, topic, diff: create_pdf(content, f"quiz_answers_{topic.replace(' ', '_')}_{diff}.pdf", f"Quiz Answers on {topic} ({diff.capitalize()})"),
        inputs=[quiz_answers_output, quiz_topic, difficulty_level],
        outputs=[answers_save_status, answers_pdf_download]
    )

if __name__ == "__main__":
    app_interface.launch()
