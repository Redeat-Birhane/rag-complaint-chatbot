import gradio as gr
import logging

from task3_load_vectorstore import load_vectorstore
from task3_retriever import load_embedding_model, retrieve
from task3_generator import load_generator, generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# LOAD MODELS (ONCE)
# =========================
try:
    index, metadata_df = load_vectorstore()
    embed_model = load_embedding_model()
    generator = load_generator()
    logger.info("All models loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise


# =========================
# CORE RAG FUNCTION
# =========================
def answer_question(question):
    try:
        if not question or not question.strip():
            return "Please enter a valid question.", ""

        # Retrieval
        chunks = retrieve(question, index, metadata_df, embed_model)

        # Generation
        result = generate_answer(question, chunks, generator)
        answer = result.get("answer", "")

        # Format sources (top 2)
        sources = ""
        for i, c in enumerate(chunks[:2], 1):
            try:
                sources += (
                    f"\n[{i}] Company: {c.get('company', 'Unknown')}\n"
                    f"Issue: {c.get('issue', 'Unknown')}\n"
                    f"Text: {c.get('document', '')[:300]}\n"
                )
            except Exception:
                continue

        return answer, sources

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return f"Error: {str(e)}", ""


# =========================
# CLEAR FUNCTION
# =========================
def clear_fields():
    try:
        return "", "", ""
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        return "", "", ""


# =========================
# GRADIO UI
# =========================
try:
    with gr.Blocks(title="CrediTrust RAG Assistant") as demo:

        gr.Markdown("# 🏦 CrediTrust Financial Complaint Assistant")

        question = gr.Textbox(
            label="Ask a question",
            placeholder="e.g. Why are customers unhappy with credit cards?"
        )

        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

        answer_box = gr.Textbox(label="Answer", lines=6)
        sources_box = gr.Textbox(label="Sources (Top 2 retrieved chunks)", lines=10)

        ask_btn.click(
            fn=answer_question,
            inputs=question,
            outputs=[answer_box, sources_box]
        )

        clear_btn.click(
            fn=clear_fields,
            inputs=[],
            outputs=[question, answer_box, sources_box]
        )

except Exception as e:
    logger.error(f"UI build failed: {e}")
    raise


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    try:
        demo.launch()
    except Exception as e:
        logger.error(f"Failed to launch app: {e}")