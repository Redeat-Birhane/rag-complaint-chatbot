"""
Task 3 — Section 3: Prompt Template and Generator
Combines retrieved chunks into a prompt and sends it to a local Hugging Face model
to generate a synthesized, evidence-backed answer to the user's question.
"""

import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/opt-125m"
MAX_NEW_TOKENS = 64


SYSTEM_PROMPT = (
    "You are a financial analyst assistant for CrediTrust Financial. "
    "Your task is to answer questions about customer complaints. "
    "Use ONLY the retrieved complaint excerpts provided. "
    "If the context does not contain enough information, say: "
    "'I don't have enough information in the available complaints to answer this question.'"
)


def build_context(retrieved_chunks: list[dict], max_chunks: int = 5) -> str:
    if not retrieved_chunks:
        return "No relevant complaint excerpts were found."
    try:
        parts = []
        for i, chunk in enumerate(retrieved_chunks[:max_chunks], 1):
            product  = chunk.get("product_category", "Unknown")
            issue    = chunk.get("issue", "Unknown")
            company  = chunk.get("company", "Unknown")
            document = chunk.get("document", "").strip()
            parts.append(
                f"[Excerpt {i}] Product: {product} | Issue: {issue} | "
                f"Company: {company}\n{document}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        raise RuntimeError(f"Failed to build context string: {e}") from e


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    OPT is a plain causal LM, not a chat model, so we use a simple
    instruction-style prompt instead of ChatML tags.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    try:
        context = build_context(retrieved_chunks)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        return prompt
    except Exception as e:
        raise RuntimeError(f"Failed to build prompt: {e}") from e


def load_generator(model_name: str = MODEL_NAME):
    try:
        logger.info(f"Loading generator model '{model_name}' ...")
        gen = pipeline("text-generation", model=model_name)
        logger.info("Generator loaded.")
        return gen
    except Exception as e:
        raise RuntimeError(f"Failed to load generator: {e}") from e


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    generator,
) -> dict:
    try:
        prompt = build_prompt(question, retrieved_chunks)
        logger.info("Sending prompt to generator ...")
        response = generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,   # return only generated part, not the prompt
        )
        answer = response[0]["generated_text"].strip()

    except Exception as e:
        raise RuntimeError(f"Answer generation failed: {e}") from e

    sources = [
        {
            "complaint_id":     c.get("complaint_id", ""),
            "product_category": c.get("product_category", ""),
            "issue":            c.get("issue", ""),
            "company":          c.get("company", ""),
            "score":            round(c.get("score", 0.0), 4),
            "excerpt":          c.get("document", "")[:300],
        }
        for c in retrieved_chunks
    ]

    return {
        "question": question,
        "answer":   answer,
        "context":  build_context(retrieved_chunks),
        "sources":  sources,
    }


if __name__ == "__main__":
    from task3_load_vectorstore import load_vectorstore
    from task3_retriever import load_embedding_model, retrieve

    try:
        index, metadata_df  = load_vectorstore()
        embed_model         = load_embedding_model()
        generator           = load_generator()

        question = "Why are customers unhappy with their credit cards?"
        chunks = retrieve(question, index, metadata_df, embed_model)
        result = generate_answer(question, chunks, generator)

        print(f"\nQuestion : {result['question']}")
        print(f"\nAnswer   :\n{result['answer']}")
        print(f"\nSources  :")
        for i, s in enumerate(result["sources"], 1):
            print(
                f"  [{i}] {s['product_category']} | {s['issue']} | "
                f"{s['company']}  (score={s['score']})"
            )
            print(f"      {s['excerpt'][:150]}")

    except Exception as err:
        logger.error(f"Generator test failed: {err}")