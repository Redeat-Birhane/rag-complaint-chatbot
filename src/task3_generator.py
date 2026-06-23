"""
Task 3 — Section 3: Prompt Template and Generator
Combines retrieved chunks into a prompt and sends it to an LLM to generate
a synthesized, evidence-backed answer to the user's question.
Uses a lightweight model (flan-t5-base) suitable for 8GB RAM machines.
"""

import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# flan-t5-base: small (250MB), instruction-tuned, runs on CPU, no GPU needed.
# Swap to "google/flan-t5-large" if you have more RAM and want better answers.
MODEL_NAME  = "google/flan-t5-base"
MAX_NEW_TOKENS = 300


PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints.
Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context does not contain enough information to answer, say:
"I don't have enough information in the available complaints to answer this question."

Context:
{context}

Question: {question}

Answer:"""


def build_context(retrieved_chunks: list[dict], max_chunks: int = 5) -> str:
    """
    Format the retrieved chunks into a single context string for the prompt.

    Args:
        retrieved_chunks: Output of task3_retriever.retrieve().
        max_chunks:       Max number of chunks to include (avoids prompt overflow).

    Returns:
        A formatted string with numbered complaint excerpts.
    """
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
    Fill the prompt template with the question and retrieved context.

    Args:
        question:         User's plain-English question.
        retrieved_chunks: Output of task3_retriever.retrieve().

    Returns:
        The fully-formatted prompt string ready to send to the LLM.
    """
    if not isinstance(question, str) or question.strip() == "":
        raise ValueError("Question must be a non-empty string.")

    try:
        context = build_context(retrieved_chunks)
        prompt  = PROMPT_TEMPLATE.format(context=context, question=question)
    except Exception as e:
        raise RuntimeError(f"Failed to build prompt: {e}") from e

    return prompt


def load_generator(model_name: str = MODEL_NAME):
    """
    Load the text generation pipeline.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A HuggingFace text2text-generation pipeline.
    """
    try:
        logger.info(f"Loading generator model '{model_name}' ...")
        gen = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        logger.info("Generator model loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load generator model '{model_name}': {e}") from e

    return gen


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    generator,
) -> dict:
    """
    Generate a synthesized answer from the retrieved chunks.

    Args:
        question:         User's plain-English question.
        retrieved_chunks: Output of task3_retriever.retrieve().
        generator:        Loaded HuggingFace pipeline.

    Returns:
        dict with keys:
            - question:   original question
            - answer:     LLM-generated answer string
            - context:    formatted context string sent to the LLM
            - sources:    list of source dicts (complaint_id, product_category,
                          issue, company, document excerpt)
    """
    try:
        prompt = build_prompt(question, retrieved_chunks)
        output = generator(prompt)
        answer = output[0]["generated_text"].strip()
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
        index, metadata_df = load_vectorstore()
        embed_model        = load_embedding_model()
        generator          = load_generator()

        question = "Why are customers unhappy with their credit cards?"
        chunks   = retrieve(question, index, metadata_df, embed_model)
        result   = generate_answer(question, chunks, generator)

        print(f"\nQuestion : {result['question']}")
        print(f"\nAnswer   : {result['answer']}")
        print(f"\nSources  :")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['product_category']} | {s['issue']} | "
                  f"{s['company']} (score={s['score']})")
            print(f"      {s['excerpt'][:150]}")
    except Exception as err:
        logger.error(f"Generator test failed: {err}")