import logging
import pandas as pd

from task3_load_vectorstore import load_vectorstore
from task3_retriever import load_embedding_model, retrieve
from task3_generator import load_generator, generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTIONS = [
    "Why are customers unhappy with their credit cards?",
    "What issues do users report about bank fees?",
    "Why do customers complain about fraud cases?",
    "What problems do users face with loan repayment?",
    "What complaints are common in customer service?",
]

def initialize():
    try:
        index, metadata = load_vectorstore()
        embed_model = load_embedding_model()
        generator = load_generator()
        return index, metadata, embed_model, generator
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise


def run_evaluation():
    index, metadata, embed_model, generator = initialize()

    results_table = []

    for q in QUESTIONS:
        try:
            logger.info(f"Processing question: {q}")

            # 1. Retrieval
            chunks = retrieve(q, index, metadata, embed_model)

            # 2. Generation
            output = generate_answer(q, chunks, generator)

            # 3. Extract top sources (1–2)
            top_sources = chunks[:2]

            results_table.append({
                "question": q,
                "answer": output["answer"],
                "sources": [
                    {
                        "company": s.get("company"),
                        "issue": s.get("issue"),
                        "score": round(s.get("score", 0), 4)
                    }
                    for s in top_sources
                ],
                "status": "success"
            })

        except Exception as e:
            logger.error(f"Failed on question '{q}': {e}")

            # still record failure so report is complete
            results_table.append({
                "question": q,
                "answer": None,
                "sources": [],
                "status": f"failed: {str(e)}"
            })

    return results_table


def score_answer(answer: str) -> int:
    try:
        if answer is None:
            return 1

        text = answer.lower()

        # very bad outputs
        if "default default" in text:
            return 1
        if len(text.strip()) < 30:
            return 2

        # repetition detection
        if len(set(text.split())) < 5:
            return 2

        # weak but usable
        if "not enough information" in text:
            return 3

        # acceptable but generic
        return 3

    except:
        return 1
    

def save_results(results):
    try:
        df = pd.DataFrame(results)

        df["score"] = df["answer"].apply(score_answer)

        df.to_csv("evaluation_results.csv", index=False)

        print("\nEvaluation saved to evaluation_results.csv")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    try:
        results = run_evaluation()
        save_results(results)
    except Exception as e:
        logger.error(f"Evaluation pipeline crashed: {e}")