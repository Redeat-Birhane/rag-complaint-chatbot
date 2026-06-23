import faiss
import pandas as pd

INDEX_PATH = "../data/complaints.faiss"
META_PATH  = "../data/metadata.pkl"


def load_vectorstore():
    index = faiss.read_index(INDEX_PATH)
    metadata_df = pd.read_pickle(META_PATH)
    return index, metadata_df