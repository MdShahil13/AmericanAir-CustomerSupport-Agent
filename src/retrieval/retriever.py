import os

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AmericanAirRetriever:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embedding_path = "data/embeddings/american_air_embeddings.npy"

        self.documents = (
            self.df["customer_text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if os.path.exists(self.embedding_path):
            print("Loading saved embeddings...")
            self.embeddings = np.load(self.embedding_path)

        else:
            print("Creating embeddings for the first time...")

            self.embeddings = self.model.encode(
                self.documents,
                show_progress_bar=True,
                normalize_embeddings=True
            )

            os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)

            np.save(self.embedding_path, self.embeddings)

            print("Embeddings saved.")

    def retrieve(
        self,
        query: str,
        intent: str | None = None,
        top_k: int = 10
    ):
        df = self.df

        if intent and intent != "NON_SUPPORT":
            filtered_indices = df.index[
                df["predicted_intent"] == intent
            ].tolist()
        else:
            filtered_indices = df.index.tolist()

        if not filtered_indices:
            filtered_indices = df.index.tolist()

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        candidate_embeddings = self.embeddings[filtered_indices]

        scores = cosine_similarity(
            query_embedding,
            candidate_embeddings
        )[0]

        top_positions = scores.argsort()[-top_k:][::-1]

        top_indices = [
            filtered_indices[position]
            for position in top_positions
        ]

        results = self.df.loc[top_indices].copy()

        results["similarity"] = scores[top_positions]

        return results