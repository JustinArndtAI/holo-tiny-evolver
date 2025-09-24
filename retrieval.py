import pickle
from transformers import pipeline
import numpy as np
from hkm_wrapper import HKMWrapper
from utils import check_gpu_temp

class DynamicRetriever:
    def __init__(self):
        self.hkm = HKMWrapper()
        self.pipe = None
        self.load_model()

    def load_model(self):
        # Load the latest trained model from Phase 3
        self.pipe = pipeline("text-generation", model="outputs/phase3_model_final")
        check_gpu_temp()

    def load_manifold(self):
        # Load the updated manifold from Phase 4
        with open('outputs/phase4_updated_manifold.pkl', 'rb') as f:
            return pickle.load(f)

    def retrieve_context(self, query, manifold):
        # Load the original graph from Phase 1 for text data
        with open('outputs/phase1_enhanced_graph.pkl', 'rb') as f:
            graph = pickle.load(f)
        node_texts = [data.get('text', '') for _, data in graph.nodes(data=True)]
        similarities = np.array([self._cosine_sim(query, text) for text in node_texts])
        top_idx = np.argmax(similarities)
        return node_texts[top_idx] if similarities[top_idx] > 0.5 else "No relevant context found"

    def _cosine_sim(self, a, b):
        # Simple cosine similarity for text comparison
        a_vec = np.array([ord(c) for c in a.lower() if c.isalnum()])
        b_vec = np.array([ord(c) for c in b.lower() if c.isalnum()])
        if len(a_vec) == 0 or len(b_vec) == 0:
            return 0.0
        a_norm = np.linalg.norm(a_vec)
        b_norm = np.linalg.norm(b_vec)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a_vec, b_vec) / (a_norm * b_norm)

    def generate_response(self, query):
        manifold = self.load_manifold()
        context = self.retrieve_context(query, manifold)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        response = self.pipe(prompt, max_new_tokens=50)[0]['generated_text']
        return response

if __name__ == "__main__":
    retriever = DynamicRetriever()
    test_query = "What is AI?"
    print(f"Query: {test_query}")
    print(f"Response: {retriever.generate_response(test_query)}")