import re
import numpy as np
from numpy.linalg import norm

def tokenize(text):
    """
    Tokeniza de forma básica: minúsculas + extraer palabras con regex.
    """
    tokens = re.findall(r"\w+", text.lower())
    return tokens

def simulate_text_generation_visual(input_text, glove_dict, embedding_dim=300):
    """
    Simula, en orden:
      1) Tokenización
      2) Cálculo de Embeddings
      3) (Fake) Self-Attention
      4) (Fake) Feed-Forward
      5) Cálculo de Logits
      6) Softmax
      7) Selección

    Retorna un dict con TODOS los datos intermedios:
      {
        "tokens": [...],
        "embeddings_per_token": [...],
        "context_vector": [...],
        "attention_vector": [...],
        "feed_forward_vector": [...],
        "candidate_words": [...],
        "logits": [...],
        "probs": [...],
        "selected_token": str
      }
    """

    # Pequeño vocab para la selección final
    candidate_words = ["water", "sky", "land", "is", "covers", 
                       "the", "ships", "sea", "deep", "earth"]

    # 1) TOKENIZACIÓN
    tokens = tokenize(input_text)

    if not tokens:
        return {
            "tokens": [],
            "embeddings_per_token": [],
            "context_vector": [],
            "attention_vector": [],
            "feed_forward_vector": [],
            "candidate_words": candidate_words,
            "logits": [],
            "probs": [],
            "selected_token": "<none>"
        }

    # 2) CÁLCULO DE EMBEDDINGS
    def get_embedding(word):
        return glove_dict[word] if word in glove_dict else np.zeros(embedding_dim, dtype=np.float32)

    embeddings_per_token = [get_embedding(t) for t in tokens]
    context_vector = np.mean(embeddings_per_token, axis=0)

    # 3) (Fake) SELF-ATTENTION
    attention_vector = context_vector.copy()  # placeholder

    # 4) (Fake) FEED-FORWARD
    rng = np.random.default_rng()  # sin semilla fija => resultados aleatorios diferentes
    W = rng.normal(0, 0.1, size=attention_vector.shape)
    b = rng.normal(0, 0.1, size=attention_vector.shape)
    feed_forward_vector = attention_vector * W + b

    # 5) LOGITS
    logits = []
    for cw in candidate_words:
        cw_emb = get_embedding(cw)
        logit = np.dot(feed_forward_vector, cw_emb)
        logits.append(logit)
    logits = np.array(logits)

    # 6) SOFTMAX
    exps = np.exp(logits - np.max(logits))
    probs = exps / exps.sum()

    # 7) SELECCIÓN (greedy)
    idx_max = np.argmax(probs)
    selected_token = candidate_words[idx_max]

    return {
        "tokens": tokens,
        "embeddings_per_token": [v.tolist() for v in embeddings_per_token],
        "context_vector": context_vector.tolist(),
        "attention_vector": attention_vector.tolist(),
        "feed_forward_vector": feed_forward_vector.tolist(),
        "candidate_words": candidate_words,
        "logits": logits.tolist(),
        "probs": probs.tolist(),
        "selected_token": selected_token
    }
