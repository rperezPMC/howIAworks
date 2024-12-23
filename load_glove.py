import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from numpy.linalg import norm

def load_glove_embeddings(glove_path, embedding_dim=300, max_words=None):
    """
    Carga embeddings GloVe desde un archivo .txt,
    retornando { palabra: vector }.
    max_words (opcional) para cargar solo las primeras N palabras
    y acelerar la demo.
    """
    embeddings = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if max_words and i >= max_words:
                break
            vals = line.strip().split()
            word = vals[0]
            vector = np.array(vals[1:], dtype=np.float32)

            # Verifica que el vector tenga la dimensión esperada
            if len(vector) == embedding_dim:
                embeddings[word] = vector

    return embeddings

def reduce_to_3d(embeddings_dict):
    """
    Aplica PCA para reducir el embedding a 3D.
    Retorna un DataFrame con columnas: ["word","x","y","z"].
    """
    words = list(embeddings_dict.keys())
    vectors = np.array(list(embeddings_dict.values()))

    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    df = pd.DataFrame({
        "word": words,
        "x": vectors_3d[:,0],
        "y": vectors_3d[:,1],
        "z": vectors_3d[:,2]
    })
    return df

def cosine_similarity(vecA, vecB):
    return np.dot(vecA, vecB) / (norm(vecA) * norm(vecB))

def compute_concept_similarity(df_3d, concept_word, embeddings_dict_300d):
    """
    Calcula la similitud coseno entre 'concept_word' y cada palabra en df_3d,
    usando los embeddings de 300d originales (embeddings_dict_300d).
    Devuelve un array de similitudes, una por cada fila de df_3d.
    """
    if concept_word not in embeddings_dict_300d:
        # Si el concepto no existe en el vocabulario GloVe, retornamos ceros
        return np.zeros(len(df_3d))

    concept_vec = embeddings_dict_300d[concept_word]
    similarities = []

    for w in df_3d["word"]:
        if w in embeddings_dict_300d:
            w_vec = embeddings_dict_300d[w]
            sim = cosine_similarity(concept_vec, w_vec)
        else:
            sim = 0.0
        similarities.append(sim)

    return np.array(similarities)
