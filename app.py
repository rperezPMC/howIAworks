import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from load_glove import load_glove_embeddings
from text_generation_sim import simulate_text_generation_visual

# -----------------------------------------------------------------------------
# 1. CARGA DE GLOVE
# -----------------------------------------------------------------------------
GLOVE_PATH = "glove.42B.300d.txt"  # Ajusta la ruta
EMBEDDING_DIM = 300
MAX_WORDS = 10000 # Para no saturar, cargamos 3000 (puedes variar)

print("Cargando embeddings GloVe, espera un poco...")
glove_dict = load_glove_embeddings(GLOVE_PATH, EMBEDDING_DIM, max_words=MAX_WORDS)
print(f"Embeddings cargados: {len(glove_dict)} palabras.")

# -----------------------------------------------------------------------------
# 2. REDUCCIÓN A 3D (GRÁFICA DE EMBEDDINGS)
# -----------------------------------------------------------------------------
def reduce_to_3d(embeddings_dict, n_points=200):
    """
    Toma un subconjunto de n_points palabras y reduce a 3D (PCA).
    Retorna un DataFrame con columns ["word","x","y","z"].
    """
    words = list(embeddings_dict.keys())[:n_points]
    vectors = np.array([embeddings_dict[w] for w in words])

    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    df = pd.DataFrame({
        "word": words,
        "x": vectors_3d[:, 0],
        "y": vectors_3d[:, 1],
        "z": vectors_3d[:, 2]
    })
    return df

df_3d = reduce_to_3d(glove_dict, n_points=1000)
fig_3d = px.scatter_3d(
    df_3d,
    x="x", y="y", z="z",
    hover_name="word",
    title="Embeddings 3D (PCA) - Subconjunto de 10.000 palabras",
    height=600
)
fig_3d.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

# -----------------------------------------------------------------------------
# 3. CREACIÓN DE LA APP DASH
# -----------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(fluid=True, children=[

    html.H1("Visualizador de Embeddings + Simulación de un Paso de Generación"),

    html.Hr(),

    # Sección A: Gráfica 3D de embeddings
    html.Div([
        html.H2("1) Exploración de Embeddings 3D (PCA)"),
        dcc.Graph(
            id="embedding-3d-plot",
            figure=fig_3d,
            style={"width": "100%", "height": "600px"}
        ),
        html.P("Mueve o rota la gráfica 3D para explorar cómo se distribuyen los embeddings.")
    ], style={"marginBottom": "50px"}),

    html.Hr(),

    # Sección B: Simulación de Generación
    html.Div([
        html.H2("2) Simulación de Generación (un solo token)"),

        html.Label("Introduce una frase en inglés:"),
        dcc.Input(
            id="input-text",
            type="text",
            placeholder="e.g. the ocean covers",
            style={"width": "400px", "marginRight": "10px"}
        ),
        html.Button("Simular", id="simulate-button", n_clicks=0, className="btn btn-primary"),

        # Contenedor para mostrar la salida textual
        html.Div(id="output-div", style={"marginTop":"20px"}),

        html.Hr(),

        html.H3("Context Vector (primeros 10 valores)"),
        dcc.Graph(id="context-graph"),

        html.H3("(Fake) Feed-Forward (primeros 10 valores)"),
        dcc.Graph(id="feedforward-graph"),

        html.H3("Logits (vocabulario candidato)"),
        dcc.Graph(id="logits-graph"),

        html.H3("Probabilidades (Softmax)"),
        dcc.Graph(id="probs-graph"),
    ]),

])

# -----------------------------------------------------------------------------
# 4. CALLBACK: Simular el proceso y mostrar gráficas
# -----------------------------------------------------------------------------
@app.callback(
    [
        Output("output-div", "children"),
        Output("context-graph", "figure"),
        Output("feedforward-graph", "figure"),
        Output("logits-graph", "figure"),
        Output("probs-graph", "figure")
    ],
    [Input("simulate-button", "n_clicks")],
    [State("input-text", "value")]
)
def run_simulation(n_clicks, user_input):
    if n_clicks < 1 or not user_input:
        return ["", {}, {}, {}, {}]

    # Llamamos a la simulación
    results = simulate_text_generation_visual(user_input, glove_dict, embedding_dim=EMBEDDING_DIM)

    # Extraemos datos
    tokens = results["tokens"]
    embeddings_per_token = results["embeddings_per_token"]  # lista de vectores
    context_vector = np.array(results["context_vector"])
    feed_forward_vector = np.array(results["feed_forward_vector"])
    candidate_words = results["candidate_words"]
    logits = np.array(results["logits"])
    probs = np.array(results["probs"])
    selected_token = results["selected_token"]

    # 1) Mostramos tokens y un preview de sus embeddings
    #    (mostramos los primeros 5 valores para no saturar)
    if tokens:
        list_items = []
        for i, t in enumerate(tokens):
            emb_preview = embeddings_per_token[i][:5]
            list_items.append(html.Li(f"{t} -> {emb_preview}..."))
        token_list = html.Ul(list_items)
    else:
        token_list = html.Div("No tokens found.")

    summary_div = html.Div([
        html.H4("Tokens de entrada:"),
        token_list,
        html.H4(f"Token seleccionado (salida): {selected_token}")
    ])

    # 2) Context Vector (solo 10 valores)
    x_inds = list(range(10))
    context_fig = px.bar(
        x=x_inds,
        y=context_vector[:10],
        labels={"x":"Índice", "y":"Valor"},
        title="Context Vector (10 primeros)"
    )

    # 3) Feed-Forward (solo 10 valores)
    ff_fig = px.bar(
        x=x_inds,
        y=feed_forward_vector[:10],
        labels={"x":"Índice", "y":"Valor"},
        title="Feed-Forward (10 primeros)"
    )

    # 4) Logits
    logits_fig = px.bar(
        x=candidate_words,
        y=logits,
        labels={"x":"Palabra Candidata", "y":"Logit"},
        title="Logits"
    )

    # 5) Probabilidades
    probs_fig = px.bar(
        x=candidate_words,
        y=probs,
        labels={"x":"Palabra Candidata", "y":"Probabilidad"},
        title="Probabilidades (Softmax)"
    )

    return [summary_div, context_fig, ff_fig, logits_fig, probs_fig]


# -----------------------------------------------------------------------------
# 5. EJECUCIÓN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=False, port=8050)
