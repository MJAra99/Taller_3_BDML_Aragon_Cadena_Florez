"""
07_embeddings_texto.py
Objetivo: representar el título y la descripción de cada anuncio con un
          embedding de frases preentrenado (transfer learning de texto, como
          BETO "congelado" en la Lecture 15). Cada anuncio pasa a un vector
          denso de 384 (MiniLM multilingüe) o 768 (BETO) dimensiones, que
          captura el significado de la frase con su contexto, no solo qué
          palabras aparecen (bolsa de palabras).
          El modelo NO se re-entrena (no hay fine-tuning): solo se usa como
          extractor de variables. Por eso corre en CPU.

Uso (desde la carpeta del repositorio):
    python 01_Scripts/01_Data/07_embeddings_texto.py --probar
        -> revisa que estén las librerías y codifica 3 frases de prueba.
    python 01_Scripts/01_Data/07_embeddings_texto.py
        -> MiniLM multilingüe (por defecto, ~15-30 min en CPU).
    python 01_Scripts/01_Data/07_embeddings_texto.py --modelo beto
        -> BETO ajustado para similitud de frases (más lento, ~1-2 h en CPU).

Entradas: 02_Data/Raw/Kaggle/{train,test}.csv
Salidas:  02_Data/Processed/embeddings_texto_<modelo>.csv.gz
          (property_id, emb_001, ..., emb_384 o emb_768)

Requisitos: pip install -U sentence-transformers pandas
            (instala torch; la primera corrida descarga el modelo, ~0,5 GB)
"""

import argparse
import sys
import time
from pathlib import Path

MODELOS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "beto": "hiiamsid/sentence_similarity_spanish_es",
}


def raiz_repo() -> Path:
    """Carpeta del repositorio: la que contiene 02_Data (sube desde este archivo)."""
    for carpeta in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (carpeta / "02_Data").is_dir():
            return carpeta
    sys.exit("No encuentro la carpeta 02_Data. Corre el script desde la carpeta del repositorio.")


def probar(nombre_modelo: str) -> None:
    """Revisa librerías, versión y que el modelo cargue y codifique."""
    print("Python:", sys.version.split()[0])
    try:
        import torch
        import pandas as pd
        import sentence_transformers
    except ImportError as e:
        sys.exit(f"Falta una librería ({e.name}). Instala con:\n"
                 "    pip install -U sentence-transformers pandas")
    print("pandas:", pd.__version__)
    print("torch:", torch.__version__, "| GPU disponible:", torch.cuda.is_available())
    print("sentence-transformers:", sentence_transformers.__version__)
    from sentence_transformers import SentenceTransformer
    print("Cargando", nombre_modelo, "(la primera vez lo descarga)...")
    modelo = SentenceTransformer(nombre_modelo)
    frases = ["Apartamento exterior con vista a los cerros en Chicó Norte",
              "Hermoso apto con balcón y vista a la montaña, sector Chicó",
              "Casa para remodelar en Kennedy, dos pisos y patio"]
    emb = modelo.encode(frases, normalize_embeddings=True)
    sim = emb @ emb.T
    print("Dimensión del embedding:", emb.shape[1])
    print(f"Similitud frase 1 vs 2 (parecidas): {sim[0, 1]:.2f}")
    print(f"Similitud frase 1 vs 3 (distintas): {sim[0, 2]:.2f}")
    print("Todo listo. Si la 1-2 es mayor que la 1-3, el modelo está entendiendo el texto.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=MODELOS.keys(), default="minilm")
    parser.add_argument("--probar", action="store_true", help="solo revisar el entorno")
    parser.add_argument("--lote", type=int, default=64, help="tamaño de lote (bájalo si falta memoria)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="largo máximo del texto en tokens (lo demás se corta)")
    args = parser.parse_args()
    nombre_modelo = MODELOS[args.modelo]

    if args.probar:
        probar(nombre_modelo)
        return

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    raiz = raiz_repo()
    ruta_raw = raiz / "02_Data" / "Raw" / "Kaggle"
    ruta_salida = raiz / "02_Data" / "Processed" / f"embeddings_texto_{args.modelo}.csv.gz"

    # 1. Texto: título + descripción, tal como vienen (el modelo maneja
    #    mayúsculas, tildes y puntuación; no hace falta normalizar).
    datos = pd.concat([pd.read_csv(ruta_raw / "train.csv", usecols=["property_id", "title", "description"]),
                       pd.read_csv(ruta_raw / "test.csv", usecols=["property_id", "title", "description"])],
                      ignore_index=True).drop_duplicates("property_id")
    datos["texto"] = (datos["title"].fillna("").str.strip() + ". " +
                      datos["description"].fillna("").str.strip()).str.replace(r"\s+", " ", regex=True)

    # Muchos anuncios repiten texto: se codifica cada texto distinto una vez.
    textos_unicos = datos["texto"].drop_duplicates().tolist()
    print(f"Anuncios: {len(datos):,} | textos distintos: {len(textos_unicos):,}")

    # 2. Modelo y codificación.
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = SentenceTransformer(nombre_modelo, device=dispositivo)
    modelo.max_seq_length = args.max_tokens
    print(f"Modelo: {nombre_modelo} | dispositivo: {dispositivo} | max tokens: {args.max_tokens}")

    inicio = time.time()
    emb = modelo.encode(textos_unicos, batch_size=args.lote, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    print(f"Codificación lista en {(time.time() - inicio) / 60:.1f} min. Dimensión: {emb.shape[1]}")

    # 3. Volver a cada anuncio y guardar.
    indice = {t: i for i, t in enumerate(textos_unicos)}
    filas = datos["texto"].map(indice).to_numpy()
    columnas = [f"emb_{j + 1:03d}" for j in range(emb.shape[1])]
    salida = pd.DataFrame(emb[filas], columns=columnas)
    salida.insert(0, "property_id", datos["property_id"].to_numpy())
    salida.to_csv(ruta_salida, index=False, float_format="%.5f", compression="gzip")
    print("Guardado en", ruta_salida)


if __name__ == "__main__":
    main()
