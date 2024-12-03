### ====================================
### Importações e Globais
### ====================================
from pathlib import Path
import os


PASTA_RAIZ = Path(__file__).parent
PASTA_ARQUIVOS = Path(__file__).parent / "uploaded_files"
PASTA_QA = Path(__file__).parent / "QA"
if not os.path.exists(PASTA_ARQUIVOS):
    os.makedirs(PASTA_ARQUIVOS)
if not os.path.exists(PASTA_QA):
    os.makedirs(PASTA_QA)