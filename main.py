################################################################################################################################
# Bibliotecas
################################################################################################################################
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
import streamlit as st
from datetime import datetime
import pandas as pd
import os

from azure.storage.blob import BlobServiceClient
from io import BytesIO

from utils import *

from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
)

from langchain_openai.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import streamlit as st

################################################################################################################################
# Ambiente
################################################################################################################################
# Configurações do Blob Storage
azure_blob_connection_string = st.secrets["AZURE_BLOB_CONNECTION_STRING"]
container_name = st.secrets["AZURE_BLOB_CONTAINER_NAME"]
blob_name = "chat_log.xlsx"

# Cria o BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(
    azure_blob_connection_string
)
container_client = blob_service_client.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)

embeddings_deployment = st.secrets["AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"]
embeddings_model = st.secrets["AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME"]

azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
azure_deployment = st.secrets["AZURE_OPENAI_DEPLOYMENT"]
azure_model = st.secrets["AZURE_OPENAI_MODEL"]
azure_api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
azure_key = st.secrets["AZURE_OPENAI_API_KEY"]

vectorstore_address = st.secrets["TRADERS_AZURESEARCH_VECTORSTORE_ADDRESS"]
vectorstore_key = st.secrets["TRADERS_AZURESEARCH_VECTORSTORE_KEY"]

TRADERS_AZURESEARCH_FIELDS_CHUNK_ID = "chunk_id"
TRADERS_AZURESEARCH_FIELDS_PARENT_ID = "parent_id"
TRADERS_AZURESEARCH_FIELDS_CHUNK = "chunk"
TRADERS_AZURESEARCH_FIELDS_TITLE = "title"
TRADERS_AZURESEARCH_FIELDS_TEXT_VECTOR = "text_vector"

AZURESEARCH_FIELDS_CHUNK_ID = st.secrets["TRADERS_AZURESEARCH_FIELDS_CHUNK_ID"]
AZURESEARCH_FIELDS_PARENT_ID = st.secrets["TRADERS_AZURESEARCH_FIELDS_PARENT_ID"]
AZURESEARCH_FIELDS_CHUNK = st.secrets["TRADERS_AZURESEARCH_FIELDS_CHUNK"]
AZURESEARCH_FIELDS_TITLE = st.secrets["TRADERS_AZURESEARCH_FIELDS_TITLE"]
AZURESEARCH_FIELDS_TEXT_VECTOR = st.secrets["TRADERS_AZURESEARCH_FIELDS_TEXT_VECTOR"]

fields = [
    SimpleField(
        name=AZURESEARCH_FIELDS_CHUNK_ID,
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SimpleField(name=AZURESEARCH_FIELDS_PARENT_ID, type=SearchFieldDataType.String),
    SimpleField(name=AZURESEARCH_FIELDS_CHUNK, type=SearchFieldDataType.String),
    SimpleField(name=AZURESEARCH_FIELDS_TITLE, type=SearchFieldDataType.String),
    SearchField(
        name=AZURESEARCH_FIELDS_TEXT_VECTOR,
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
    ),
]
#############################################################################################################
# Parâmetros das APIs
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment,
    model=embeddings_model,
    azure_endpoint=azure_endpoint,
    openai_api_type="azure",
    chunk_size=1,
)

llm = AzureChatOpenAI(
    azure_deployment=azure_deployment,
    model=azure_model,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
    api_key=azure_key,
    openai_api_type="azure",
)

# TRADERS_AZURESEARCH_INDEX_NAME = "vector-1727123085165"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vectorstore_address,
    azure_search_key=vectorstore_key,
    index_name=st.secrets["TRADERS_AZURESEARCH_INDEX_NAME"],
    embedding_function=embeddings.embed_query,
    fields=fields,
)

#############################################################################################################
# Funções do Chat
data_hoje = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prompt_addition = f"Today is {data_hoje}. "
PROMPT = """
You are the assistant for the bank manager and talks exclusively about financial subjects. The reply must be always in Portuguese from Brazil.
Answers must be based on the knowledge base.

Context:
{context}

Current conversation:
{chat_history}

Human: {question}
"""


def cria_chain_conversa():
    # Defina chaves únicas para a memória e a cadeia desta página
    memory_key = "memory_traders"
    chain_key = "chain_traders"

    # Verifique se a memória já existe no session_state
    if memory_key not in st.session_state:
        st.session_state[memory_key] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
        )
    memory = st.session_state[memory_key]

    retriever = vector_store.as_retriever(search_type="similarity", k=k_similarity)
    prompt = PromptTemplate.from_template(template=prompt_addition + PROMPT)

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True,
    )

    st.session_state[chain_key] = chat_chain


###############################################################################################################
####################### Parâmetros de modelagem ###############################################################
k_similarity = 10  # lang_chain similarity search

# Tamanhos de chunk_size recomendados
pdf_chunk = 2048

# Sobreposição recomendada de 10%
pdf_overlap = 205
##############################################################################################################

################################################################################################################################
# UX
################################################################################################################################

# Início da aplicação
st.set_page_config(
    page_title="Traders",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Leitura do arquivo CSS de estilização
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


################################################################################################################################
# UI
################################################################################################################################
def main():
    chain_key = "chain_traders"
    memory_key = "memory_traders"

    if chain_key not in st.session_state:
        cria_chain_conversa()

    chain = st.session_state[chain_key]
    memory = st.session_state[memory_key]

    mensagens = memory.load_memory_variables({})["chat_history"]

    # Container para exibição no estilo Chat message
    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    # Espaço para o usuário incluir a mensagem e estruturação da conversa
    nova_mensagem = st.chat_input("Digite uma mensagem")
    if nova_mensagem:
        chat = container.chat_message("human")
        chat.markdown(nova_mensagem)
        chat = container.chat_message("ai")
        chat.markdown("Gerando resposta...")

        with get_openai_callback() as cb:
            resposta = chain.invoke({"question": nova_mensagem})
        st.session_state["ultima_resposta"] = resposta
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

        filename = f"{PASTA_QA}/chat_log.xlsx"

        data = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pergunta": nova_mensagem,
            "resposta": resposta["answer"],
            "total tokens": cb.total_tokens,
        }

        df_new_row = pd.DataFrame([data])

        # REGISTRAR LOG BLOB STORAGE
        # Verifica se o blob já existe
        try:
            # Tenta fazer o download do blob existente
            stream_downloader = blob_client.download_blob()
            downloaded_bytes = stream_downloader.readall()
            df_existing = pd.read_excel(BytesIO(downloaded_bytes))
            df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        except Exception as e:
            # Se o blob não existir, cria um novo DataFrame
            df_combined = df_new_row

        # Escreve o DataFrame combinado em um objeto BytesIO
        output = BytesIO()
        df_combined.to_excel(output, index=False)
        output.seek(0)  # Move o ponteiro para o início do arquivo

        # Faz o upload do objeto BytesIO para o Blob Storage
        blob_client.upload_blob(output, overwrite=True)

        # REGISTRAR LOG LOCAL
        # if os.path.exists(filename):
        #     # O arquivo existe, lê os dados existentes
        #     df_existing = pd.read_excel(filename)
        #     df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)
        # else:
        #     # O arquivo não existe, cria um novo DataFrame
        #     df_combined = df_new_row

        # # Escreve o DataFrame no arquivo Excel
        # df_combined.to_excel(filename, index=False)

        st.rerun()


if __name__ == "__main__":
    main()
