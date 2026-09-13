import os
import tempfile
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Hemşire Asistanı Yapay Zeka",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Klinik Doküman & Hemşire Asistanı")
st.caption("Sağlık protokolleri, kılavuzlar ve prosedür dokümanları üzerinden güvenilir bilgi erişim sistemi.")

# Sol Panel - Ayarlar
with st.sidebar:
    st.header("⚙️ Ayarlar & Dokümanlar")
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=secret_key, type="password", help="API anahtarınızı giriniz.")
    st.divider()
    
    uploaded_files = st.file_uploader("Ek Geçici Doküman Yükleyin (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

# API Anahtarı Kontrolü
active_key = api_key if api_key else secret_key

if not active_key:
    st.warning("⚠️ Lütfen devam etmek için sol menüden OpenAI API Anahtarınızı giriniz.")
    st.stop()

os.environ["OPENAI_API_KEY"] = active_key
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding()

# Dokümanları Yükleme ve İndeksleme İşlemi
@st.cache_resource(show_spinner="Kalıcı ve geçici dokümanlar indeksleniyor...")
def load_data_and_create_index(uploaded_files_list):
    documents = []
    
    # 1. GitHub'daki 'data' klasöründeki kalıcı dosyaları oku
    if os.path.exists("data"):
        try:
            data_reader = SimpleDirectoryReader("data")
            documents.extend(data_reader.load_data())
        except Exception as e:
            pass

    # 2. Arayüzden anlık yüklenen geçici dosyaları oku
    if uploaded_files_list:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files_list:
                temp_filepath = os.path.join(temp_dir, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getvalue())
            
            temp_reader = SimpleDirectoryReader(temp_dir)
            documents.extend(temp_reader.load_data())

    if not documents:
        return None

    return VectorStoreIndex.from_documents(documents)

# İndeksi Oluştur
index = load_data_and_create_index(uploaded_files)

if index is None:
    st.info("ℹ️ Henüz sisteme yüklü bir doküman bulunmuyor. GitHub 'data' klasörüne dosya ekleyebilir veya sol menüden yükleyebilirsiniz.")
else:
    query_engine = index.as_query_engine()

    # Chat Geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Klinik prosedür veya dokümanlar hakkında soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Dokümanlar taranıyor..."):
                response = query_engine.query(prompt)
                st.markdown(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
