import os
import tempfile
import streamlit as st
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

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
    secret_key = st.secrets.get("GOOGLE_API_KEY", "")
    api_key = st.text_input("Google Gemini API Key", value=secret_key, type="password", help="Google AI Studio API anahtarı.")
    st.divider()
    
    uploaded_files = st.file_uploader("Ek Geçici Doküman Yükleyin (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

# API Anahtarı Kontrolü
active_key = api_key if api_key else secret_key

if not active_key:
    st.warning("⚠️ Lütfen devam etmek için Google Gemini API Anahtarınızı Secrets alanına giriniz.")
    st.stop()

# Yapılandırma
os.environ["GOOGLE_API_KEY"] = active_key
genai.configure(api_key=active_key)

# LLM ve Embedding Ayarları
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=active_key)
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=active_key)

# Dokümanları Yükleme ve İndeksleme İşlemi
@st.cache_resource(show_spinner="Dokümanlar taranıyor ve yapay zeka hafızası oluşturuluyor...")
def load_data_and_create_index(_uploaded_files_list):
    documents = []
    
    # 1. GitHub'daki 'data' klasöründeki kalıcı dosyaları oku
    if os.path.exists("data"):
        try:
            data_reader = SimpleDirectoryReader("data")
            documents.extend(data_reader.load_data())
        except Exception as e:
            st.error(f"Data klasörü okunurken hata: {e}")

    # 2. Arayüzden anlık yüklenen geçici dosyaları oku
    if _uploaded_files_list:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in _uploaded_files_list:
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
    query_engine = index.as_query_engine(streaming=True)

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
            streaming_response = query_engine.query(prompt)
            response_text = st.write_stream(streaming_response.response_gen)
            st.session_state.messages.append({"role": "assistant", "content": str(response_text)})
