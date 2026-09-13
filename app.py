import os
import tempfile
import streamlit as st
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Hemşire Asistanı Yapay Zeka",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Klinik Doküman & Hemşire Asistanı")
st.caption("Sağlık protokolleri, kılavuzlar ve prosedür dokümanları üzerinden güvenilir bilgi erişim sistemi.")

# API Key Alımı
secret_key = st.secrets.get("GOOGLE_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Ayarlar & Dokümanlar")
    api_key = st.text_input("Google Gemini API Key", value=secret_key, type="password")
    st.divider()
    uploaded_files = st.file_uploader("Ek Geçici Doküman Yükleyin (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

active_key = api_key.strip() if api_key else secret_key.strip()

if not active_key:
    st.warning("⚠️ Lütfen devam etmek için geçerli bir Google Gemini API Anahtarı giriniz.")
    st.stop()

# Google API Konfigürasyonu
os.environ["GOOGLE_API_KEY"] = active_key
genai.configure(api_key=active_key)

# Doküman Yükleme Fonksiyonu
@st.cache_resource(show_spinner="Dokümanlar taranıyor ve indeksleniyor...")
def load_data_and_create_index(_files):
    documents = []
    
    if os.path.exists("data"):
        try:
            documents.extend(SimpleDirectoryReader("data").load_data())
        except Exception as e:
            st.error(f"Data klasörü hatası: {e}")

    if _files:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in _files:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
            documents.extend(SimpleDirectoryReader(temp_dir).load_data())

    if not documents:
        return None

    return VectorStoreIndex.from_documents(documents)

# İndeksleme
index = load_data_and_create_index(uploaded_files)

if index is None:
    st.info("ℹ️ Lütfen GitHub `data` klasörüne doküman ekleyin veya sol menüden dosya yükleyin.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Dokümanlar hakkında bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Retriever ile dokümanlardan ilgili kısımları bulma
            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(prompt)
            
            context_text = "\n\n".join([node.get_content() for node in nodes])
            
            # Doğrudan Gemini ile güvenli yanıt üretme
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            full_prompt = f"""Sen profesyonel bir hemşire asistanısın. Aşağıdaki klinik dokümanlara dayanarak soruyu yanıtla. Eğer bilgi dokümanlarda yoksa bunu belirt.

Klinik Dokümanlar:
{context_text}

Soru: {prompt}
Yanıt:"""

            response = gemini_model.generate_content(full_prompt, stream=True)
            response_text = st.write_stream(chunk.text for chunk in response)
            
            st.session_state.messages.append({"role": "assistant", "content": str(response_text)})
