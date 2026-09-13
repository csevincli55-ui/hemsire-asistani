import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.prompts import PromptTemplate

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Hemşire Asistanı Yapay Zeka",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Klinik Doküman & Hemşire Asistanı")
st.caption("Sağlık protokolleri, kılavuzlar ve prosedür dokümanları üzerinden güvenilir bilgi erişim sistemi.")

# Sol Panel - Ayarlar ve Dosya Yükleme
with st.sidebar:
    st.header("⚙️ Ayarlar & Dokümanlar")
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=secret_key, type="password", help="API anahtarınızı giriniz.")
    st.divider()
    
    uploaded_files = st.file_uploader(
        "Klinik Dokümanları Yükleyin (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

# Oturum Durumu (Session State) Tanımları
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

# Hemşire Asistanına Özel Sıkı Sistem Talimatı (Hallucination Önleyici)
NURSE_SYSTEM_PROMPT = (
    "Sen uzman bir Klinik Hemşire Asistanısın.\n"
    "Görevin, sana sağlanan klinik dokümanlar ve protokoller doğrultusunda hemşirelerin sorularını doğru ve net bir şekilde yanıtlamaktır.\n\n"
    "KURALLAR:\n"
    "1. SADECE sana sağlanan bağlam (dokümanlar) içerisindeki bilgilere dayanarak cevap ver.\n"
    "2. Eğer aranan bilgi yüklenen dokümanlarda yoksa, KESİNLİKLE tahmin yürütme. 'Bu bilgi yüklenen belgelerde bulunmamaktadır.' yanıtını ver.\n"
    "3. Yanıt verirken kullanılan protokol veya doküman adını/sayfasını kaynak olarak belirt.\n"
    "4. Tıbbi güvenliği kesinlikle riske atma.\n"
)

QA_TEMPLATE_STR = (
    "Aşağıdaki klinik doküman parçalarını kullanarak hemşirenin sorusunu yanıtla.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Soru: {query_str}\n"
    "Yanıt:"
)

# Dokümanların İndekslenmesi
if uploaded_files and api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    if st.session_state.index is None or st.sidebar.button("Dokümanları Yeniden İndeksle"):
        with st.spinner("Dokümanlar işleniyor ve vektör veritabanı oluşturuluyor..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Doküman Okuyucu ve Model Yapılandırması
                reader = SimpleDirectoryReader(temp_dir)
                documents = reader.load_data()
                
                Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1, system_prompt=NURSE_SYSTEM_PROMPT)
                Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
                
                # Vektör İndeksini Oluştur
                st.session_state.index = VectorStoreIndex.from_documents(documents)
                st.success("✅ Dokümanlar başarıyla indekslendi! Sorularınızı sorabilirsiniz.")

elif not api_key:
    st.warning("⚠️ Lütfen devam etmek için sol menüden OpenAI API Anahtarınızı giriniz.")

# Sohbet Arayüzü
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Klinik prosedür veya dokümanlar hakkında soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.index is None:
        with st.chat_message("assistant"):
            st.error("Lütfen önce sol panelden doküman yükleyin ve API anahtarınızı girin.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Dokümanlar taranıyor..."):
                query_engine = st.session_state.index.as_query_engine(
                    similarity_top_k=3,
                    text_qa_template=PromptTemplate(QA_TEMPLATE_STR)
                )
                response = query_engine.query(prompt)
                response_text = str(response)
                
                # Kaynak ve Sayfa Bilgisi Çıkarma
                sources = []
                if hasattr(response, "source_nodes"):
                    for node in response.source_nodes:
                        file_name = node.metadata.get("file_name", "Doküman")
                        page_label = node.metadata.get("page_label", "")
                        source_str = f"📄 **{file_name}**" + (f" (Sayfa {page_label})" if page_label else "")
                        if source_str not in sources:
                            sources.append(source_str)
                
                if sources:
                    response_text += "\n\n---\n**📚 Referans Dokümanlar:**\n" + "\n".join([f"- {s}" for s in sources])
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
