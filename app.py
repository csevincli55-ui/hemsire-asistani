import os
import tempfile
import streamlit as st
import google.generativeai as genai

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
    uploaded_files = st.file_uploader("Doküman Yükleyin (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

active_key = api_key.strip() if api_key else secret_key.strip()

if not active_key:
    st.warning("⚠️ Lütfen devam etmek için geçerli bir Google Gemini API Anahtarı giriniz.")
    st.stop()

# Google API Konfigürasyonu
genai.configure(api_key=active_key)

# Metin Çıkarma ve Birleştirme Fonksiyonu (PyPDF ve yerel okuma)
@st.cache_resource(show_spinner="Dokümanlar okunuyor...")
def load_all_documents(_files):
    all_text = ""
    
    # 1. GitHub 'data' klasöründeki dosyaları oku
    if os.path.exists("data"):
        for filename in os.listdir("data"):
            file_path = os.path.join("data", filename)
            if filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        all_text += f"\n\n--- Dosya: {filename} ---\n" + f.read()
                except Exception:
                    pass
            elif filename.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    pdf_content = "".join([page.extract_text() or "" for page in reader.pages])
                    all_text += f"\n\n--- Dosya: {filename} ---\n" + pdf_content
                except Exception:
                    pass

    # 2. Sol menüden yüklenen geçici dosyaları oku
    if _files:
        for file in _files:
            if file.name.endswith(".txt"):
                try:
                    content = file.getvalue().decode("utf-8")
                    all_text += f"\n\n--- Yüklenen Dosya: {file.name} ---\n" + content
                except Exception:
                    pass
            elif file.name.endswith(".pdf"):
                try:
                    import pypdf
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    reader = pypdf.PdfReader(tmp_path)
                    pdf_content = "".join([page.extract_text() or "" for page in reader.pages])
                    all_text += f"\n\n--- Yüklenen Dosya: {file.name} ---\n" + pdf_content
                    os.unlink(tmp_path)
                except Exception:
                    pass

    return all_text

document_corpus = load_all_documents(uploaded_files)

if not document_corpus.strip():
    st.info("ℹ️ Lütfen GitHub `data` klasörüne doküman ekleyin veya sol menüden PDF/TXT dosyası yükleyin.")
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
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
            full_prompt = f"""Sen profesyonel bir klinik hemşire asistanısın. Aşağıda sağlanan klinik dokümanlardaki bilgilere dayanarak kullanıcının sorusunu net, doğru ve detaylı bir şekilde yanıtla. 

Klinik Dokümanlar:
{document_corpus}

Kullanıcı Sorusu: {prompt}
Yanıt:"""

            response = gemini_model.generate_content(full_prompt, stream=True)
            response_text = st.write_stream(chunk.text for chunk in response)
            
            st.session_state.messages.append({"role": "assistant", "content": str(response_text)})
