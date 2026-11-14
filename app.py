import os
import time
import streamlit as st
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa Yapılandırması (Geniş ekran modu)
st.set_page_config(
    page_title="Hemşire Asistanı Yapay Zeka",
    page_icon="🩺",
    layout="wide"
)

# Arayüzü tamamen temizleyen (rozetleri ve sol paneli gizleyen) CSS
hide_streamlit_style = """
<style>
[data-testid="stSidebar"] {display: none !important;}
section[data-testid="stSidebar"] {display: none !important;}
button[kind="header"] {display: none !important;}
#MainMenu {visibility: hidden !important; display: none !important;}
footer {visibility: hidden !important; display: none !important;}
.stDeployButton {display: none !important;}
div[data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
div[class*="viewerBadge"] {display: none !important;}
iframe {display: none !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🩺 Klinik Doküman & Hemşire Asistanı")
st.caption("Sağlık protokolleri, kılavuzlar ve prosedür dokümanları üzerinden akıllı bilgi erişim sistemi (RAG).")

# Önce Render ortam değişkenini kontrol et, yoksa Streamlit secrets dene
active_key = os.getenv("GOOGLE_API_KEY")
if not active_key:
    try:
        active_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        active_key = ""

if not active_key:
    st.warning("⚠️ Lütfen Render Environment Variables veya Streamlit Secrets ayarlarından geçerli bir `GOOGLE_API_KEY` tanımlayınız.")
    st.stop()
client = genai.Client(api_key=active_key)

# Akıllı RAG: Dokümanları okuma, parçalama (chunking) ve indeksleme
@st.cache_resource(show_spinner="Klinik dokümanlar analiz ediliyor ve indeksleniyor...")
def initialize_rag_system():
    chunks = []
    chunk_sources = []
    
    if os.path.exists("data"):
        for filename in os.listdir("data"):
            file_path = os.path.join("data", filename)
            file_text = ""
            
            if filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_text = f.read()
                except Exception:
                    pass
            elif filename.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    file_text = "".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    pass
            
            if file_text.strip():
                # Metni 500 karakterlik anlamlı parçalara böl (RAG Chunking)
                chunk_size = 500
                overlap = 100
                for i in range(0, len(file_text), chunk_size - overlap):
                    chunk = file_text[i:i + chunk_size]
                    if len(chunk.strip()) > 50:  # Çok kısa parçaları atla
                        chunks.append(chunk)
                        chunk_sources.append(filename)

    # TF-IDF Vektörleştirici ile arama motoru matrisi oluşturma
    if chunks:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, tfidf_matrix, chunks, chunk_sources
    
    return None, None, [], []

vectorizer, tfidf_matrix, chunks, chunk_sources = initialize_rag_system()

# Soru ile en alakalı doküman parçalarını bulan fonksiyon
def search_relevant_context(query, top_k=4):
    if not chunks or vectorizer is None:
        return ""
    
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # En yüksek skorlu top_k parçayı seç
    best_indices = similarity_scores.argsort()[::-1][:top_k]
    
    context = ""
    for idx in best_indices:
        if similarity_scores[idx] > 0.05:  # Minimum benzerlik eşiği
            context += f"\n[Kaynak Dosya: {chunk_sources[idx]}]\n{chunks[idx]}\n"
            
    return context

if not chunks:
    st.warning("⚠️ `data` klasöründe okunabilir doküman bulunamadı. Lütfen depoya doküman ekleyin.")
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
            # Sadece soruyla ilgili kısımları doküman havuzundan çek (RAG Akıllı Eşleşme)
            relevant_corpus = search_relevant_context(prompt)
            
            if not relevant_corpus.strip():
                relevant_corpus = "Yüklenen dokümanlarda bu soruyla doğrudan eşleşen bir bilgi bulunamadı."

            response_text = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    full_prompt = f"""Sen profesyonel bir klinik hemşire asistanısın. Aşağıda yalnızca kullanıcının sorusuyla en ilgili klinik doküman parçaları filtrelenmiştir. Bu bilgilere dayanarak soruyu net, doğru ve profesyonel bir şekilde yanıtla.

İlgili Klinik Doküman Parçaları:
{relevant_corpus}

Kullanıcı Sorusu: {prompt}
Yanıt:"""

                    response = client.models.generate_content_stream(
                        model='gemini-2.5-flash',
                        contents=full_prompt,
                    )
                    
                    response_text = st.write_stream(chunk.text for chunk in response)
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        try:
                            response = client.models.generate_content_stream(
                                model='gemini-3.6-flash',
                                contents=full_prompt,
                            )
                            response_text = st.write_stream(chunk.text for chunk in response)
                            break
                        except Exception as err:
                            st.error(f"Sunucu yoğunluğu geçici olarak aşılamadı, lütfen birkaç saniye sonra tekrar deneyin. (Hata: {err})")

            if response_text:
                disclaimer = "\n\n⭐ **Uyarı:** *Bu asistan tarafından sağlanan yanıtlar yalnızca bilgilendirme amaçlıdır; klinik kararlar ve hasta bakımı için mutlaka resmi hastane protokollerine ve uzman hekim görüşüne başvurulmalıdır.*"
                st.markdown(disclaimer)
                
                full_message = str(response_text) + disclaimer
                st.session_state.messages.append({"role": "assistant", "content": full_message})
