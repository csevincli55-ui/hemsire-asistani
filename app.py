import os
import streamlit as st
from google import genai

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
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Google Gemini API Key", value=secret_key, type="password")
    st.divider()
    st.info("ℹ️ Bu asistan, yalnızca GitHub deposundaki kalıcı klinik dokümanlar üzerinden yanıt vermektedir.")

active_key = api_key.strip() if api_key else secret_key.strip()

if not active_key:
    st.warning("⚠️ Lütfen devam etmek için geçerli bir Google Gemini API Anahtarı giriniz.")
    st.stop()

# Yeni Nesil Google GenAI İstemcisi
client = genai.Client(api_key=active_key)

# Metin Çıkarma ve Birleştirme Fonksiyonu (Yalnızca GitHub 'data' klasörü)
@st.cache_resource(show_spinner="Klinik dokümanlar okunuyor...")
def load_all_documents():
    all_text = ""
    
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

    return all_text

document_corpus = load_all_documents()

if not document_corpus.strip():
    st.warning("⚠️ GitHub üzerindeki `data` klasöründe okunabilir doküman bulunamadı. Lütfen depoya doküman ekleyin.")
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
            try:
                full_prompt = f"""Sen profesyonel bir klinik hemşire asistanısın. Aşağıda sağlanan klinik dokümanlardaki bilgilere dayanarak kullanıcının sorusunu net, doğru ve detaylı bir şekilde yanıtla. 

Klinik Dokümanlar:
{document_corpus}

Kullanıcı Sorusu: {prompt}
Yanıt:"""

                response = client.models.generate_content_stream(
                    model='gemini-2.5-flash',
                    contents=full_prompt,
                )
                
                response_text = st.write_stream(chunk.text for chunk in response)
                
                # Her cevabın altında görünecek yıldızlı uyarı
                disclaimer = "\n\n⭐ **Uyarı:** *Bu asistan tarafından sağlanan yanıtlar yalnızca bilgilendirme amaçlıdır; klinik kararlar ve hasta bakımı için mutlaka resmi hastane protokollerine ve uzman hekim görüşüne başvurulmalıdır.*"
                st.markdown(disclaimer)
                
                full_message = str(response_text) + disclaimer
                st.session_state.messages.append({"role": "assistant", "content": full_message})
            except Exception as e:
                try:
                    response = client.models.generate_content_stream(
                        model='gemini-3.6-flash',
                        contents=full_prompt,
                    )
                    response_text = st.write_stream(chunk.text for chunk in response)
                    disclaimer = "\n\n⭐ **Uyarı:** *Bu asistan tarafından sağlanan yanıtlar yalnızca bilgilendirme amaçlıdır; klinik kararlar ve hasta bakımı için mutlaka resmi hastane protokollerine ve uzman hekim görüşüne başvurulmalıdır.*"
                    st.markdown(disclaimer)
                    full_message = str(response_text) + disclaimer
                    st.session_state.messages.append({"role": "assistant", "content": full_message})
                except Exception as err:
                    st.error(f"Yanıt oluşturulurken bir hata oluştu: {err}")
