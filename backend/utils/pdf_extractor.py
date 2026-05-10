# backend/utils/pdf_extractor.py
import io
import pdfplumber   # pip install pdfplumber

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t.strip())
    except Exception as e:
        print(f"[PDF] Erreur extraction : {e}")
    return "\n".join(text_parts)