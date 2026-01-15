from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)  # file_path must be a PDF file
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
