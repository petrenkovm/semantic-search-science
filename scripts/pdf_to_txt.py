import os, fitz, sys

def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def convert_all(folder):
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            pdf_path = os.path.join(folder, fname)
            txt_path = pdf_path.replace(".pdf", ".txt")
            text = extract_text(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ {fname} → {os.path.basename(txt_path)}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    convert_all(folder)
