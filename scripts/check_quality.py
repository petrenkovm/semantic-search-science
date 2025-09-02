import os, sys

def check_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return {
        "file": os.path.basename(path),
        "length": len(text),
        "lines": text.count("\n"),
        "non_ascii": any(ord(c) > 127 for c in text),
        "symbols": any(c in "!@#$%^&*()[]{}<>?" for c in text)
    }

def scan_folder(folder):
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            result = check_text(os.path.join(folder, fname))
            print(result)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    scan_folder(folder)
