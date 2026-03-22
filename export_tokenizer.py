import struct
from transformers import AutoTokenizer

print("Downloading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Exporting tokenizer.bin...")
with open("tokenizer.bin", "wb") as f:
    # 1. Write exact vocab size (32000)
    f.write(struct.pack("<I", 32000))
    
    for i in range(32000):
        t = tokenizer.convert_ids_to_tokens(i)
        if t is None: t = ""
        if isinstance(t, bytes): t = t.decode("utf-8", errors="ignore")
        
        # Convert SentencePiece blocks into standard spaces
        t = t.replace('\u2581', ' ')
        
        # Ignore ugly raw-byte fallback tags (like <0x0A>)
        if t.startswith("<0x") and t.endswith(">"): t = "\n" if t == "<0x0A>" else ""
        
        b = t.encode("utf-8", errors="ignore")
        
        # 2. Write length, 3. Write bytes
        f.write(struct.pack("<I", len(b)))
        if len(b) > 0:
            f.write(b)

print("Tokenizer exported successfully!")