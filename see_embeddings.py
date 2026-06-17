# Import the SentenceTransformer library (may need a quick install)
from sentence_transformers import SentenceTransformer

# 1. LOAD THE MODEL: This is like hiring a translator who knows how to turn text into numbers
# We use a small, fast model called 'all-MiniLM-L6-v2'
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. THE TEXT: A simple sentence we want to convert
text = "The quick brown fox jumps over the lazy dog."

# 3. ENCODE: This is where the magic happens—turning text into a list of numbers (Embedding)
embedding = model.encode(text)

print("--- EMBEDDING REVEALED ---")
# Show the first 10 numbers of the embedding
print(f"First 10 numbers of the vector: {embedding[:10]}")

# 4. DIMENSIONS: This tells us how many 'properties' the AI uses to define this sentence
print(f"Total numbers (Dimensions) in this vector: {len(embedding)}")

# 5. MEANING CHECK: A human sees words, the AI sees these coordinates
print("\nYour text has been successfully turned into a 'Vector'!")
