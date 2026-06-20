# Import the FAISS library to store our results
from langchain_community.vectorstores import FAISS
# Import the same model we used to see embeddings earlier
from langchain_community.embeddings import DeterministicFakeEmbedding

# 1. PREPARE OUR CHUNKS: Imagine these were pulled from your PDF
pdf_chunks = [
    "The heart pumps blood to the rest of the body.",
    "The lungs are responsible for oxygenating the blood.",
    "The brain sends electrical signals to the muscles.",
    "Digestion begins in the mouth with saliva."
]

# 2. SETUP THE TRANSLATOR: Using our 'Math Model' (Size 384)
embeddings = DeterministicFakeEmbedding(size=384)

print("--- STARTING THE ASSEMBLY LINE ---")

# 3. THE LOOP & STORAGE: We use 'from_texts' which handles the loop for us!
# This takes the list, passes each through the model, and creates the Vector Store.
vector_db = FAISS.from_texts(pdf_chunks, embeddings)

# 4. VERIFY: Let's see how many items are now in our 'Memory Bank'
# We check the 'index', which is the internal map FAISS creates.
total_items = vector_db.index.ntotal

print(f"Success! We converted {len(pdf_chunks)} chunks into vectors.")
print(f"The Memory Bank now contains {total_items} indexed items.")

# 5. SAVE: Store this processed data into a folder
vector_db.save_local("processed_pdf_bank")
