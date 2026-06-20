# Import the FAISS library to handle our vector database
from langchain_community.vectorstores import FAISS
# Import the same embedding model we used to build the bank
from langchain_community.embeddings import DeterministicFakeEmbedding

# 1. SETUP THE 'TRANSLATOR': Must use the same size (384) as before
embeddings = DeterministicFakeEmbedding(size=384)

# 2. LOAD THE MEMORY BANK: Opening our saved local database
# We use allow_dangerous_deserialization because we created this file ourselves
db = FAISS.load_local("processed_pdf_bank", embeddings, allow_dangerous_deserialization=True)

# 3. THE QUESTION: Change this string to ask something else!
user_question = "How does the heart work?"

print(f"--- SEARCHING FOR: {user_question} ---")

# 4. PERFORM SEARCH: We ask for the 'k=3' (top 3) most similar chunks
results = db.similarity_search(user_question, k=3)

# 5. DISPLAY RESULTS: We loop through the results and print them
print(f"\nFound {len(results)} relevant paragraphs:\n")
for i, doc in enumerate(results):
    # Print the result number and the content found
    print(f"Result #{i+1}:")
    print(f"{doc.page_content}\n")

