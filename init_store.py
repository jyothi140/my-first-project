# Import the FAISS class from LangChain's community tools
from langchain_community.vectorstores import FAISS
# Import a basic embedding model to help initialize the store
from langchain_community.embeddings import DeterministicFakeEmbedding

# 1. SETUP THE 'TRANSLATOR': We need a model to define the size of our 'filing cabinet'
# We use a 'Fake' embedding with size 384 (matching our previous lesson)
embeddings = DeterministicFakeEmbedding(size=384)

# 2. CREATE DUMMY DATA: You can't initialize an empty FAISS store, 
# so we give it one tiny piece of 'starter' information.
text_data = ["Initial knowledge base started."]

# 3. INITIALIZE THE STORE: This creates the 'Memory Bank' in your computer's RAM
vector_store = FAISS.from_texts(text_data, embeddings)

# 4. SAVE LOCALLY: This turns the RAM memory into actual files in your VS Code folder
# We will name our folder 'my_local_bank'
vector_store.save_local("my_local_bank")

print("--- SYSTEM MESSAGE ---")
print(" Local Vector Store 'my_local_bank' has been created!")
print("Look at your VS Code file explorer on the left to see the new folder.")
