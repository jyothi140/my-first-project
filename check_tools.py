# Import the sys library to check our Python version
import sys 

# Try to import the FAISS library (the memory bank tool)
try:
    import faiss
    print(" FAISS is installed and ready!")
except ImportError:
    print(" FAISS is MISSING. Check the terminal steps below.")

# Try to import LangChain's community vector store module
try:
    from langchain_community.vectorstores import FAISS
    print(" LangChain-FAISS integration is ready!")
except ImportError:
    print(" LangChain Community is MISSING.")

# Print the Python version to ensure we are in the right environment
print(f"Running on Python version: {sys.version}")
