from ctransformers import AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        "yarn-llama-2-7b-128k.Q8_0.gguf",
        model_type='llama',
        gpu_layers=5
    )
    return llm

llm = load_llm()
