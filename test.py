import streamlit as st
import tempfile

from langchain.document_loaders.csv_loader import CSVLoader
from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "yarn-llama-2-7b-128k.Q8_0.gguf",
        model_type="llama",
        max_new_tokens=512,
        temperature = 0.5
    )
    return llm

st.title("Chat with yout files 🦙")

# Choose file type
file_type = st.sidebar.radio("Select file type:", ("CSV", "PDF"))

# Upload file
uploaded_file = st.sidebar.file_uploader(f"Upload {file_type} file", type=[file_type.lower()])

if uploaded_file:
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Process the uploaded file (you can add your logic here)
    if file_type == "CSV":

        # use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
            'delimiter': ','})
        data = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cuda:0'})

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)
        llm = load_llm()

        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
        def qa_bot(query):
            result = chain({'question': query, 'chat_history': st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result['answer']


    elif file_type == "PDF":
        # Example: Process PDF file
        # You can use libraries like PyPDF2 or pdfplumber to handle PDF files
        st.write("PDF processing logic here")

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    container = st.container()
    response_container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask about your csv file here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = qa_bot(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")



st.sidebar.write("---")
st.sidebar.write("You can choose either a CSV or PDF file and upload it.")
