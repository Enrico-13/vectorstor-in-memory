from dotenv import load_dotenv
import os
from langchain.document_loaders.pdf	import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI



def main():
    load_dotenv()
    
    embeddings = OpenAIEmbeddings()
    if not os.path.isdir('faiss_index_react'):
        pdf_path = './2210.03629.pdf'
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
        docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local('faiss_index_react')

    new_vectorstore = FAISS.load_local('faiss_index_react', embeddings, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    res = qa.invoke(input="Give me the gist of ReAct in 3 sentences")
    print(res['result'])

if __name__ == "__main__":
    main()
