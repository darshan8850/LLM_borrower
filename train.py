import pandas as pd
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def main():
  
    loader = CSVLoader(file_path="./new_data.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    model = "stabilityai/stablelm-zephyr-3b"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", do_sample=True, top_k=1, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,max_new_tokens=2048)

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectorstore = FAISS.from_documents(data, embeddings)
    DB_FAISS_PATH = "vectorstore/db_faiss"
    vectorstore.save_local(DB_FAISS_PATH)

    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", return_source_documents=True, retriever=vectorstore.as_retriever())

    query = "What are details of Marjorie Snyder?"
    result = chain(query)
    print(result['result'])

if __name__ == '__main__':
    main()
