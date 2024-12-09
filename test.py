#1.加载数据
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://arxiv.org/pdf/2309.10305.pdf")

pages = loader.load_and_split()

#2.知识切片 将文档分割成均匀的块。每个块是一段原始文本
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

docs = text_splitter.split_documents(pages[:1])

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

model_name = "sentence-transformers/sentence-t5-large"

embedding = HuggingFaceEmbeddings(model_name=model_name)

vectorstore_hf = Chroma.from_documents(documents=docs, embedding=embedding , collection_name="huggingface_embed")

query = "How large is the baichuan2 vocabulary?"
result = vectorstore_hf.similarity_search(query ,k = 2)

print(result)