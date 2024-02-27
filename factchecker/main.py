from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 50
)

loader = TextLoader('factchecker/facts.txt')

docs = loader.load_and_split(
    text_splitter=text_splitter
)

db = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    persist_directory = "factchecker/emb"
)

results = db.similarity_search_with_score("intresting fact about english language")

print(results)