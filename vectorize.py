import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # Используем OpenAI-совместимый класс
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. Load environment variables
load_dotenv()

def main():
    # Используем твой ключ OpenRouter для всего
    api_key = os.getenv("OPENROUTER_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key or not pinecone_api_key:
        print("Error: API keys missing in secrets/.env")
        return

    # 2. Load .md files
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        return

    print("Loading documents...")
    loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    # 3. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks.")

    # 4. Initialize OpenRouter Embeddings
    # Мы используем класс OpenAIEmbeddings, но перенаправляем его на OpenRouter
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    # 5. Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "seo-analysis"
    dimension = 1536 # Размерность для text-embedding-3-small

    # 6. Smart Index Check & Recreate
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        desc = pc.describe_index(index_name)
        if desc.dimension != dimension:
            print(f"Dimension mismatch ({desc.dimension} vs {dimension}). Recreating index...")
            pc.delete_index(index_name)
            time.sleep(2) # Пауза для очистки
            existing_indexes.remove(index_name)

    if index_name not in existing_indexes:
        print(f"Creating new index '{index_name}' (dim: {dimension})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index is ready.")

    # 7. Upload
    print("Uploading to Pinecone via OpenRouter...")
    try:
        PineconeVectorStore.from_documents(
            all_splits,
            embeddings,
            index_name=index_name
        )
        print("Success!")
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == "__main__":
    main()
