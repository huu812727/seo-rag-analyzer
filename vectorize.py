import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. Load environment variables
load_dotenv()

def main():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("Error: PINECONE_API_KEY not found in .env file.")
        return

    # 2. Load .md files from data folder
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist. Run scraper.py first.")
        return

    print("Loading documents...")
    loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # 3. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(all_splits)} chunks.")

    # 4. Initialize HuggingFace embeddings
    print("Initializing embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # 6. Check and create index
    index_name = "seo-analysis"
    dimension = 384
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Waiting for index '{index_name}' to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index is ready.")
    else:
        print(f"Index '{index_name}' already exists.")

    # 7. Upload chunks to Pinecone
    print("Uploading chunks to Pinecone...")
    try:
        vector_store = PineconeVectorStore.from_documents(
            all_splits,
            embeddings,
            index_name=index_name
        )
        print("Successfully uploaded all chunks.")
    except Exception as e:
        print(f"Error during upload: {e}")

    # Final message
    print(f"\nVectorization complete. Total chunks created and uploaded: {len(all_splits)}")

if __name__ == "__main__":
    main()
