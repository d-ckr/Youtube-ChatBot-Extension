from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
import getpass
import os
# from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Only use the context provided to answer the question and if context is not sufficient, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """
)


def video_id_from_url(url):
    # simple extract
    import re
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None


def query(video_url, question):

    video_id = video_id_from_url(video_url)
    if not video_id:
        return {"error": "Invalid video URL"}
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except NoTranscriptFound:
        return {"error": "No transcript found for this video"}
    
    # vector_store = Chroma(
    #     collection_name="example_collection",
    #     embedding_function=embeddings,
    #     persist_directory=os.getenv("PERSIST_DIRECTORY", "./chroma_db")
    # )

    full_text = " ".join([item['text'] for item in transcript])

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # outputParser = lambda x: x  # Identity function for now            

    return {"message": "Ingestion started for video", "video_id": video_id}


if __name__ == "__main__":
    print(query("VIDEOID_HERE", "What are the main three takeaways about X?"))

