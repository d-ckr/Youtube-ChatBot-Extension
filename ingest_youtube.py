import os, json, math
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings   # or HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pytube import YouTube   # optional fallback for audio download
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

def video_id_from_url(url):
    # simple extract
    import re
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None

def get_captions(video_id):
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        # tr: list of {"text": "...", "start": 12.34, "duration": 3.21}
        return tr
    except NoTranscriptFound:
        return None

def captions_to_documents(captions):
    docs = []
    for seg in captions:
        text = seg['text'].strip()
        start = float(seg['start'])
        end = start + float(seg.get('duration', 0.0))
        if text:
            docs.append({'text': text, 'start': start, 'end': end})
    return docs

def merge_segments_to_chunks(segments, chunk_size_chars=1000, overlap_chars=200):
    # create combined text with timestamps retained per chunk
    texts = []
    buffer = ""
    buffer_st = None
    buffer_en = None
    for s in segments:
        piece = s['text'] + " "
        if buffer == "":
            buffer_st = s['start']
        buffer += piece
        buffer_en = s['end']
        if len(buffer) >= chunk_size_chars:
            texts.append({'text': buffer.strip(), 'start': buffer_st, 'end': buffer_en})
            # keep overlap: take last overlap_chars
            buffer = buffer[-overlap_chars:]
            buffer_st = None
    if buffer.strip():
        texts.append({'text': buffer.strip(), 'start': buffer_st or 0, 'end': buffer_en or 0})
    return texts

def ingest(video_url, persist_dir='chroma_data'):
    vid = video_id_from_url(video_url)
    if not vid:
        raise ValueError("bad url")
    captions = get_captions(vid)
    if captions is None:
        raise RuntimeError("No captions found — fallback to audio+ASR (not implemented in this snippet).")
    segments = captions_to_documents(captions)
    chunks = merge_segments_to_chunks(segments, chunk_size_chars=1200, overlap_chars=200)

    texts = [c['text'] for c in chunks]
    metadatas = [{'video_id': vid, 'start': c['start'], 'end': c['end']} for c in chunks]

    # Embeddings (OpenAI as example)
    embed = OpenAIEmbeddings()  # make sure OPENAI_API_KEY is set
    vectordb = Chroma.from_texts(texts=texts, embedding=embed, metadatas=metadatas, collection_name=f"yt_{vid}", persist_directory=persist_dir)
    vectordb.persist()
    print(f"Ingested {len(texts)} chunks for video {vid}")

def ingestYoutube(video_url):

    ingest(video_url=video_url)
    
