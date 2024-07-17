from django.db import models
from django.contrib.auth.models import User
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
import os
from pdfminer.high_level import extract_text
from pinecone import Pinecone, ServerlessSpec

class ChatSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.session_id}"

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, related_name='messages', on_delete=models.CASCADE)
    sender = models.CharField(max_length=255)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.message[:50]}"

class PDFDocument(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='pdfs/')
    content = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        file_path = self.file.path
        text = extract_text(file_path)
        self.content = text

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents([{"page_content": text}])

        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = "scpragapp"

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name, 
                dimension=1536, 
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

        pinecone_index = pc.Index(index_name)

        openai_api_key = os.getenv('OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        Pinecone.from_documents(documents, embeddings, index_name=index_name)
        
        super().save(*args, **kwargs)
