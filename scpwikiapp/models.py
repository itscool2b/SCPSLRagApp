from django.db import models
from django.contrib.auth.models import User
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import os

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
        loader = PDFLoader(file_path=file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='us-east-1')
        index_name = "scpragapp"
        pinecone_index = pinecone.Index(index_name)

        openai_api_key = os.getenv('OPENAI_API_KEY')
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        Pinecone.from_documents(documents, embeddings, index_name=index_name)
        
        self.content = "\n".join(doc.page_content for doc in documents)
        super().save(*args, **kwargs)
