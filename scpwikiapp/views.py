import os
import uuid
from dotenv import load_dotenv

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required

from .models import ChatSession, ChatMessage, PDFDocument
from .forms import SignUpForm, CustomLoginForm

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent
from langchain.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone
import openai

load_dotenv()

pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='us-east-1')  # Use your Pinecone environment
index_name = "scpragapp"
pinecone_index = pinecone.Index(index_name)

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username,password=password)
            login(request,user)
    else:
        form = SignUpForm()
    return render()

def login(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.post)
        if form.is_valid():
            username = form.cleaned_data.get(username)
            password = form.cleaned_data.get(password)
            user = authenticate(username=username,password=password)
            if user is not None:
                login(request,user)
                return()
    else:
        form = CustomLoginForm()
    return()

#ez
#this is for starting new chat. it is linked with the function below it by foreign key
def start_chat_session(request):
    session_id = str(uuid.uuid4())
    ChatSession.objects.create(session_id=session_id,user=request.user.username)
    return redirect('chat', session_id=session_id)
#ez
#make this happen every time they click the button to get their question answered and saves it to the chat session above
def handle_chat(request,session_id):
    chat_session = get_object_or_404(ChatSession,session_id=session_id)

    if request.method == 'POST':
        user_message = request.POST.get('message','')
        if user_message:
            usermessage = ChatMessage.objects.create(session=chat_session,sender='user',message=user_message)
            response = ragapp(user_message)
            botmessage = ChatMessage.objects.create(session=chat_session,sender='bot', message=response)
            return redirect('chat',session_id=session_id)
    return redirect('chat',session_id=session_id)


class OpenAIChatLLM:
    def __init__(self, api_key, model="gpt-4"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def call(self, prompt, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert on the game SCP: Secret Laboratory."},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        return response['choices'][0]['message']['content']


def ragapp(question):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = 'us-east-1'
    index_name = "scpragapp"

    def get_relevant_documents(query):
        # Initialize Pinecone and embeddings
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        pinecone_index = pinecone.Index(index_name)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Pinecone(pinecone_index, embeddings)

        # Embed the query
        query_embedding = embeddings.embed_query(query)

        # Perform the similarity search
        results = vectorstore.similarity_search(query_embedding, k=5)  # Adjust k as needed

        # Extract content from the results
        return [result.metadata['content'] for result in results]

    info_prompt_template = PromptTemplate.from_template("""
    You are an expert on the game SCP: Secret Laboratory. Answer the following question based on the game:

    Question: {question}
    Answer:
    """)

    llm = OpenAIChatLLM(api_key=openai_api_key)

    memory = ConversationBufferMemory()
    tools = [
        Tool(
            name="InformationProvider",
            func=lambda query: get_relevant_documents(query),
            description="Use this tool to retrieve detailed information about SCP: Secret Laboratory."
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm.call,  # Use the call method of our LLM wrapper
        agent_type="zero-shot-react-description",
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=50
    )

    response = agent(question)
    return response
