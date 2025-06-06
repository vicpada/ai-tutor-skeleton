# Standard Library Imports
import logging
import os

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import gradio as gr
from huggingface_hub import snapshot_download

# LlamaIndex (Formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.postprocessor.cohere_rerank import CohereRerank

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPT_SYSTEM_MESSAGE = """You are an AI assistant and expert instructor responding to technical questions from software architects and developers who are working in enterprise software architecture. 
These users are particularly focused on Microsoft technologies and Azure cloud services. Topics they are exploring include architecture patterns in Azure (serverless, microservices, event-driven systems), Azure services comparison (Functions, App Service, AKS, Logic Apps, etc.), DevOps practices (IaC with Bicep/Terraform, CI/CD with Azure DevOps or GitHub Actions), observability with Application Insights, secure design using Key Vault, identity management with Azure AD and B2C.
You should treat each question as part of this context. Your responses should be complete, accurate, and educational — suitable for technical professionals with intermediate to advanced knowledge in cloud architecture and AI application development. 
To find relevant information for answering questions, always use the "Azure_AI_Knowledge" tool. This tool returns technical documentation, architecture guides, official examples, and troubleshooting data focused on Azure and AI integration.
Only part of the tool's output may be relevant to the question — discard the irrelevant sections. Your answer should rely **exclusively** on the content provided by the tool. Do **not** inject external or speculative knowledge. If the user refines their question or focuses on a specific sub-topic, reformulate the tool query to capture that specificity and retrieve deeper information.
If a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Structure your answers in clear sections with multiple paragraphs if needed. If code is returned, include full code blocks in your response (formatted in Markdown) so the user can copy and run them directly.
If the tool doesn't return relevant content, inform the user clearly that the topic exceeds the current knowledge base and mention that no relevant documentation was found via the tool.
Always close your answers by inviting the user to ask follow-up or deeper questions related to the topic.
"""

QA_TEMPLATE = "Answer questions about Azure using 'Azure_AI_Knowledge' tool"


def download_knowledge_base_if_not_exists():
    """Download the knowledge base from the Hugging Face Hub if it doesn't exist locally"""
    if not os.path.exists("data/azure-architect"):
        os.makedirs("data/azure-architect")

        logging.warning(
            f"Vector database does not exist at 'data/', downloading from Hugging Face Hub..."
        )
        snapshot_download(
            repo_id="vicpada/AzureArchitectKnowledge",
            local_dir="data/azure-architect",            
            repo_type="dataset",
        )
        logging.info(f"Downloaded vector database to 'data/azure-architect'")


def get_tools(db_collection="azure-architect"):    
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    
    logging.info(f"Vector store initialized with {chroma_collection.count()} documents.")
    
    # Create the vector store index
    logging.info("Creating vector store index...")
    
    # Use the vector store to create an index

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )

    logging.info("Creating vector retriever...")
    
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=400,
        embed_model=Settings.embed_model,
        use_async=True,
        verbose=True,
    )
    

    cohere_rerank3 = CohereRerank(top_n=5, model = 'rerank-english-v3.0')

    logging.info("Creating tool...")
    
    tools = [
        RetrieverTool(
            retriever=vector_retriever,
            metadata=ToolMetadata(
                name="Azure_AI_Knowledge",
                description="Useful for info related to Azure and microsoft. Best practices, architecture, and other related resources."                
            ),
            node_postprocessors=[cohere_rerank3],
        )
    ]
    return tools


def generate_completion(query, history, memory):
    logging.info(f"User query: {query}")

    # Manage memory
    chat_list = memory.get()
    if len(chat_list) != 0:
        user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
        if len(user_index) > len(history):
            user_index_to_remove = user_index[len(history)]
            chat_list = chat_list[:user_index_to_remove]
            memory.set(chat_list)
    logging.info(f"chat_history: {len(memory.get())} {memory.get()}")
    logging.info(f"gradio_history: {len(history)} {history}")

    # Create agent
    tools = get_tools(db_collection="azure-architect")   


    agent = OpenAIAgent.from_tools(
        llm=Settings.llm,        
        memory=memory,
        tools=tools,
        system_prompt=QA_TEMPLATE
    )

    # Generate answer
    completion = agent.stream_chat(query)
    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str 

    logging.info(f"Source count: {len(completion.sources)}")
    logging.info(f"Sources: {completion.sources}")    

def launch_ui():
    with gr.Blocks(
        fill_height=True,
        title="AI Azure Architect 🤖",
        analytics_enabled=True,
    ) as demo:

        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(
                token_limit=120000,
            )
        )
        chatbot = gr.Chatbot(
            scale=1,
            placeholder="<strong>Azure AI Architect 🤖: A Question-Answering Bot for anything Azure related</strong><br>",
            show_label=False,
            show_copy_button=True,
        )

        gr.ChatInterface(
            fn=generate_completion,
            chatbot=chatbot,
            additional_inputs=[memory_state],
        )

        demo.queue(default_concurrency_limit=64)
        demo.launch(debug=True, share=False) # Set share=True to share the app online


if __name__ == "__main__":
    # Download the knowledge base if it doesn't exist
    download_knowledge_base_if_not_exists()

    # Set up llm and embedding model
    Settings.llm = OpenAI(temperature=1, model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # launch the UI
    launch_ui()