from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter


class DocumentProcessor:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def load_documents(self, path):
        loader = DirectoryLoader(path, glob=["**/*.pdf", "**/*.txt", "**/*.docx"])
        documents = loader.load()
        return documents

    def split_documents(self, documents):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    def get_retriever(self, texts):
        openai_embedding = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        vector_store = Chroma(
            collection_name="localRAG",
            embedding_function=openai_embedding,
        )

        vector_store.add_documents(documents=texts)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        return retriever


class RAGQueryProcessor:
    def __init__(self, openai_api_key, retriever):
        self.openai_api_key = openai_api_key
        self.retriever = retriever
        self.init()

    def init(self):
        llm = ChatOpenAI(openai_api_key=self.openai_api_key)

        contextualize_system_prompt = """
            Given a chat history and the latest user question
            which might reference context in the chat history,
            formulate a standalone question which can be understood
            without the chat history. Do NOT answer the question,
            just reformulate it if needed and otherwise return it as is.
            """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, self.retriever, contextualize_q_prompt
        )

        system_prompt = """
            You are a assistance for a context-based question-aswering task.
            The following are pieces of text as the context to answer user's questions.
            You sould answer questions SOLEY based on the provided context.
            If you can't find the answer to the question in the context, DO NOT asnwer it
            and say you don't know it. 
            \n\n
            {context}
            """
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        self.local_rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

    def query_LLM(self, question, chat_history):
        llm_response = self.local_rag_chain.invoke(
            {"input": question, "chat_history": chat_history}
        )
        return llm_response["answer"]
