from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import answer_examples
from mongoDBClient import MongoDBChatMessageHistory
import textwrap
import os

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return MongoDBChatMessageHistory(session_id)

def get_retriever():
    embedding = UpstageEmbeddings(model="embedding-query")
    index_name = 'career-upstage-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_llm(model='solar-pro'):
    llm = ChatUpstage(model=model,api_key=os.getenv("UPSTAGE_API_KEY"))
    return llm

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "You are given a chat history and the user's most recent question. "
        "This question may refer to prior parts of the conversation (e.g., using words like 'it', 'that', or 'he/she')."

        "Your task is to rewrite the latest question so that it is fully self-contained and understandable "
        "without needing to refer to the earlier messages."

        "Important:"
        "- Do NOT answer the question."
        "- Only reformulate the question if it contains references to previous context."
        "- If it is already self-contained, return it as-is."

        "Formatting Instructions:"
        "- Insert natural line breaks ('\\n') in long sentences to improve readability."
        "- Line breaks should go after punctuation marks such as periods (.), commas (,), dashes (-), or colons (:), "
        "where appropriate."
        "- Make sure the reformulated question retains the original intent and meaning."

        "Return only the final, reformulated question with line breaks included where necessary."
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_dictionary_chain():
    dictionary = []
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        textwrap.dedent(f"""
            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
            그런 경우에는 질문만 리턴해주세요
            사전: {dictionary}
            질문: {{question}}
        """)
    )
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 노동 및 고용 분야의 전문가입니다. 사용자의 질문에 대해 노동 및 고용 관련 지식을 바탕으로 답변해주세요. "
        "아래에 제공된 문서를 참고하여 답변하되, 명확한 정보가 없다면 '모르겠습니다'라고 답변해주세요.\n"
        "답변을 시작할 때는 반드시 '출처에 따르면'이라는 표현으로 시작하고, 4~5문장 정도로 간결하고 명확하게 작성해주세요.\n"
        "답변에는 문장의 자연스러운 흐름을 유지하고, 의미가 불분명하거나 어색한 표현은 피해주세요.\n"
        "또한, 모든 한자는 제거하거나 가능한 경우 해당 한자를 한글로 바꾸어 작성해주세요.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain