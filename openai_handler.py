from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Cargar variables de entorno
load_dotenv()

# Configurar el cliente de OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configurar el modelo de lenguaje
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Puedes cambiar a gpt-4 si lo prefieres
    temperature=0.3,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Template del prompt similar al original pero adaptado para OpenAI
raw_prompt = PromptTemplate.from_template("""
    Eres un asistente técnico que proporciona respuestas basadas únicamente en la información proporcionada. 
    Si no conoces la respuesta con la información proporcionada, sé honesto y responde: 
    'Lo siento, no puedo ayudarte con temas que no estén relacionados a información con la que fui entrenado'. 
    Todas las respuestas deben ser en español.

    Pregunta: {input}
    Contexto: {context}
    Respuesta:
""")

def ask_pdf_with_openai(query: str, folder_path: str, embedding_function) -> dict:
    """
    Función para consultar PDFs usando OpenAI como modelo de lenguaje
    
    Args:
        query (str): La pregunta del usuario
        folder_path (str): Ruta donde está almacenada la base de datos vectorial
        embedding_function: Función de embedding a utilizar
    
    Returns:
        dict: Diccionario con la respuesta del modelo
    """
    try:
        # Cargar la base de datos vectorial
        vector_store = Chroma(
            persist_directory=folder_path,
            embedding_function=embedding_function
        )
        
        # Configurar el recuperador de documentos
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.1
            }
        )
        
        # Crear la cadena de documentos
        document_chain = create_stuff_documents_chain(llm, raw_prompt)
        
        # Crear la cadena de recuperación
        chain = create_retrieval_chain(retriever, document_chain)
        
        # Obtener la respuesta
        result = chain.invoke({"input": query})
        
        return {"respuesta": result['answer'].strip()}
    
    except Exception as e:
        return {"error": f"Error al procesar la consulta: {str(e)}"}