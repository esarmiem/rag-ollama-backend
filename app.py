from flask import Flask, request
import os
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from chromadb import Client
from chromadb.config import Settings
import shutil 
import time
from openai_handler import ask_pdf_with_openai

app= Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

folder_path ="db"

llm= Ollama(model="deepseek-r1:8b", temperature=0.3, num_predict=512) #el num_predict es el tamaño del prompt que se le pasa al modelo de LLM (conocido como max tokens)
embedding = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") #el modelo de embedding que agregue para los pdf en español
text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 1024, chunk_overlap=80, 
    length_function= len, is_separator_regex=False
)


raw_prompt= PromptTemplate.from_template("""
    <s>[INST] Eres un asistente técnico que proporciona respuestas basadas únicamente en la información proporcionada. Si no conoces la respuesta con la información proporcionada, sé honesto y responde: 'Lo siento, no puedo ayudarte con temas que no estén relacionados a información con la que fui entrenado'. Todas las respuestas deben ser en español. [/INST] </s>
    [INST] {input}
            Context: {context}
            Answer: 
    [/INST]
"""
)

# Iniciar la app
# metodo sencillo para probar el modelo
@app.route("/model", methods=["POST"])
def modelPost():
    print("POST /model usado")
    json_content = request.json
    query= json_content.get("query")
    print(f"query: {query}")
    response = llm.invoke(query)
    response_answer= {"respuesta": response}
    return response_answer

# Subir pdf
@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name= file.filename
    save_file= "pdf/"+ file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader= PDFPlumberLoader(save_file) # Cargar archivo subido
    docs = loader.load_and_split()  # Split docs
    print(f"docs len= {len(docs)}") 
    chunks = text_splitter.split_documents(docs) # Chunks de texto
    print(f"Chunks len= {len(chunks)}")
    # Base vectorial
    vector_store= Chroma.from_documents(
        documents= chunks,
        embedding= embedding,
        persist_directory= folder_path)
    vector_store.persist()

    response= {"status":"Subida correcta",
               "filename":file_name,
               "doc_len": len(docs),
               "chunks": len(chunks)}
    return response

# Preguntar al pdf
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("POST /ask_pdf usado")
    json_content = request.json
    query= json_content.get("query")
    print(f"query: {query}")
    
    print("Loading vector store")
    vector_store= Chroma(
        persist_directory=folder_path, embedding_function=embedding
    )
    # Creando el chain
    retriever = vector_store.as_retriever(
        search_type= "similarity_score_threshold",
        search_kwargs={
            "k":3,
            "score_threshold":0.1 # Limite de busqueda
        }
    )
        # Creando el chain
    document_chain= create_stuff_documents_chain(
        llm, raw_prompt
    )
    chain= create_retrieval_chain(retriever, document_chain)
    result= chain.invoke({"input":query})
    print(result)
    
    response_answer= {"respuesta": result['answer'].replace("<s>[INST]", "").replace("[/INST]</s>","").strip()}
    return response_answer

# Endpoint para preguntar al PDF usando OpenAI
@app.route("/ask_pdf_openai", methods=["POST"])
def askPDFOpenAI():
    print("POST /ask_pdf_openai usado")
    try:
        json_content = request.json
        query = json_content.get("query")
        print(f"query: {query}")
        
        response = ask_pdf_with_openai(query, folder_path, embedding)
        return response
    
    except Exception as e:
        return {"error": f"Error en el endpoint: {str(e)}"}, 500

# Resetear la base de datos vectorial y limpiar el path db (ChromaDB)
@app.route("/reset_db", methods=["POST"])
def reset_db():
    print("POST /reset_db usado")
    try:
        client = Client(Settings(allow_reset=True))
        client.reset()
        
        # Limpieza completa del directorio db
        db_path = os.path.abspath(folder_path)
        if os.path.exists(db_path):
            for filename in os.listdir(db_path):
                file_path = os.path.join(db_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Error al eliminar {file_path}. Razón: {e}')
        
        # Limpieza de la carpeta pdf
        pdf_path = os.path.abspath("pdf")
        if os.path.exists(pdf_path):
            for filename in os.listdir(pdf_path):
                file_path = os.path.join(pdf_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Error al eliminar PDF {file_path}. Razón: {e}')

        return {"status": "Base de datos y archivos PDF limpiados completamente"}
    except Exception as e:
        return {"status": f"Error al resetear: {str(e)}"}, 500


# Verificar el estado de la base de datos
@app.route("/db_status", methods=["GET"])
def db_status():
    try:
        client = Client(Settings(allow_reset=True))
        collections = client.list_collections()
        db_path = os.path.abspath(folder_path)
        db_exists = os.path.exists(db_path) and os.listdir(db_path)
        
        return {
            "status": "success",
            "collections_count": len(collections),
            "db_path": db_path,
            "db_exists": db_exists,
            "db_empty": not db_exists or not os.listdir(db_path)
        }
    except Exception as e:
        return {"status": f"Error checking DB status: {str(e)}"}, 500

# Listar archivos PDF
@app.route("/list_pdfs", methods=["GET"])
def list_pdfs():
    try:
        pdf_path = os.path.abspath("pdf")
        if not os.path.exists(pdf_path):
            return {
                "status": "success",
                "file_count": 0,
                "files": []
            }

        files = []
        for filename in os.listdir(pdf_path):
            file_path = os.path.join(pdf_path, filename)
            if os.path.isfile(file_path):
                file_info = {
                    "name": filename,
                    "size_bytes": os.path.getsize(file_path),
                    "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                    "last_modified": os.path.getmtime(file_path),
                    "last_modified_human": time.ctime(os.path.getmtime(file_path))
                }
                files.append(file_info)

        return {
            "status": "success",
            "file_count": len(files),
            "files": files
        }
    except Exception as e:
        return {"status": f"Error listing PDFs: {str(e)}"}, 500

# Iniciar la app
# comando para iniciar la app: python .\app.py
def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ =="__main__":
    start_app()