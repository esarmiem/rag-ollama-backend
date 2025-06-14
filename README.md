# RAG Application with Ollama and FastEmbed

Esta aplicación integra tecnologías para procesamiento de lenguaje natural y análisis de documentos. Utiliza el modelo de Ollama para generar respuestas técnicas basadas en información proporcionada, y el modelo de embeddings FastEmbed para procesar documentos en español. La aplicación permite cargar archivos PDF, dividir su contenido en fragmentos manejables, y luego buscar respuestas dentro de esos documentos utilizando una base de vectores creada con Chroma. Además, ofrece una interfaz simple para interactuar con el modelo, permitiendo a los usuarios hacer preguntas directamente o cargar documentos para análisis posterior.

---

## 🧠 Características clave

- **Integración de modelos de lenguaje**: Utiliza Ollama para generar respuestas técnicas y FastEmbed para embeddings en español.
- **Procesamiento de documentos PDF**: Capacidad para cargar, dividir y analizar documentos PDF.
- **Búsqueda de información**: Implementa una búsqueda por similitud de vectores para encontrar respuestas relevantes dentro de los documentos cargados.
- **Interfaz de usuario**: Ofrece rutas API para interactuar con el modelo y cargar documentos, facilitando su uso y desarrollo.
- **Gestión de base de datos vectorial**: Almacenamiento y gestión eficiente de embeddings con ChromaDB.

---

## 📋 Tabla de Contenidos

1. [Fase 1: Configuración de Ollama](#fase-1-configuración-de-ollama)
2. [Fase 2: Estructura de la aplicación](#fase-2-estructura-de-la-aplicación)
3. [Fase 3: Interacción con el LLM](#fase-3-interacción-con-el-llm)
4. [Fase 4: Integración de capacidad de subir PDF](#fase-4-integración-de-capacidad-de-subir-pdf)
5. [Fase 5: Agregando embeddings](#fase-5-agregando-embeddings)
6. [Fase 6: Respuestas del modelo](#fase-6-respuestas-del-modelo)
7. [Funcionalidades adicionales](#funcionalidades-adicionales)
8. [Ejecución y flujo recomendado](#ejecución-y-flujo-recomendado)

---

## Fase 1: Configuración de Ollama

### 1. Instalar Ollama
Descarga e instala Ollama desde [https://ollama.com](https://ollama.com/)

### 2. Descargar el modelo
Trae el modelo recomendado:

```bash
ollama pull deepseek-r1:8b
```

**Nota**: También puedes usar `llama3` si prefieres:
```bash
ollama pull llama3
```

### 3. Verificar modelos instalados
Confirma que el modelo se instaló correctamente:

```bash
ollama list
```

### 4. Iniciar servidor (opcional)
Puedes ejecutar Ollama como servidor:

```bash
ollama serve
```

### 5. Probar el modelo
Ejecuta el modelo directamente:

```bash
ollama run deepseek-r1:8b
```

### 6. Verificar funcionamiento con API
Prueba que el modelo funcione correctamente con una solicitud HTTP:

```bash
curl http://localhost:11434/api/chat -d '{
    "model": "deepseek-r1:8b",
    "messages": [
        { "role": "system", "content": "You are a service agent to schedule medical appointments" },
        { "role": "user", "content": "I need to schedule an appointment please" }
    ],
    "stream":false
  }'
```

---

## Fase 2: Estructura de la aplicación

### 7. Crear entorno virtual
Crea un ambiente aislado para trabajar:

```bash
python -m venv myenv
```

### 8. Activar entorno virtual
Activa las dependencias:

**Windows:**
```bash
.\myenv\Scripts\activate
```

**Linux/Mac:**
```bash
source myenv/bin/activate
```

### 8.1 Instalar dependencias
Instala las dependencias necesarias:

```bash
python -m pip install -q langchain_community flask langchain-text-splitters fastembed pdfplumber chromadb langchain
```

### 8.2 Alternativa con requirements.txt
Si tienes el archivo `requirements.txt` en la ruta principal de tu proyecto:

```bash
python -m pip install -r requirements.txt
```

### 9. Crear estructura básica
Crea el archivo `app.py` con el bosquejo inicial de la aplicación:

```python
from flask import Flask, request

app = Flask(__name__)

# Iniciar la app
@app.route("/model", methods=["POST"])
def modelPost():
    print("POST /model usado")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response_answer = "Respuesta de prueba"
    return response_answer

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
```

### 10. Probar la aplicación básica
- Abre Postman
- Configura una solicitud POST a `http://localhost:8080/model`
- En el body (JSON), envía: `{"query":"Hola como estas"}`
- Deberías recibir: `Respuesta de prueba`

---

## Fase 3: Interacción con el LLM

### 11. Integrar Ollama en la aplicación
Modifica `app.py` para que el modelo responda a las preguntas:

```python
from flask import Flask, request
from langchain_community.llms import Ollama

app = Flask(__name__)

# Configuración del modelo
llm = Ollama(
    model="deepseek-r1:8b",
    temperature=0.3,
    num_predict=512
)

# Endpoint para interactuar con el modelo
@app.route("/model", methods=["POST"])
def modelPost():
    print("POST /model usado")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = llm.invoke(query)
    response_answer = {"respuesta": response}
    return response_answer

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
```

### 12. Probar integración con LLM
Ahora puedes hacer cualquier pregunta y deberías recibir una respuesta generada por el modelo.

---

## Fase 4: Integración de capacidad de subir PDF

### 13. Crear directorio para PDFs
Crea un directorio llamado `pdf` para almacenar los archivos subidos.

### 14. Agregar endpoint para subir PDFs
Añade el nuevo método en la aplicación para procesar PDFs:

```python
@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")
    response = {"status": "Subida correcta", "filename": file_name}
    return response
```

### 15. Probar subida de PDFs
- Ve a Postman
- Configura una solicitud POST a `http://localhost:8080/pdf`
- En el Body, elige `form-data`
- Configura `Key: file` (tipo File) y sube tu archivo PDF
- Deberías recibir una respuesta como:

```json
{
    "filename": "documento.pdf",
    "status": "Subida correcta"
}
```

---

## Fase 5: Agregando embeddings

### 16. Importar dependencias para embeddings
Agrega las dependencias necesarias para trabajar con embeddings:

```python
from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

app = Flask(__name__)

# Configuración del modelo LLM
llm = Ollama(
    model="deepseek-r1:8b",
    temperature=0.3,
    num_predict=512
)

# Configuración de embeddings para español
embedding = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Configuración del divisor de texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=80, 
    length_function=len, 
    is_separator_regex=False
)
```

**Nota importante sobre embeddings**: Para documentos en español, es crucial usar un modelo de embeddings que lo soporte. El modelo `paraphrase-multilingual-MiniLM-L12-v2` funciona muy bien para español.

### 17. Configurar directorio de base de datos
Define la carpeta donde se almacenarán los embeddings:

```python
app = Flask(__name__)

folder_path = "db"  # Directorio para almacenar embeddings

# ... resto de la configuración
```

### 18. Actualizar el endpoint de PDF para generar embeddings
Modifica el método `pdfPost` para procesar y almacenar embeddings:

```python
@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    # Cargar y procesar el documento
    loader = PDFPlumberLoader(save_file)  # Cargar archivo subido
    docs = loader.load_and_split()  # Dividir documento
    print(f"docs len= {len(docs)}") 
    
    # Crear chunks de texto
    chunks = text_splitter.split_documents(docs)
    print(f"Chunks len= {len(chunks)}")
    
    # Crear base vectorial
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=folder_path
    )
    vector_store.persist()

    response = {
        "status": "Subida correcta",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks)
    }
    return response
```

### 19. Probar generación de embeddings
Vuelve a subir un PDF usando el endpoint `/pdf`. Ahora deberías recibir una respuesta más detallada:

```json
{
    "chunks": 34,
    "doc_len": 11,
    "filename": "documento.pdf",
    "status": "Subida correcta"
}
```

---

## Fase 6: Respuestas del modelo

### 20. Importar dependencias adicionales y crear prompt personalizado
Agrega las dependencias finales y define un prompt personalizado:

```python
from flask import Flask, request
import os
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

# Prompt personalizado para respuestas técnicas
raw_prompt = PromptTemplate.from_template("""
    <s>[INST] You are a technical assistant to provide answers based only on the provided information. If you dont know the answer with the provided information be honest and answer: 'Lo siento, no puedo ayudarte con temas que no esten relacionados a información con la que fui entrenado' [/INST] </s>
    [INST] {input}
            Context: {context}
            Answer: 
    [/INST]
""")
```

### 21. Crear endpoint para consultas sobre documentos
Agrega el método `ask_pdf` para responder preguntas basadas en los documentos:

```python
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("POST /ask_pdf usado")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
  
    print("Cargando vector store")
    vector_store = Chroma(
        persist_directory=folder_path, 
        embedding_function=embedding
    )
    
    # Crear el retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.1  # Límite de búsqueda
        }
    )

    # Crear las cadenas de procesamiento
    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
  
    response_answer = {"respuesta": result['answer']}
    return response_answer
```

### 22. Probar consultas sobre documentos
Realiza una consulta sobre el contenido de tus documentos:

**Solicitud POST a** `http://localhost:8080/ask_pdf`:
```json
{
    "query": "¿Qué es Docker?"
}
```

**Respuesta esperada**:
```json
{
    "respuesta": "Docker es una plataforma de contenedores que permite a los desarrolladores y equipos de IT crear, ejecutar y administrar aplicaciones en entornos aislados y portátiles. Docker utiliza una tecnología de contenedores llamada Linux Containers (LXC) para crear un entorno de ejecución aislado para cada aplicación.\n\nCon Docker, puedes crear imágenes de contenedor que incluyan todo lo necesario para ejecutar una aplicación, como el sistema operativo, bibliotecas y dependencias."
}
```

### 23. Ejemplo de pregunta fuera del contexto
**Solicitud**:
```json
{
    "query": "¿Cuántos jugadores tiene un equipo de fútbol?"
}
```

**Respuesta**:
```json
{
    "respuesta": "Lo siento, no puedo ayudarte con temas que no esten relacionados a información con la que fui entrenado"
}
```

---

## 🚀 Funcionalidades adicionales

La aplicación incluye endpoints adicionales para una gestión completa:

### Gestión de la base de datos

#### Resetear base de datos
- **Método**: `POST`
- **Ruta**: `/reset_db`
- **Descripción**: Elimina todos los documentos y embeddings almacenados, además de limpiar completamente la carpeta de PDFs

```python
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
```

#### Estado de la base de datos
- **Método**: `GET`
- **Ruta**: `/db_status`
- **Descripción**: Muestra información detallada sobre las colecciones y estado de la DB

```python
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
```

### Gestión de archivos PDF

#### Listar PDFs almacenados
- **Método**: `GET`
- **Ruta**: `/list_pdfs`
- **Descripción**: Devuelve lista detallada de PDFs con metadatos (tamaño, fecha modificación)

```python
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
```

---

## 📡 Resumen de Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/model` | Interacción directa con el modelo LLM |
| POST | `/pdf` | Subir y procesar documentos PDF |
| POST | `/ask_pdf` | Consultar información de documentos cargados |
| POST | `/reset_db` | Resetear base de datos vectorial |
| GET | `/db_status` | Estado de la base de datos |
| GET | `/list_pdfs` | Listar PDFs almacenados |

---

## 🚀 Ejecución y flujo recomendado

### Ejecutar la aplicación
Inicia la aplicación con:

```bash
python app.py
```

**Nota**: El código incluye el comentario: `# comando para iniciar la app: python .\app.py`

### Flujo completo recomendado

```bash
# 1. Crear entorno virtual (solo la primera vez)
python -m venv myenv

# 2. Activar entorno virtual
source myenv/bin/activate        # Linux/Mac
.\myenv\Scripts\activate         # Windows

# 3. Instalar dependencias (solo la primera vez o cuando cambien)
pip install -r requirements.txt

# 4. Levantar la aplicación
python app.py

# 5. Cuando termines, desactivar el entorno
deactivate
```

### Orden de uso recomendado

1. **Subir documentos PDF** mediante `/pdf`
2. **Realizar consultas** sobre los documentos con `/ask_pdf`
3. **Gestionar la base de datos** según necesidad con `/reset_db` y `/db_status`
4. **Listar documentos** disponibles con `/list_pdfs`
5. **Interacción directa** con el modelo usando `/model`

---

## ⚙️ Configuración avanzada

### Modelo LLM
- **Modelo**: `deepseek-r1:8b`
- **Temperature**: `0.3`
- **Max tokens**: `512`

### Modelo de Embeddings
- **Modelo**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Optimizado para**: Texto en español y multilingüe

### Configuración de chunks
- **Tamaño de chunk**: `1024` caracteres
- **Overlap**: `80` caracteres
- **Búsqueda**: Top 3 resultados con umbral de similitud 0.1

---

## 🌐 Acceso

La aplicación estará disponible en: **http://localhost:8080**

---

## 🔧 Solución de problemas

### Problemas comunes

1. **Error al instalar dependencias**: Asegúrate de tener Python 3.8+ instalado
2. **Ollama no responde**: Verifica que el servicio esté ejecutándose con `ollama serve`
3. **Embeddings lentos**: El primer uso descarga el modelo de embeddings
4. **PDFs no se procesan**: Verifica que el directorio `pdf/` exista
5. **Base de datos corrupta**: Usa `/reset_db` para reiniciar

### Logs y debugging

La aplicación incluye logs detallados para facilitar el debugging. Revisa la consola para información sobre:
- Carga de documentos
- Generación de embeddings
- Consultas realizadas
- Errores de procesamiento

---

## 👨‍💻 Autor

**By:** [Elder Sarmiento](https://www.linkedin.com/in/elder-sarmiento/)

---

¡Perfecto! Ahora tienes tu propio asistente que responde solo basándose en la información que le proporcionaste. 😉