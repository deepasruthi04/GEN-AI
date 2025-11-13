from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import os
import json
import secrets
from model import fetch_and_deduplicate, format_documents, normalize_documents, create_vector_db, persist_directory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(16))

os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

vectordb = None
try:
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Loaded existing vector DB.")
except Exception as e:
    print("No vector DB found:", e)


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    print("Method:", request.method)
    print("URL:", request.url)
    print("Headers:", request.headers)
    print("Query params:", request.query_params)
    print("Cookies:", request.cookies)
    print("Session:", request.session)

    if "history" not in request.session:
        request.session["history"] = []
    return templates.TemplateResponse("index.html", {"request": request, "history": request.session["history"]})


@app.post("/upload", response_class=HTMLResponse)
async def upload_data(request: Request, file: UploadFile = File(...)):
    global vectordb
    try:
        contents = await file.read()
        data = json.loads(contents)
    except Exception:
        return HTMLResponse("Invalid JSON file.", status_code=400)

    all_docs_dedup = fetch_and_deduplicate()
    formatted_docs = format_documents(all_docs_dedup)
    normalized_docs = normalize_documents(formatted_docs)
    vectordb = create_vector_db(normalized_docs, formatted_docs)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": request.session.get("history", []),
    })


@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    global vectordb
    if vectordb is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "history": [],
        })

    if "history" not in request.session:
        request.session["history"] = []

    docs = vectordb.similarity_search(user_input, k=10)
    context_text = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    system_prompt = (
        "You are an assistant for question-answering only for the context based on IoT data.\n"
        "Use ONLY the given context below to answer the question. "
        "If the question is unrelated to the context, respond with 'I don't know'.\n"
        "Keep the answer concise (up to 20 sentences).\n\n"
        f"CONTEXT:\n{context_text}"
    )

    messages = [{"role": "system", "content": system_prompt}] + request.session["history"]
    messages.append({"role": "user", "content": user_input})

    response = llm.invoke(messages)
    assistant_response = response.content if hasattr(response, "content") else str(response)

    request.session["history"].append({"role": "user", "content": user_input})
    request.session["history"].append({"role": "assistant", "content": assistant_response})
    request.session["history"] = request.session["history"][-20:]  # Keep last 20 messages

    print("Method:", request.method)
    print("URL:", request.url)
    print("Headers:", request.headers)
    print("Query params:", request.query_params)
    print("Cookies:", request.cookies)
    print("Session:", request.session)


    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": request.session["history"],
    })


@app.post("/reset")
async def reset_session(request: Request):
    request.session.clear()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": [],
    })


@app.on_event("startup")
async def check_vectordb():
    global vectordb
    if vectordb:
        data = vectordb.get()
        if data and "ids" in data:
            print("Number of documents in Chroma:", len(data["ids"]))
        else:
            print("Chroma DB is empty or could not be loaded.")
    else:
        print("Vector DB not initialized yet.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
