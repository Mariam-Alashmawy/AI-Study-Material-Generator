import io
import re
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Configuration
MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"
API_KEY = "secret123"

# Initialize Model & Embeddings
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Schema Setup
response_schemas = [
    ResponseSchema(name="quiz_title", description="A title for the study session"),
    ResponseSchema(name="questions", description="A list of 5 questions. Each must have 'question_text', 'options', and 'correct_answer'."),
    ResponseSchema(name="flashcards", description="A list of 5 flashcards with 'front' and 'back'.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

@app.post("/generate_study_material")
async def generate_study_material(file: UploadFile = File(...), authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    pdf_bytes = await file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw_text = "".join([page.extract_text() for page in reader.pages])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    relevant_docs = vector_db.similarity_search("key concepts and summaries", k=4)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are an expert tutor. Create a quiz and flashcards based on the text.
    The 'correct_answer' MUST be the EXACT text string from the options.
    
    {format_instructions}
    
    Text Content: "{context_text}"
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=2500, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON logic
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    json_text = json_match.group(1) if json_match else response
    return output_parser.parse(json_text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
