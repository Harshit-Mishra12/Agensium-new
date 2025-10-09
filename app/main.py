from fastapi import FastAPI
from app import routes
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv() 
app = FastAPI(title="Data Agents Platform 🚀")
origins = [
    "http://localhost:3000",  # For Create React App
    "http://localhost:5173",  # For Vite
    "http://localhost:8080",  # Common development port
    "*"                      # Wildcard for easy development - BE CAREFUL IN PRODUCTION
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of origins that are allowed to make requests
    allow_credentials=True,    # Allows cookies to be included in requests
    allow_methods=["*"],         # Allows all standard HTTP methods
    allow_headers=["*"],         # Allows all headers
)
# include API routes
app.include_router(routes.router)

@app.get("/")
def root():
    return {"message": "Welcome to Data Agents Platform 🚀"}
