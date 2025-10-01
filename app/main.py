from fastapi import FastAPI
from app import routes

app = FastAPI(title="Data Agents Platform 🚀")

# include API routes
app.include_router(routes.router)

@app.get("/")
def root():
    return {"message": "Welcome to Data Agents Platform 🚀"}
