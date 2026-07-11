from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/form")
async def submit_form(
        username: str = Form( ... ),
        age: int = Form(None),
        desc: str = Form("", max_length=200)
                    ):
    return {
    "username": username,
    "age": age,
    "desc": desc
    }
