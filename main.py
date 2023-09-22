import json
from typing import Generator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import logging
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse, JSONResponse

from tgi_adapter import TGIAdapter, OpenAIRequest

app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)

# CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variable for API key
API_KEY = os.environ.get("OPENAI_API_KEY")
TGI_URL = os.environ.get("TGI_URL")

security = HTTPBearer()


def get_current_api_key(authorization: HTTPAuthorizationCredentials = Depends(security)):
    token = authorization.credentials
    if token != API_KEY:
        logging.error("Invalid API Key attempted.")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return token


adapter = TGIAdapter(TGI_URL)


def stream_data(request: OpenAIRequest) -> Generator[str, None, None]:
    for chunk in adapter.process_request_stream(request):
        # Convert the chunk (which is a dict) to a JSON string
        # and format it as an SSE event
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", description="This endpoint mimics OpenAI's chat completions API.")
async def generate_text(request: OpenAIRequest, api_key: str = Depends(get_current_api_key)):
    try:
        if request.stream:
            return StreamingResponse(stream_data(request), media_type="text/event-stream")
        else:
            response_data = adapter.process_request(request)
            return JSONResponse(content=response_data, status_code=200, media_type="application/json")

    except ValueError as ve:
        logging.error(f"Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in /v1/chat/completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Custom OpenAPI docs
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="OpenAI-like API",
        version="1.0.0",
        description="This is an OpenAI-like API using TGI backend",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
