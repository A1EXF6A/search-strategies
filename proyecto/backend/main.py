from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import MISSING_PARAMS_MESSAGE
from model import MaintenanceModel
from schemas import PredictionInput, PredictionResponse

model: MaintenanceModel | None = None


async def validation_exception_handler(
    request: Request, exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=500, content={"detail": "Error interno"})

    messages: list[str] = []

    for error in exc.errors():
        message: str = str(error.get("msg", "Entrada inválida"))

        if message.startswith("Value error, "):
            message = message[len("Value error, "):]

        messages.append(message)

    detail: str = "; ".join(messages) if messages else "Entrada inválida"

    return JSONResponse(status_code=422, content={"detail": detail})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        model = MaintenanceModel()
        print(f"Sistema difuso cargado (versión {model.version}).")
    except RuntimeError as error:
        model = None
        print(f"ERROR: {error}")

    yield


app = FastAPI(title="Mantenimiento preventivo con lógica difusa", lifespan=lifespan)

app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=MISSING_PARAMS_MESSAGE)

    return model.predict(payload)


@app.get("/api/memberships")
def memberships() -> dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail=MISSING_PARAMS_MESSAGE)

    return model.system.memberships_info()