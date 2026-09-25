import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import MISSING_PARAMS_MESSAGE
from model import MaintenanceModel
from schemas import PredictionInput, PredictionResponse

model: MaintenanceModel | None = None


def _sanitize(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, dict):
        sanitized: dict[object, object] = {}

        for key, item in value.items():
            sanitized[key] = _sanitize(item)

        return sanitized

    if isinstance(value, list):
        sanitized_list: list[object] = []

        for item in value:
            sanitized_list.append(_sanitize(item))

        return sanitized_list

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return str(value)


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=500, content={"detail": "Error interno"})

    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize(exc.errors())},
    )


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