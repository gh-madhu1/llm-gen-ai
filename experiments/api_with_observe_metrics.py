"""
required dependencies: 

pip install fastapi uvicorn prometheus-client opentelemetry-api \
opentelemetry-sdk opentelemetry-exporter-jaeger

"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
import logging
import json
import time
import random
import hashlib
from starlette.responses import Response

# ------------------ FastAPI ------------------
app = FastAPI(title="LLM Observability Demo")

# ------------------ Prometheus Metrics ------------------
REQUESTS = Counter(
    "genapp_requests_total",
    "Total requests",
    ["route", "model_version"]
)

ERRORS = Counter(
    "genapp_errors_total",
    "Total errors",
    ["route", "error_type"]
)

LATENCY = Histogram(
    "genapp_latency_seconds",
    "Latency of LLM calls",
    ["route", "model_version"]
)

TOKENS = Counter(
    "genapp_tokens_generated_total",
    "Total tokens generated",
    ["model_version"]
)

HALLUCINATION_RATE = Gauge(
    "genapp_hallucination_rate",
    "Hallucination rate",
    ["model_version"]
)

# ------------------ OpenTelemetry (Jaeger) ------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# ------------------ Logging ------------------
logger = logging.getLogger("genapp")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

# ------------------ Request Schema ------------------
class PromptRequest(BaseModel):
    user_id: str
    prompt: str

# ------------------ Helpers ------------------
def fake_llm(prompt: str) -> str:
    time.sleep(random.uniform(0.1, 0.4))
    return f"LLM Response to: {prompt}"

def token_count(text: str) -> int:
    return len(text.split())

def hallucination_check(text: str) -> bool:
    # Simulated hallucination detection
    return "fake_fact" in text.lower()

def hash_user(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()

# ------------------ Metrics Endpoint ------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ------------------ Main LLM Endpoint ------------------
@app.post("/generate")
async def generate(req: PromptRequest, request: Request):
    route = "/generate"
    model_version = "v1.0"

    REQUESTS.labels(route=route, model_version=model_version).inc()
    start_time = time.time()

    try:
        with tracer.start_as_current_span("user.request") as span:
            span.set_attribute("model", "demo-llm")
            span.set_attribute("model_version", model_version)
            span.set_attribute("prompt_length", len(req.prompt))

            # ---------- LLM call ----------
            with tracer.start_as_current_span("llm.call"):
                response = fake_llm(req.prompt)

            # ---------- Token metrics ----------
            out_tokens = token_count(response)
            TOKENS.labels(model_version=model_version).inc(out_tokens)

            # ---------- Quality check (sampled) ----------
            sampled = random.random() < 0.3
            hallucinated = False
            if sampled:
                hallucinated = hallucination_check(response)
                HALLUCINATION_RATE.labels(model_version=model_version).set(
                    1 if hallucinated else 0
                )

            # ---------- Structured log ----------
            log_data = {
                "timestamp": time.time(),
                "request_id": request.headers.get("x-request-id", "auto"),
                "user_hash": hash_user(req.user_id),
                "model": "demo-llm",
                "model_version": model_version,
                "prompt_length": len(req.prompt),
                "response_length": len(response),
                "latency_ms": int((time.time() - start_time) * 1000),
                "tokens_out": out_tokens,
                "sampled_for_quality": sampled,
                "hallucinated": hallucinated
            }
            logger.info(json.dumps(log_data))

            return {
                "response": response,
                "model_version": model_version
            }

    except Exception as e:
        ERRORS.labels(route=route, error_type=type(e).__name__).inc()
        raise e

    finally:
        LATENCY.labels(route=route, model_version=model_version).observe(
            time.time() - start_time
        )
