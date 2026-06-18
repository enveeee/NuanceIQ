import time
import uuid
from loguru import logger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        logger.info(
            f"[{request_id}] → {request.method} {request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}"
        )

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"[{request_id}] ← {response.status_code} "
            f"({elapsed_ms:.1f}ms)"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response