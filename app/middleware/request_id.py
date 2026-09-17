"""Request ID Middleware for HTTP request tracing and correlation headers.
"""

import uuid
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware attaching unique X-Request-ID headers to incoming requests and responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Retrieve X-Request-ID from request headers or generate a new UUID4 string
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.perf_counter()
        logger.info(f"Incoming Request [{request_id}] {request.method} {request.url.path}")

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            f"Completed Request [{request_id}] {request.method} {request.url.path} "
            f"Status: {response.status_code} Latency: {elapsed_ms:.2f}ms"
        )

        return response
