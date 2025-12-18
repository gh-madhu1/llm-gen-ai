"""
Observability Module for LLM Agents
Provides structured logging, metrics, and tracing capabilities.
"""

import json
import logging
import time
import uuid
import contextlib
from typing import Dict, Any, Optional
from datetime import datetime
from threading import local

# Thread-local storage for context tracking (trace_id, etc.)
_context = local()

class AgentLogger:
    """
    Structured logger for AI Agents with context tracking.
    """
    
    def __init__(self, name: str = "llm_agent", log_file: str = "agent_activity.jsonl"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if re-initialized
        if not self.logger.handlers:
            # File handler for JSONL logs
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.JsonFormatter())
            self.logger.addHandler(file_handler)
            
            # Console handler for simplified viewing
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(console_handler)

    class JsonFormatter(logging.Formatter):
        """Format logs as JSON lines."""
        def format(self, record):
            log_obj = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "trace_id": getattr(_context, 'trace_id', None)
            }
            if hasattr(record, 'extra_data'):
                log_obj.update(record.extra_data)
            return json.dumps(log_obj)

    @contextlib.contextmanager
    def trace_context(self, trace_id: Optional[str] = None):
        """Context manager to set a trace ID for a block of operations."""
        token = getattr(_context, 'trace_id', None)
        _context.trace_id = trace_id or str(uuid.uuid4())
        try:
            yield _context.trace_id
        finally:
            _context.trace_id = token

    def info(self, message: str, **kwargs):
        """Log info with optional structured data."""
        self.logger.info(message, extra={'extra_data': kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning with optional structured data."""
        self.logger.warning(message, extra={'extra_data': kwargs})

    def error(self, message: str, **kwargs):
        """Log error with optional structured data."""
        self.logger.error(message, extra={'extra_data': kwargs})
        
    def log_decision(self, stage: str, decision: str, reason: str, **details):
        """Specific method for logging agent decisions."""
        self.info(
            f"Decision at {stage}: {decision}",
            event_type="agent_decision",
            stage=stage,
            decision=decision,
            reason=reason,
            details=details
        )

    def log_metrics(self, metric_name: str, value: float, unit: str, **tags):
        """Log a numerical metric."""
        self.info(
            f"Metric {metric_name}: {value} {unit}",
            event_type="metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags
        )

    @contextlib.contextmanager
    def track_latency(self, operation_name: str):
        """Context manager to track latency of an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log_metrics(
                f"{operation_name}_latency", 
                duration, 
                "seconds"
            )
