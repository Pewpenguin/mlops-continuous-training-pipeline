#!/usr/bin/env python
"""
Model API Server Runner

This script starts the FastAPI model serving application.
It provides a command-line interface to configure the API server.
"""

import os
import sys
import argparse
import logging
import socket
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_port_available(host, port):
    """Check if a port is available on the specified host.
    
    Args:
        host: Host address to check
        port: Port number to check
        
    Returns:
        bool: True if port is available, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def find_available_port(host, start_port, max_attempts=10):
    """Find an available port starting from the specified port.
    
    Args:
        host: Host address to check
        start_port: Starting port number
        max_attempts: Maximum number of ports to try
        
    Returns:
        int: Available port number, or None if no port is available
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    return None

def main():
    """Start the model API server."""
    parser = argparse.ArgumentParser(description="Model API Server")
    parser.add_argument('--host', default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--auto-port', action='store_true', help='Automatically find an available port if the specified port is in use')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Logging level')
    parser.add_argument('--mlflow-tracking-uri', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    # Set MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        logger.info(f"Set MLflow tracking URI to {args.mlflow_tracking_uri}")
    
    # Check if the specified port is available
    port = args.port
    if not is_port_available(args.host, port):
        if args.auto_port:
            # Try to find an available port
            new_port = find_available_port(args.host, port + 1)
            if new_port:
                logger.warning(f"Port {port} is already in use. Using port {new_port} instead.")
                port = new_port
            else:
                logger.error(f"Port {port} is already in use and no alternative ports are available.")
                sys.exit(1)
        else:
            logger.error(f"Port {port} is already in use. Use --auto-port to automatically find an available port.")
            sys.exit(1)
    
    # Start the API server
    import uvicorn
    logger.info(f"Starting model API server on {args.host}:{port}")
    uvicorn.run("api.app:app", host=args.host, port=port, reload=args.reload)

if __name__ == "__main__":
    main()