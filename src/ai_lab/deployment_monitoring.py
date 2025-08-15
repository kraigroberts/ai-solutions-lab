"""Deployment and monitoring system for AI Solutions Lab with Docker support and observability."""

import time
import json
import logging
import psutil
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
import subprocess
import docker
from contextlib import contextmanager

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int

@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: float
    active_users: int
    total_requests: int
    requests_per_minute: float
    average_response_time: float
    error_rate: float
    cache_hit_rate: float
    search_queries: int
    rag_queries: int

@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time: float
    error_message: Optional[str] = None
    last_check: float = None

class DeploymentManager:
    """Manages application deployment and configuration."""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            print(f"Docker not available: {e}")
            self.docker_available = False
        
        # Configuration
        self.config = self._load_config()
        
        # Deployment status
        self.deployment_status = "not_deployed"
        self.last_deployment = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        config_file = self.config_dir / "deployment.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load deployment config: {e}")
        
        # Default configuration
        return {
            'app_name': 'ai-solutions-lab',
            'version': '1.0.0',
            'port': 8000,
            'host': '0.0.0.0',
            'workers': 4,
            'environment': 'development',
            'docker': {
                'enabled': True,
                'image_name': 'ai-solutions-lab',
                'container_name': 'ai-solutions-lab-app',
                'ports': {'8000/tcp': 8000},
                'volumes': ['./data:/app/data'],
                'environment': {
                    'ENVIRONMENT': 'production',
                    'LOG_LEVEL': 'INFO'
                }
            },
            'monitoring': {
                'enabled': True,
                'metrics_interval': 60,
                'health_check_interval': 30,
                'log_retention_days': 30
            }
        }
    
    def save_config(self):
        """Save deployment configuration."""
        config_file = self.config_dir / "deployment.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save deployment config: {e}")
    
    def build_docker_image(self, tag: str = "latest") -> bool:
        """Build Docker image for the application."""
        if not self.docker_available:
            print("Docker not available")
            return False
        
        try:
            print(f"Building Docker image: {self.config['docker']['image_name']}:{tag}")
            
            # Build context is the project root
            build_context = Path.cwd()
            dockerfile_path = build_context / "Dockerfile"
            
            if not dockerfile_path.exists():
                print("Dockerfile not found")
                return False
            
            # Build the image
            image, build_logs = self.docker_client.images.build(
                path=str(build_context),
                tag=f"{self.config['docker']['image_name']}:{tag}",
                dockerfile="Dockerfile",
                rm=True
            )
            
            print(f"Docker image built successfully: {image.tags}")
            return True
            
        except Exception as e:
            print(f"Error building Docker image: {e}")
            return False
    
    def deploy_docker(self, tag: str = "latest") -> bool:
        """Deploy application using Docker."""
        if not self.docker_available:
            print("Docker not available")
            return False
        
        try:
            # Stop existing container
            self.stop_docker_container()
            
            # Pull or build image
            image_name = f"{self.config['docker']['image_name']}:{tag}"
            try:
                image = self.docker_client.images.get(image_name)
            except docker.errors.ImageNotFound:
                print(f"Image {image_name} not found, building...")
                if not self.build_docker_image(tag):
                    return False
                image = self.docker_client.images.get(image_name)
            
            # Create and start container
            container = self.docker_client.containers.run(
                image,
                name=self.config['docker']['container_name'],
                ports=self.config['docker']['ports'],
                volumes=self.config['docker']['volumes'],
                environment=self.config['docker']['environment'],
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            print(f"Container started: {container.id}")
            self.deployment_status = "deployed"
            self.last_deployment = time.time()
            
            return True
            
        except Exception as e:
            print(f"Error deploying Docker container: {e}")
            return False
    
    def stop_docker_container(self) -> bool:
        """Stop Docker container."""
        if not self.docker_available:
            return False
        
        try:
            container = self.docker_client.containers.get(self.config['docker']['container_name'])
            container.stop(timeout=30)
            container.remove()
            print("Docker container stopped and removed")
            return True
        except docker.errors.NotFound:
            print("No container to stop")
            return True
        except Exception as e:
            print(f"Error stopping container: {e}")
            return False
    
    def get_docker_status(self) -> Dict[str, Any]:
        """Get Docker deployment status."""
        if not self.docker_available:
            return {"status": "docker_not_available"}
        
        try:
            container = self.docker_client.containers.get(self.config['docker']['container_name'])
            
            return {
                "status": "running",
                "container_id": container.id,
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "created": container.attrs['Created'],
                "ports": container.attrs['NetworkSettings']['Ports'],
                "state": container.attrs['State']['Status'],
                "health": container.attrs['State'].get('Health', {}).get('Status', 'unknown')
            }
        except docker.errors.NotFound:
            return {"status": "not_running"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def create_dockerfile(self) -> bool:
        """Create a Dockerfile for the application."""
        dockerfile_content = """# AI Solutions Lab Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/data/cache /app/data/users /app/data/analytics

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.ai_lab.main_app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        try:
            dockerfile_path = Path.cwd() / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            print("Dockerfile created successfully")
            return True
        except Exception as e:
            print(f"Error creating Dockerfile: {e}")
            return False
    
    def create_docker_compose(self) -> bool:
        """Create docker-compose.yml for the application."""
        compose_content = """version: '3.8'

services:
  ai-solutions-lab:
    build: .
    container_name: ai-solutions-lab-app
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add Redis for caching
  redis:
    image: redis:7-alpine
    container_name: ai-solutions-lab-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
"""
        
        try:
            compose_path = Path.cwd() / "docker-compose.yml"
            with open(compose_path, 'w') as f:
                f.write(compose_content)
            print("docker-compose.yml created successfully")
            return True
        except Exception as e:
            print(f"Error creating docker-compose.yml: {e}")
            return False


class MonitoringSystem:
    """Comprehensive monitoring and observability system."""
    
    def __init__(self, data_dir: str = "./data/monitoring"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.system_metrics: List[SystemMetrics] = []
        self.application_metrics: List[ApplicationMetrics] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring configuration
        self.metrics_interval = 60  # seconds
        self.health_check_interval = 30  # seconds
        self.log_retention_days = 30
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize health checks
        self._initialize_health_checks()
    
    def _setup_logging(self):
        """Setup structured logging."""
        log_file = self.data_dir / "application.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ai-solutions-lab')
    
    def _initialize_health_checks(self):
        """Initialize health check components."""
        self.health_checks = {
            'database': HealthCheck('database', 'unknown', 0.0),
            'cache': HealthCheck('cache', 'unknown', 0.0),
            'search_engine': HealthCheck('search_engine', 'unknown', 0.0),
            'llm_providers': HealthCheck('llm_providers', 'unknown', 0.0),
            'api_endpoints': HealthCheck('api_endpoints', 'unknown', 0.0)
        }
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_metrics = time.time()
        last_health_check = time.time()
        
        while self.monitoring_active:
            current_time = time.time()
            
            # Collect system metrics
            if current_time - last_metrics >= self.metrics_interval:
                self._collect_system_metrics()
                last_metrics = current_time
            
            # Perform health checks
            if current_time - last_health_check >= self.health_check_interval:
                self._perform_health_checks()
                last_health_check = current_time
            
            time.sleep(1)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            # Active connections
            connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_total_mb=memory.total / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_total_gb=disk.total / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=connections
            )
            
            self.system_metrics.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
            
            self.logger.debug(f"System metrics collected: CPU {cpu_percent}%, Memory {memory.percent}%")
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _perform_health_checks(self):
        """Perform health checks on system components."""
        try:
            # Database health check
            self._check_database_health()
            
            # Cache health check
            self._check_cache_health()
            
            # Search engine health check
            self._check_search_engine_health()
            
            # LLM providers health check
            self._check_llm_providers_health()
            
            # API endpoints health check
            self._check_api_endpoints_health()
            
        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")
    
    def _check_database_health(self):
        """Check database health."""
        start_time = time.time()
        try:
            # Check if data directory is accessible
            data_dir = Path("./data")
            if data_dir.exists() and data_dir.is_dir():
                status = "healthy"
                error_message = None
            else:
                status = "unhealthy"
                error_message = "Data directory not accessible"
        except Exception as e:
            status = "unhealthy"
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        self.health_checks['database'] = HealthCheck(
            'database', status, response_time, error_message, time.time()
        )
    
    def _check_cache_health(self):
        """Check cache health."""
        start_time = time.time()
        try:
            # Check if cache directory exists and is writable
            cache_dir = Path("./data/cache")
            if cache_dir.exists() and os.access(cache_dir, os.W_OK):
                status = "healthy"
                error_message = None
            else:
                status = "unhealthy"
                error_message = "Cache directory not writable"
        except Exception as e:
            status = "unhealthy"
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        self.health_checks['cache'] = HealthCheck(
            'cache', status, response_time, error_message, time.time()
        )
    
    def _check_search_engine_health(self):
        """Check search engine health."""
        start_time = time.time()
        try:
            # Check if vector store files exist
            index_dir = Path("./data/index")
            if index_dir.exists():
                status = "healthy"
                error_message = None
            else:
                status = "degraded"
                error_message = "No search index found"
        except Exception as e:
            status = "unhealthy"
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        self.health_checks['search_engine'] = HealthCheck(
            'search_engine', status, response_time, error_message, time.time()
        )
    
    def _check_llm_providers_health(self):
        """Check LLM providers health."""
        start_time = time.time()
        try:
            # Check if LLM config exists
            llm_config = Path("./llm_config.json")
            if llm_config.exists():
                status = "healthy"
                error_message = None
            else:
                status = "degraded"
                error_message = "No LLM configuration found"
        except Exception as e:
            status = "unhealthy"
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        self.health_checks['llm_providers'] = HealthCheck(
            'llm_providers', status, response_time, error_message, time.time()
        )
    
    def _check_api_endpoints_health(self):
        """Check API endpoints health."""
        start_time = time.time()
        try:
            # Check if main app can be imported
            import importlib.util
            spec = importlib.util.spec_from_file_location("main_app", "./src/ai_lab/main_app.py")
            if spec and spec.loader:
                status = "healthy"
                error_message = None
            else:
                status = "unhealthy"
                error_message = "Main app module not found"
        except Exception as e:
            status = "unhealthy"
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        self.health_checks['api_endpoints'] = HealthCheck(
            'api_endpoints', status, response_time, error_message, time.time()
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_checks:
            return {"status": "unknown", "message": "No health checks available"}
        
        # Count health statuses
        status_counts = {}
        for check in self.health_checks.values():
            status_counts[check.status] = status_counts.get(check.status, 0) + 1
        
        # Determine overall status
        if status_counts.get('unhealthy', 0) > 0:
            overall_status = 'unhealthy'
        elif status_counts.get('degraded', 0) > 0:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        # Calculate average response time
        response_times = [check.response_time for check in self.health_checks.values()]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'status': overall_status,
            'timestamp': time.time(),
            'components': {
                name: {
                    'status': check.status,
                    'response_time': check.response_time,
                    'last_check': check.last_check,
                    'error_message': check.error_message
                }
                for name, check in self.health_checks.items()
            },
            'summary': {
                'total_components': len(self.health_checks),
                'healthy': status_counts.get('healthy', 0),
                'degraded': status_counts.get('degraded', 0),
                'unhealthy': status_counts.get('unhealthy', 0),
                'average_response_time': round(avg_response_time, 3)
            }
        }
    
    def get_metrics_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        # Filter recent metrics
        recent_system_metrics = [
            m for m in self.system_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_system_metrics:
            return {"message": "No metrics available for the specified time window"}
        
        # Calculate system metrics summary
        cpu_values = [m.cpu_percent for m in recent_system_metrics]
        memory_values = [m.memory_percent for m in recent_system_metrics]
        
        system_summary = {
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory': {
                'average': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'total_metrics': len(recent_system_metrics)
        }
        
        return {
            'time_window_hours': time_window_hours,
            'system_metrics': system_summary,
            'health_status': self.get_system_health(),
            'timestamp': time.time()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export monitoring metrics."""
        if format.lower() == "csv":
            return self._export_metrics_csv()
        else:
            return self._export_metrics_json()
    
    def _export_metrics_json(self) -> str:
        """Export metrics as JSON."""
        export_data = {
            'system_metrics': [asdict(m) for m in self.system_metrics[-1000:]],
            'health_checks': {
                name: asdict(check) for name, check in self.health_checks.items()
            },
            'export_timestamp': time.time()
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_metrics_csv(self) -> str:
        """Export metrics as CSV."""
        if not self.system_metrics:
            return "No metrics to export"
        
        # CSV header
        csv_lines = ["Timestamp,CPU_Percent,Memory_Percent,Memory_Used_MB,Memory_Total_MB,Disk_Usage_Percent"]
        
        # CSV data
        for metric in self.system_metrics[-1000:]:
            timestamp = datetime.fromtimestamp(metric.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            csv_lines.append(f'"{timestamp}",{metric.cpu_percent},{metric.memory_percent},{metric.memory_used_mb},{metric.memory_total_mb},{metric.disk_usage_percent}')
        
        return "\n".join(csv_lines)
    
    def cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy."""
        cutoff_time = time.time() - (self.log_retention_days * 24 * 3600)
        
        # Clean up system metrics
        old_metrics = [m for m in self.system_metrics if m.timestamp < cutoff_time]
        self.system_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        
        if old_metrics:
            self.logger.info(f"Cleaned up {len(old_metrics)} old system metrics")


def main():
    """Demo the deployment and monitoring system."""
    print("Deployment & Monitoring System Demo")
    print("=" * 40)
    
    # Test deployment manager
    print("Testing deployment manager...")
    deployment_manager = DeploymentManager()
    
    # Create Docker files
    deployment_manager.create_dockerfile()
    deployment_manager.create_docker_compose()
    
    # Show Docker status
    docker_status = deployment_manager.get_docker_status()
    print(f"Docker status: {docker_status}")
    
    # Test monitoring system
    print("\nTesting monitoring system...")
    monitoring = MonitoringSystem()
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Let it collect some metrics
    print("Collecting metrics for 10 seconds...")
    time.sleep(10)
    
    # Get health status
    health_status = monitoring.get_system_health()
    print(f"System health: {health_status['status']}")
    
    # Get metrics summary
    metrics_summary = monitoring.get_metrics_summary(1)  # Last hour
    print(f"Metrics summary: {metrics_summary}")
    
    # Stop monitoring
    monitoring.stop_monitoring()
    
    # Export metrics
    metrics_json = monitoring.export_metrics("json")
    print(f"Exported {len(metrics_json)} characters of metrics data")


if __name__ == "__main__":
    main()
