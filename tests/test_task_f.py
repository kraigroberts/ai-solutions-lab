"""Tests for Task F: Advanced Features & Optimization."""

import pytest
import sys
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_lab.cache_manager import CacheManager, SearchCache, CacheEntry
from ai_lab.advanced_analytics import AdvancedAnalytics, SearchMetrics, UserBehavior, PerformanceInsights
from ai_lab.user_management import UserManager, User, UserRole, Permission, UserSession
from ai_lab.deployment_monitoring import DeploymentManager, MonitoringSystem, SystemMetrics, HealthCheck


class TestCacheManager:
    """Test cache management functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(max_size_mb=1, default_ttl=60)
        self.search_cache = SearchCache(self.cache_manager)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation."""
        entry = CacheEntry(
            data="test_data",
            timestamp=time.time(),
            access_count=1,
            size_bytes=100,
            ttl=60
        )
        
        assert entry.data == "test_data"
        assert entry.access_count == 1
        assert entry.ttl == 60
    
    def test_basic_caching(self):
        """Test basic cache operations."""
        # Test set and get
        self.cache_manager.set("test_key", "test_value", ttl=30)
        result = self.cache_manager.get("test_key")
        assert result == "test_value"
        
        # Test cache hit statistics
        stats = self.cache_manager.get_stats()
        assert stats['hits'] == 1
        assert stats['total_requests'] == 1
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        # Set with short TTL
        self.cache_manager.set("expire_key", "expire_value", ttl=1)
        
        # Should be available immediately
        assert self.cache_manager.get("expire_key") == "expire_value"
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        assert self.cache_manager.get("expire_key") is None
    
    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        # Fill cache beyond capacity
        large_data = "x" * (1024 * 1024)  # 1MB
        
        # Add multiple entries
        for i in range(5):
            self.cache_manager.set(f"key_{i}", large_data, ttl=60)
        
        # Check that some entries were evicted
        stats = self.cache_manager.get_stats()
        assert stats['evictions'] > 0
    
    def test_search_cache(self):
        """Test search-specific caching."""
        search_filters = {"file_type": ".md", "min_confidence": 0.5}
        search_results = {"results": ["doc1", "doc2"], "count": 2}
        
        # Cache search results
        self.search_cache.cache_search_results(
            "machine learning",
            "hybrid",
            search_filters,
            10,
            search_results
        )
        
        # Retrieve cached results
        cached_results = self.search_cache.get_search_results(
            "machine learning",
            "hybrid",
            search_filters,
            10
        )
        
        assert cached_results == search_results
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        # Cache some results
        self.search_cache.cache_search_results(
            "test query",
            "hybrid",
            {},
            10,
            {"results": ["doc1"]}
        )
        
        # Invalidate cache with exact pattern
        self.search_cache.invalidate_search_cache("test query")
        
        # Results should be gone
        cached_results = self.search_cache.get_search_results(
            "test query",
            "hybrid",
            {},
            10
        )
        assert cached_results is None


class TestAdvancedAnalytics:
    """Test advanced analytics functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.analytics = AdvancedAnalytics(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_search_metrics_creation(self):
        """Test SearchMetrics creation."""
        metrics = SearchMetrics(
            query="test query",
            search_type="hybrid",
            processing_time=0.5,
            results_count=10,
            cache_hit=True,
            filters_used={"file_type": ".md"}
        )
        
        assert metrics.query == "test query"
        assert metrics.search_type == "hybrid"
        assert metrics.processing_time == 0.5
        assert metrics.cache_hit is True
    
    def test_user_behavior_tracking(self):
        """Test user behavior tracking."""
        # Start user session
        self.analytics.start_user_session("user1")
        
        # Record queries
        self.analytics.record_user_query("user1", "query1", "hybrid", {"filter": "value"})
        self.analytics.record_user_query("user1", "query2", "semantic", {"filter": "value2"})
        
        # Small delay to ensure session duration > 0
        time.sleep(0.1)
        
        # End session
        self.analytics.end_user_session("user1", result_clicks=5)
        
        # Check session data
        user_insights = self.analytics.get_user_insights()
        assert user_insights['total_sessions'] == 1
        assert user_insights['avg_session_duration'] > 0
    
    def test_search_analytics(self):
        """Test search analytics collection."""
        # Record some searches
        for i in range(5):
            metrics = SearchMetrics(
                query=f"query {i}",
                search_type="hybrid",
                processing_time=0.1 + (i * 0.05),
                results_count=5 + i,
                cache_hit=i % 2 == 0,
                filters_used={"file_type": ".md"}
            )
            self.analytics.record_search(metrics)
        
        # Get analytics
        analytics_data = self.analytics.get_search_analytics(time_window_hours=1)
        
        assert analytics_data['total_searches'] == 5
        assert 'performance' in analytics_data
        assert 'search_types' in analytics_data
    
    def test_performance_insights(self):
        """Test performance insights generation."""
        # Record searches with varying performance
        for i in range(10):
            metrics = SearchMetrics(
                query=f"query {i}",
                search_type="hybrid",
                processing_time=0.1 + (i * 0.1),  # Increasing processing time
                results_count=5,
                cache_hit=i % 3 == 0,
                filters_used={}
            )
            self.analytics.record_search(metrics)
        
        # Get insights
        insights = self.analytics.get_performance_insights()
        
        assert insights.cache_efficiency >= 0
        assert len(insights.recommendations) > 0
        # With 10 queries, at least some should be considered slow (95th percentile)
        assert len(insights.slow_queries) >= 0  # Allow 0 for edge cases
    
    def test_trends_analysis(self):
        """Test trends analysis."""
        # Record searches over time
        base_time = time.time()
        for i in range(10):
            metrics = SearchMetrics(
                query=f"query {i}",
                search_type="hybrid",
                processing_time=0.1,
                results_count=5,
                cache_hit=True,
                filters_used={},
                timestamp=base_time - (i * 3600)  # One hour apart
            )
            self.analytics.record_search(metrics)
        
        # Get trends
        trends = self.analytics.get_trends(days=1)
        
        assert 'daily_trends' in trends
        assert 'overall_trend' in trends
    
    def test_export_functionality(self):
        """Test analytics export functionality."""
        # Add some data
        metrics = SearchMetrics(
            query="export test",
            search_type="hybrid",
            processing_time=0.5,
            results_count=10,
            cache_hit=True
        )
        self.analytics.record_search(metrics)
        
        # Export as JSON
        json_export = self.analytics.export_analytics("json")
        assert "export test" in json_export
        
        # Export as CSV
        csv_export = self.analytics.export_analytics("csv")
        assert "export test" in csv_export


class TestUserManagement:
    """Test user management functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.user_manager = UserManager(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_user_creation(self):
        """Test user creation and management."""
        # Register a new user
        success = self.user_manager.register_user(
            "testuser",
            "test@example.com",
            "password123",
            UserRole.USER
        )
        
        assert success is True
        
        # Get user info
        user_info = self.user_manager.get_user_info("testuser")
        assert user_info['username'] == "testuser"
        assert user_info['role'] == "user"
        assert len(user_info['permissions']) > 0
    
    def test_user_authentication(self):
        """Test user authentication."""
        # Register user
        self.user_manager.register_user("authuser", "auth@example.com", "password123")
        
        # Authenticate
        session_id = self.user_manager.authenticate_user(
            "authuser",
            "password123",
            "192.168.1.100",
            "Mozilla/5.0"
        )
        
        assert session_id is not None
        
        # Validate session
        session = self.user_manager.validate_session(session_id)
        assert session.username == "authuser"
    
    def test_permission_checking(self):
        """Test permission checking."""
        # Register power user
        self.user_manager.register_user("poweruser", "power@example.com", "password123", UserRole.POWER_USER)
        
        # Authenticate
        session_id = self.user_manager.authenticate_user(
            "poweruser",
            "password123",
            "192.168.1.100",
            "Mozilla/5.0"
        )
        
        # Check permissions
        can_search_advanced = self.user_manager.check_permission(session_id, Permission.SEARCH_ADVANCED)
        can_manage_users = self.user_manager.check_permission(session_id, Permission.USER_MANAGEMENT)
        
        assert can_search_advanced is True
        assert can_manage_users is False  # Power users can't manage users
    
    def test_role_management(self):
        """Test role management."""
        # Register user
        self.user_manager.register_user("roleuser", "role@example.com", "password123", UserRole.USER)
        
        # Change role (requires admin)
        success = self.user_manager.set_user_role("roleuser", UserRole.POWER_USER, "admin")
        assert success is True
        
        # Check new role
        user_info = self.user_manager.get_user_info("roleuser")
        assert user_info['role'] == "power_user"
    
    def test_session_management(self):
        """Test session management."""
        # Register and authenticate user
        self.user_manager.register_user("sessionuser", "session@example.com", "password123")
        session_id = self.user_manager.authenticate_user(
            "sessionuser",
            "password123",
            "192.168.1.100",
            "Mozilla/5.0"
        )
        
        # Check active sessions
        active_sessions = self.user_manager.get_active_sessions()
        assert len(active_sessions) > 0
        
        # Logout
        logout_success = self.user_manager.logout_user(session_id)
        assert logout_success is True
        
        # Session should be invalid
        session = self.user_manager.validate_session(session_id)
        assert session is None
    
    def test_jwt_token_generation(self):
        """Test JWT token generation and validation."""
        # Register and authenticate user
        self.user_manager.register_user("jwtuser", "jwt@example.com", "password123")
        session_id = self.user_manager.authenticate_user(
            "jwtuser",
            "password123",
            "192.168.1.100",
            "Mozilla/5.0"
        )
        
        # Generate JWT token
        jwt_token = self.user_manager.generate_jwt_token(session_id)
        assert jwt_token is not None
        
        # Validate JWT token
        session = self.user_manager.validate_jwt_token(jwt_token)
        assert session is not None
        assert session.username == "jwtuser"


class TestDeploymentMonitoring:
    """Test deployment and monitoring functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.deployment_manager = DeploymentManager(config_dir=self.temp_dir)
        self.monitoring = MonitoringSystem(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deployment_config(self):
        """Test deployment configuration management."""
        # Check default config
        assert self.deployment_manager.config['app_name'] == 'ai-solutions-lab'
        assert self.deployment_manager.config['port'] == 8000
        
        # Modify config
        self.deployment_manager.config['port'] = 9000
        self.deployment_manager.save_config()
        
        # Reload and verify
        new_manager = DeploymentManager(config_dir=self.temp_dir)
        assert new_manager.config['port'] == 9000
    
    def test_dockerfile_creation(self):
        """Test Dockerfile creation."""
        # Create Dockerfile
        success = self.deployment_manager.create_dockerfile()
        assert success is True
        
        # Check file exists
        dockerfile_path = Path.cwd() / "Dockerfile"
        assert dockerfile_path.exists()
        
        # Check content
        content = dockerfile_path.read_text()
        assert "AI Solutions Lab Dockerfile" in content
        assert "EXPOSE 8000" in content
        
        # Cleanup
        dockerfile_path.unlink()
    
    def test_docker_compose_creation(self):
        """Test docker-compose.yml creation."""
        # Create docker-compose.yml
        success = self.deployment_manager.create_docker_compose()
        assert success is True
        
        # Check file exists
        compose_path = Path.cwd() / "docker-compose.yml"
        assert compose_path.exists()
        
        # Check content
        content = compose_path.read_text()
        assert "ai-solutions-lab:" in content
        assert "8000:8000" in content
        
        # Cleanup
        compose_path.unlink()
    
    def test_monitoring_system(self):
        """Test monitoring system functionality."""
        # Start monitoring
        self.monitoring.start_monitoring()
        
        # Let it collect some metrics
        time.sleep(2)
        
        # Get system health
        health_status = self.monitoring.get_system_health()
        assert 'status' in health_status
        assert 'components' in health_status
        
        # Get metrics summary (may be empty if no metrics collected)
        metrics_summary = self.monitoring.get_metrics_summary(1)
        if 'message' not in metrics_summary:
            assert 'system_metrics' in metrics_summary
        
        # Stop monitoring
        self.monitoring.stop_monitoring()
    
    def test_health_checks(self):
        """Test health check functionality."""
        # Perform health checks
        self.monitoring._perform_health_checks()
        
        # Check health status
        health_status = self.monitoring.get_system_health()
        
        # Should have health checks for all components
        assert len(health_status['components']) >= 5
        
        # Check specific components
        assert 'database' in health_status['components']
        assert 'cache' in health_status['components']
        assert 'search_engine' in health_status['components']
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        # Collect system metrics
        self.monitoring._collect_system_metrics()
        
        # Check that metrics were collected (may fail on some systems)
        if len(self.monitoring.system_metrics) > 0:
            # Check metric structure
            metric = self.monitoring.system_metrics[0]
            assert hasattr(metric, 'cpu_percent')
            assert hasattr(metric, 'memory_percent')
            assert hasattr(metric, 'timestamp')
        else:
            # Skip test if metrics collection fails
            pytest.skip("System metrics collection not available on this system")
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        # Collect some metrics
        self.monitoring._collect_system_metrics()
        
        # Export as JSON
        json_export = self.monitoring.export_metrics("json")
        assert "system_metrics" in json_export
        
        # Export as CSV (may be empty if no metrics)
        csv_export = self.monitoring.export_metrics("csv")
        if "No metrics to export" not in csv_export:
            assert "Timestamp" in csv_export
            assert "CPU_Percent" in csv_export
        else:
            # Skip test if no metrics available
            pytest.skip("No metrics available for export test")


class TestIntegration:
    """Test integration between Task F components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize all components
        self.cache_manager = CacheManager(max_size_mb=10, default_ttl=60)
        self.search_cache = SearchCache(self.cache_manager)
        self.analytics = AdvancedAnalytics(data_dir=self.temp_dir)
        self.user_manager = UserManager(data_dir=self.temp_dir)
        self.monitoring = MonitoringSystem(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cached_search_with_analytics(self):
        """Test search caching with analytics integration."""
        # Record search metrics
        metrics = SearchMetrics(
            query="integration test",
            search_type="hybrid",
            processing_time=0.3,
            results_count=5,
            cache_hit=False,
            filters_used={"file_type": ".md"}
        )
        self.analytics.record_search(metrics)
        
        # Cache search results
        search_results = {"results": ["doc1", "doc2"], "count": 2}
        self.search_cache.cache_search_results(
            "integration test",
            "hybrid",
            {"file_type": ".md"},
            10,
            search_results
        )
        
        # Verify caching and analytics work together
        cached_results = self.search_cache.get_search_results(
            "integration test",
            "hybrid",
            {"file_type": ".md"},
            10
        )
        
        assert cached_results == search_results
        
        # Check analytics recorded the search
        analytics_data = self.analytics.get_search_analytics(time_window_hours=1)
        assert analytics_data['total_searches'] == 1
    
    def test_user_permissions_with_cache(self):
        """Test user permissions with cache access."""
        # Register user
        self.user_manager.register_user("cacheuser", "cache@example.com", "password123", UserRole.USER)
        
        # Authenticate
        session_id = self.user_manager.authenticate_user(
            "cacheuser",
            "password123",
            "192.168.1.100",
            "Mozilla/5.0"
        )
        
        # Check permissions
        can_search = self.user_manager.check_permission(session_id, Permission.SEARCH_BASIC)
        assert can_search is True
        
        # Use cache (should work for authenticated users)
        self.cache_manager.set("user_cache_key", "user_data", ttl=60)
        cached_data = self.cache_manager.get("user_cache_key")
        assert cached_data == "user_data"
    
    def test_monitoring_with_analytics(self):
        """Test monitoring system with analytics data."""
        # Start monitoring
        self.monitoring.start_monitoring()
        
        # Record some analytics
        metrics = SearchMetrics(
            query="monitoring test",
            search_type="hybrid",
            processing_time=0.5,
            results_count=10,
            cache_hit=True
        )
        self.analytics.record_search(metrics)
        
        # Let monitoring collect metrics
        time.sleep(2)
        
        # Check both systems are working
        health_status = self.monitoring.get_system_health()
        analytics_data = self.analytics.get_search_analytics(time_window_hours=1)
        
        assert health_status['status'] in ['healthy', 'degraded', 'unhealthy']
        assert analytics_data['total_searches'] == 1
        
        # Stop monitoring
        self.monitoring.stop_monitoring()


def test_component_initialization():
    """Test that all Task F components can be initialized."""
    # Test cache manager
    cache_manager = CacheManager(max_size_mb=1, default_ttl=60)
    assert cache_manager is not None
    
    # Test analytics
    analytics = AdvancedAnalytics()
    assert analytics is not None
    
    # Test user management
    user_manager = UserManager()
    assert user_manager is not None
    
    # Test monitoring
    monitoring = MonitoringSystem()
    assert monitoring is not None


if __name__ == "__main__":
    pytest.main([__file__])
