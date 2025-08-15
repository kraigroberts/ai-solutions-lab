"""Advanced analytics system for AI Solutions Lab with detailed insights and recommendations."""

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import math
from datetime import datetime, timedelta
from pathlib import Path

@dataclass
class SearchMetrics:
    """Detailed metrics for search performance."""
    query: str
    search_type: str
    processing_time: float
    results_count: int
    cache_hit: bool
    user_satisfaction: Optional[float] = None  # 0-1 scale
    filters_used: Dict[str, Any] = None
    timestamp: float = None

@dataclass
class UserBehavior:
    """User behavior patterns and preferences."""
    session_id: str
    queries: List[str]
    search_types: List[str]
    filters_used: Dict[str, int]
    session_duration: float
    result_clicks: int
    timestamp: float

@dataclass
class PerformanceInsights:
    """Performance insights and recommendations."""
    slow_queries: List[Dict[str, Any]]
    popular_filters: List[Tuple[str, int]]
    cache_efficiency: float
    search_type_performance: Dict[str, float]
    recommendations: List[str]

class AdvancedAnalytics:
    """Advanced analytics system with machine learning insights."""
    
    def __init__(self, data_dir: str = "./data/analytics"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Analytics storage
        self.search_metrics: List[SearchMetrics] = []
        self.user_sessions: Dict[str, UserBehavior] = {}
        self.query_patterns: Counter = Counter()
        self.filter_usage: Counter = Counter()
        self.search_type_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Load existing analytics
        self._load_analytics()
    
    def record_search(self, metrics: SearchMetrics):
        """Record search metrics for analysis."""
        if metrics.timestamp is None:
            metrics.timestamp = time.time()
        
        self.search_metrics.append(metrics)
        
        # Update counters
        self.query_patterns[metrics.query.lower()] += 1
        if metrics.filters_used:
            for filter_key, filter_value in metrics.filters_used.items():
                self.filter_usage[f"{filter_key}:{filter_value}"] += 1
        
        # Update search type performance
        self.search_type_performance[metrics.search_type].append(metrics.processing_time)
        
        # Keep only last 10,000 searches to prevent memory issues
        if len(self.search_metrics) > 10000:
            self.search_metrics = self.search_metrics[-10000:]
    
    def start_user_session(self, session_id: str):
        """Start tracking a user session."""
        self.user_sessions[session_id] = UserBehavior(
            session_id=session_id,
            queries=[],
            search_types=[],
            filters_used=defaultdict(int),
            session_duration=0.0,
            result_clicks=0,
            timestamp=time.time()
        )
    
    def record_user_query(self, session_id: str, query: str, search_type: str, filters: Dict[str, Any] = None):
        """Record a user query in their session."""
        if session_id in self.user_sessions:
            session = self.user_sessions[session_id]
            session.queries.append(query)
            session.search_types.append(search_type)
            
            if filters:
                for filter_key, filter_value in filters.items():
                    session.filters_used[f"{filter_key}:{filter_value}"] += 1
    
    def end_user_session(self, session_id: str, result_clicks: int = 0):
        """End a user session and calculate metrics."""
        if session_id in self.user_sessions:
            session = self.user_sessions[session_id]
            session.session_duration = time.time() - session.timestamp
            session.result_clicks = result_clicks
    
    def get_search_analytics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive search analytics for the specified time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        # Filter recent searches
        recent_searches = [
            m for m in self.search_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_searches:
            return {"message": "No search data available for the specified time window"}
        
        # Calculate metrics
        total_searches = len(recent_searches)
        avg_processing_time = sum(m.processing_time for m in recent_searches) / total_searches
        cache_hit_rate = sum(1 for m in recent_searches if m.cache_hit) / total_searches * 100
        
        # Search type distribution
        search_type_counts = Counter(m.search_type for m in recent_searches)
        
        # Performance by search type
        type_performance = {}
        for search_type in search_type_counts:
            type_times = [m.processing_time for m in recent_searches if m.search_type == search_type]
            type_performance[search_type] = {
                'count': len(type_times),
                'avg_time': sum(type_times) / len(type_times),
                'min_time': min(type_times),
                'max_time': max(type_times)
            }
        
        # Popular queries
        popular_queries = self.query_patterns.most_common(10)
        
        # Filter usage
        popular_filters = self.filter_usage.most_common(10)
        
        # Time-based patterns
        hourly_distribution = defaultdict(int)
        for search in recent_searches:
            hour = datetime.fromtimestamp(search.timestamp).hour
            hourly_distribution[hour] += 1
        
        return {
            'time_window_hours': time_window_hours,
            'total_searches': total_searches,
            'performance': {
                'avg_processing_time': round(avg_processing_time, 3),
                'cache_hit_rate': round(cache_hit_rate, 2),
                'total_cache_hits': sum(1 for m in recent_searches if m.cache_hit),
                'total_cache_misses': sum(1 for m in recent_searches if not m.cache_hit)
            },
            'search_types': dict(search_type_counts),
            'type_performance': type_performance,
            'popular_queries': popular_queries,
            'popular_filters': popular_filters,
            'hourly_distribution': dict(hourly_distribution),
            'timestamp': time.time()
        }
    
    def get_performance_insights(self) -> PerformanceInsights:
        """Get performance insights and recommendations."""
        if not self.search_metrics:
            return PerformanceInsights(
                slow_queries=[],
                popular_filters=[],
                cache_efficiency=0.0,
                search_type_performance={},
                recommendations=["No search data available for analysis"]
            )
        
        # Find slow queries (above 95th percentile)
        processing_times = [m.processing_time for m in self.search_metrics]
        slow_threshold = sorted(processing_times)[int(len(processing_times) * 0.95)]
        
        slow_queries = [
            {
                'query': m.query,
                'search_type': m.search_type,
                'processing_time': m.processing_time,
                'timestamp': m.timestamp
            }
            for m in self.search_metrics
            if m.processing_time > slow_threshold
        ][-10:]  # Last 10 slow queries
        
        # Popular filters
        popular_filters = self.filter_usage.most_common(10)
        
        # Cache efficiency
        total_searches = len(self.search_metrics)
        cache_hits = sum(1 for m in self.search_metrics if m.cache_hit)
        cache_efficiency = (cache_hits / total_searches) * 100 if total_searches > 0 else 0
        
        # Search type performance
        type_performance = {}
        for search_type, times in self.search_type_performance.items():
            if times:
                type_performance[search_type] = {
                    'avg_time': sum(times) / len(times),
                    'count': len(times),
                    'efficiency': 1.0 / (sum(times) / len(times)) if sum(times) > 0 else 0
                }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            cache_efficiency, type_performance, slow_queries
        )
        
        return PerformanceInsights(
            slow_queries=slow_queries,
            popular_filters=popular_filters,
            cache_efficiency=cache_efficiency,
            search_type_performance=type_performance,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, cache_efficiency: float, type_performance: Dict[str, Any], slow_queries: List[Dict[str, Any]]) -> List[str]:
        """Generate intelligent recommendations based on analytics."""
        recommendations = []
        
        # Cache recommendations
        if cache_efficiency < 50:
            recommendations.append("Consider increasing cache size or TTL to improve performance")
        elif cache_efficiency < 70:
            recommendations.append("Cache efficiency could be improved - review cache invalidation strategy")
        
        # Performance recommendations
        if slow_queries:
            recommendations.append(f"Found {len(slow_queries)} slow queries - consider query optimization")
        
        # Search type recommendations
        if type_performance:
            slowest_type = min(type_performance.items(), key=lambda x: x[1]['avg_time'])
            fastest_type = max(type_performance.items(), key=lambda x: x[1]['avg_time'])
            
            if slowest_type[1]['avg_time'] > fastest_type[1]['avg_time'] * 2:
                recommendations.append(f"Consider using {fastest_type[0]} search type for better performance")
        
        # General recommendations
        if len(self.search_metrics) > 1000:
            recommendations.append("High search volume detected - consider implementing query result caching")
        
        if not recommendations:
            recommendations.append("System performance is optimal - no immediate actions required")
        
        return recommendations
    
    def get_user_insights(self) -> Dict[str, Any]:
        """Get insights about user behavior patterns."""
        if not self.user_sessions:
            return {"message": "No user session data available"}
        
        # Session analysis
        session_durations = [s.session_duration for s in self.user_sessions.values()]
        avg_session_duration = sum(session_durations) / len(session_durations)
        
        # Query patterns
        all_queries = []
        for session in self.user_sessions.values():
            all_queries.extend(session.queries)
        
        query_count = Counter(all_queries)
        most_common_queries = query_count.most_common(10)
        
        # Search type preferences
        all_search_types = []
        for session in self.user_sessions.values():
            all_search_types.extend(session.search_types)
        
        search_type_preferences = Counter(all_search_types)
        
        # Filter usage patterns
        all_filters = defaultdict(int)
        for session in self.user_sessions.values():
            for filter_key, count in session.filters_used.items():
                all_filters[filter_key] += count
        
        popular_filters = sorted(all_filters.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_sessions': len(self.user_sessions),
            'avg_session_duration': round(avg_session_duration, 2),
            'most_common_queries': most_common_queries,
            'search_type_preferences': dict(search_type_preferences),
            'popular_filters': popular_filters,
            'timestamp': time.time()
        }
    
    def get_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get trends over the specified number of days."""
        end_time = time.time()
        start_time = end_time - (days * 24 * 3600)
        
        # Group searches by day
        daily_searches = defaultdict(list)
        for metric in self.search_metrics:
            if start_time <= metric.timestamp <= end_time:
                day = datetime.fromtimestamp(metric.timestamp).strftime('%Y-%m-%d')
                daily_searches[day].append(metric)
        
        # Calculate daily metrics
        trends = {}
        for day, searches in daily_searches.items():
            if searches:
                trends[day] = {
                    'total_searches': len(searches),
                    'avg_processing_time': sum(s.processing_time for s in searches) / len(searches),
                    'cache_hit_rate': sum(1 for s in searches if s.cache_hit) / len(searches) * 100,
                    'search_types': Counter(s.search_type for s in searches)
                }
        
        return {
            'period_days': days,
            'daily_trends': trends,
            'overall_trend': self._calculate_trend(trends),
            'timestamp': time.time()
        }
    
    def _calculate_trend(self, daily_trends: Dict[str, Any]) -> str:
        """Calculate overall trend direction."""
        if len(daily_trends) < 2:
            return "insufficient_data"
        
        # Calculate trend for total searches
        days = sorted(daily_trends.keys())
        search_counts = [daily_trends[day]['total_searches'] for day in days]
        
        if len(search_counts) >= 2:
            first_half = search_counts[:len(search_counts)//2]
            second_half = search_counts[len(search_counts)//2:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg > first_avg * 1.1:
                return "increasing"
            elif second_avg < first_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
        
        return "stable"
    
    def export_analytics(self, format: str = "json") -> str:
        """Export analytics data in various formats."""
        if format.lower() == "csv":
            return self._export_csv()
        else:
            return self._export_json()
    
    def _export_json(self) -> str:
        """Export analytics as JSON."""
        export_data = {
            'search_metrics': [asdict(m) for m in self.search_metrics[-1000:]],  # Last 1000 searches
            'user_sessions': {k: asdict(v) for k, v in list(self.user_sessions.items())[-100:]},  # Last 100 sessions
            'query_patterns': dict(self.query_patterns.most_common(100)),
            'filter_usage': dict(self.filter_usage.most_common(100)),
            'export_timestamp': time.time()
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_csv(self) -> str:
        """Export analytics as CSV."""
        if not self.search_metrics:
            return "No data to export"
        
        # CSV header
        csv_lines = ["Query,SearchType,ProcessingTime,ResultsCount,CacheHit,Timestamp"]
        
        # CSV data
        for metric in self.search_metrics[-1000:]:  # Last 1000 searches
            timestamp = datetime.fromtimestamp(metric.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            csv_lines.append(f'"{metric.query}","{metric.search_type}",{metric.processing_time},{metric.results_count},{metric.cache_hit},"{timestamp}"')
        
        return "\n".join(csv_lines)
    
    def _load_analytics(self):
        """Load existing analytics data from disk."""
        try:
            analytics_file = self.data_dir / "analytics.json"
            if analytics_file.exists():
                with open(analytics_file, 'r') as f:
                    data = json.load(f)
                
                # Restore search metrics
                if 'search_metrics' in data:
                    for metric_data in data['search_metrics']:
                        metric = SearchMetrics(**metric_data)
                        self.search_metrics.append(metric)
                
                # Restore counters
                if 'query_patterns' in data:
                    self.query_patterns = Counter(data['query_patterns'])
                
                if 'filter_usage' in data:
                    self.filter_usage = Counter(data['filter_usage'])
                
                print(f"Loaded {len(self.search_metrics)} search metrics from disk")
                
        except Exception as e:
            print(f"Warning: Could not load analytics: {e}")
    
    def save_analytics(self):
        """Save analytics data to disk."""
        try:
            analytics_file = self.data_dir / "analytics.json"
            
            export_data = {
                'search_metrics': [asdict(m) for m in self.search_metrics],
                'query_patterns': dict(self.query_patterns),
                'filter_usage': dict(self.filter_usage),
                'save_timestamp': time.time()
            }
            
            with open(analytics_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
            print(f"Analytics saved to {analytics_file}")
            
        except Exception as e:
            print(f"Warning: Could not save analytics: {e}")


def main():
    """Demo the advanced analytics system."""
    print("Advanced Analytics System Demo")
    print("=" * 40)
    
    # Create analytics system
    analytics = AdvancedAnalytics()
    
    # Simulate some search data
    print("Simulating search data...")
    
    # Record some searches
    for i in range(10):
        metrics = SearchMetrics(
            query=f"machine learning {i}",
            search_type="hybrid",
            processing_time=0.1 + (i * 0.05),
            results_count=5 + i,
            cache_hit=i % 3 == 0,
            filters_used={"file_type": ".md", "min_confidence": 0.5}
        )
        analytics.record_search(metrics)
    
    # Simulate user sessions
    analytics.start_user_session("user1")
    analytics.record_user_query("user1", "machine learning", "hybrid", {"file_type": ".md"})
    analytics.record_user_query("user1", "neural networks", "semantic", {"file_type": ".pdf"})
    analytics.end_user_session("user1", result_clicks=3)
    
    # Get analytics
    print("\nSearch Analytics (last 24 hours):")
    search_analytics = analytics.get_search_analytics()
    print(json.dumps(search_analytics, indent=2, default=str))
    
    print("\nPerformance Insights:")
    insights = analytics.get_performance_insights()
    print(json.dumps(asdict(insights), indent=2, default=str))
    
    print("\nUser Insights:")
    user_insights = analytics.get_user_insights()
    print(json.dumps(user_insights, indent=2, default=str))
    
    # Save analytics
    analytics.save_analytics()
    print("\nAnalytics saved to disk")


if __name__ == "__main__":
    main()
