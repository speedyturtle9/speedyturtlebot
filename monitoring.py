import psutil
import logging
import time
from datetime import datetime
from typing import Dict, List, Any

class SystemMonitor:
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'network_usage': []
        }
        self.alert_thresholds = {
            'cpu_usage': 80.0,  # 80% CPU usage
            'memory_usage': 85.0,  # 85% memory usage
            'disk_usage': 90.0,  # 90% disk usage
        }

    def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        try:
            metrics = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_usage': self._get_network_usage()
            }
            
            # Store metrics in history
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
                
                # Keep only last 24 hours of data (assuming 1-minute intervals)
                if len(self.metrics_history[key]) > 1440:
                    self.metrics_history[key].pop(0)
                    
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {str(e)}")
            return {}

    def _get_network_usage(self) -> float:
        """Calculate network usage"""
        try:
            net_io = psutil.net_io_counters()
            time.sleep(1)
            net_io_after = psutil.net_io_counters()
            
            bytes_sent = net_io_after.bytes_sent - net_io.bytes_sent
            bytes_recv = net_io_after.bytes_recv - net_io.bytes_recv
            
            return (bytes_sent + bytes_recv) / 1024  # Convert to KB
            
        except Exception as e:
            logging.error(f"Error calculating network usage: {str(e)}")
            return 0.0

    def check_alerts(self) -> List[str]:
        """Check for any metric alerts"""
        alerts = []
        current_metrics = self.collect_metrics()
        
        for metric, value in current_metrics.items():
            if metric in self.alert_thresholds:
                if value > self.alert_thresholds[metric]:
                    alert_msg = f"HIGH {metric}: {value}%"
                    alerts.append(alert_msg)
                    logging.warning(alert_msg)
                    
        return alerts

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    metric: {
                        'current': self.metrics_history[metric][-1],
                        'avg_1h': sum(self.metrics_history[metric][-60:]) / 60,
                        'avg_24h': sum(self.metrics_history[metric]) / len(self.metrics_history[metric])
                    }
                    for metric in self.metrics_history
                }
            }
        except Exception as e:
            logging.error(f"Error generating performance report: {str(e)}")
            return {}