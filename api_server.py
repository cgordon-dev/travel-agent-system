"""
Travel Agent System API Server
Exposes the travel agent functionality via a REST API and provides metrics for monitoring
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, List

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from prometheus_client import Counter, Summary, Gauge, Info, start_http_server

from travel_agent_architecture import TravelPlanningSystem
from reinforcement_learning import RLEnhancedTravelSystem
from llm_config import LLMProvider, ModelType, get_llm_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# Initialize metrics
REQUESTS = Counter(
    'travel_agent_requests_total', 
    'Total requests to the Travel Agent API',
    ['endpoint']
)

ERRORS = Counter(
    'travel_agent_requests_errors_total', 
    'Total errors in the Travel Agent API',
    ['endpoint', 'error_type']
)

REQUEST_LATENCY = Summary(
    'travel_agent_request_duration_seconds', 
    'Request duration in seconds',
    ['endpoint']
)

ACTIVE_USERS = Gauge(
    'travel_agent_active_users', 
    'Number of active users in the last 30 minutes'
)

AGENT_USAGE = Counter(
    'travel_agent_agent_usage_total', 
    'Usage count for each agent',
    ['agent']
)

LLM_TOKENS = Counter(
    'travel_agent_llm_tokens_total', 
    'Total tokens used by LLM providers',
    ['provider', 'model']
)

SYSTEM_INFO = Info(
    'travel_agent_system_info', 
    'Information about the Travel Agent system'
)

# Initialize Flask app
app = Flask(__name__, static_folder='ui/build', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Simple API authentication
API_USERNAME = os.environ.get('API_USERNAME', 'admin')
API_PASSWORD = os.environ.get('API_PASSWORD', 'travel123')

def check_auth(username, password):
    """Check if the provided credentials are valid"""
    return username == API_USERNAME and password == API_PASSWORD

def requires_auth(f):
    """Decorator for API endpoints that require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# Global system instance
system = None
user_activity = {}  # Track user activity for active user metric

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        request_latency = time.time() - request.start_time
        REQUEST_LATENCY.labels(endpoint=request.endpoint).observe(request_latency)
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    REQUESTS.labels(endpoint='health').inc()
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.route('/api/v1/travel/request', methods=['POST'])
@requires_auth
def process_travel_request():
    """Process a travel planning request"""
    REQUESTS.labels(endpoint='travel_request').inc()
    
    try:
        data = request.get_json()
        
        if not data:
            ERRORS.labels(endpoint='travel_request', error_type='invalid_request').inc()
            return jsonify({"error": "Invalid request data"}), 400
        
        user_id = data.get('user_id')
        user_input = data.get('user_input')
        
        if not user_id or not user_input:
            ERRORS.labels(endpoint='travel_request', error_type='missing_parameters').inc()
            return jsonify({"error": "Missing required parameters: user_id and user_input"}), 400
        
        # Track user activity for active users metric
        user_activity[user_id] = datetime.now()
        update_active_users()
        
        # Process the request
        result = system.process_user_request(user_id, user_input)
        
        # Update metrics for agent usage
        for message in result.get('messages', []):
            if 'sender' in message:
                AGENT_USAGE.labels(agent=message['sender']).inc()
        
        # Mock LLM token usage (in a real system, you'd get this from the API response)
        # This is just for demonstration purposes
        for provider, token_count in _mock_token_usage(result).items():
            LLM_TOKENS.labels(provider=provider, model="default").inc(token_count)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing travel request: {str(e)}", exc_info=True)
        ERRORS.labels(endpoint='travel_request', error_type='server_error').inc()
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route('/api/v1/travel/feedback', methods=['POST'])
@requires_auth
def submit_feedback():
    """Submit feedback for a travel planning request"""
    REQUESTS.labels(endpoint='feedback').inc()
    
    try:
        data = request.get_json()
        
        if not data:
            ERRORS.labels(endpoint='feedback', error_type='invalid_request').inc()
            return jsonify({"error": "Invalid request data"}), 400
        
        user_id = data.get('user_id')
        feedback = data.get('feedback')
        
        if not user_id or not feedback:
            ERRORS.labels(endpoint='feedback', error_type='missing_parameters').inc()
            return jsonify({"error": "Missing required parameters: user_id and feedback"}), 400
        
        # Track user activity
        user_activity[user_id] = datetime.now()
        update_active_users()
        
        # Process feedback
        if isinstance(system, RLEnhancedTravelSystem):
            system.provide_explicit_feedback(user_id, feedback)
            return jsonify({"status": "Feedback recorded"})
        else:
            # Basic system doesn't support feedback
            return jsonify({"status": "Feedback recorded", "note": "RL system not enabled, feedback stored but not processed"})
    
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        ERRORS.labels(endpoint='feedback', error_type='server_error').inc()
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route('/api/v1/system/llm-config', methods=['GET'])
@requires_auth
def get_llm_configuration():
    """Get the current LLM configuration"""
    REQUESTS.labels(endpoint='llm_config').inc()
    
    try:
        config = system.get_current_llm_configuration()
        return jsonify(config)
    
    except Exception as e:
        logger.error(f"Error getting LLM configuration: {str(e)}", exc_info=True)
        ERRORS.labels(endpoint='llm_config', error_type='server_error').inc()
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route('/api/v1/system/llm-config', methods=['PUT'])
@requires_auth
def update_llm_configuration():
    """Update the LLM configuration for an agent"""
    REQUESTS.labels(endpoint='update_llm_config').inc()
    
    try:
        data = request.get_json()
        
        if not data:
            ERRORS.labels(endpoint='update_llm_config', error_type='invalid_request').inc()
            return jsonify({"error": "Invalid request data"}), 400
        
        agent_name = data.get('agent_name')
        provider = data.get('provider')
        
        if not agent_name or not provider:
            ERRORS.labels(endpoint='update_llm_config', error_type='missing_parameters').inc()
            return jsonify({"error": "Missing required parameters: agent_name and provider"}), 400
        
        # Convert provider string to enum
        try:
            provider_enum = LLMProvider(provider)
        except ValueError:
            ERRORS.labels(endpoint='update_llm_config', error_type='invalid_provider').inc()
            return jsonify({"error": f"Invalid provider: {provider}"}), 400
        
        # Update the configuration
        system.configure_agent_provider(agent_name, provider_enum)
        
        return jsonify({"status": "Configuration updated", "config": system.get_current_llm_configuration()})
    
    except Exception as e:
        logger.error(f"Error updating LLM configuration: {str(e)}", exc_info=True)
        ERRORS.labels(endpoint='update_llm_config', error_type='server_error').inc()
        return jsonify({"error": "Server error", "message": str(e)}), 500

@app.route('/api/v1/system/stats', methods=['GET'])
@requires_auth
def get_system_stats():
    """Get system statistics"""
    REQUESTS.labels(endpoint='system_stats').inc()
    
    try:
        if isinstance(system, RLEnhancedTravelSystem):
            metrics = system.get_performance_metrics()
        else:
            metrics = {"status": "RL system not enabled"}
        
        # Add additional stats
        stats = {
            "active_users": len([ts for ts in user_activity.values() if (datetime.now() - ts).total_seconds() < 1800]),
            "total_requests": sum([c.get() for c in REQUESTS._metrics.values()]),
            "error_rate": sum([c.get() for c in ERRORS._metrics.values()]) / max(1, sum([c.get() for c in REQUESTS._metrics.values()])),
            "llm_tokens": {k: v.get() for k, v in LLM_TOKENS._metrics.items()},
            "rl_metrics": metrics
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}", exc_info=True)
        ERRORS.labels(endpoint='system_stats', error_type='server_error').inc()
        return jsonify({"error": "Server error", "message": str(e)}), 500

def update_active_users():
    """Update the active users metric based on recent activity"""
    # Count users active in the last 30 minutes
    thirty_minutes_ago = datetime.now().timestamp() - 1800
    active_count = sum(1 for ts in user_activity.values() 
                     if ts.timestamp() > thirty_minutes_ago)
    
    ACTIVE_USERS.set(active_count)

def _mock_token_usage(result):
    """Mock function to generate token usage metrics
    In a real system, you would get this from the LLM provider's API response
    """
    # This is just for demonstration purposes
    tokens = {}
    
    if result.get('itinerary'):
        # Mock token usage based on the result size
        itinerary_size = len(json.dumps(result.get('itinerary', {})))
        message_count = len(result.get('messages', []))
        
        # Get usage per provider from the current config
        config = system.get_current_llm_configuration()
        for agent, provider in config.get('agent_providers', {}).items():
            token_estimate = 0
            
            if agent == 'user_interaction_agent':
                token_estimate = 500 + (message_count * 200)
            elif agent == 'planning_agent':
                token_estimate = 1000 + (itinerary_size / 10)
            elif agent == 'booking_agent':
                token_estimate = 800
            elif agent == 'monitoring_agent':
                token_estimate = 300
            
            if provider in tokens:
                tokens[provider] += token_estimate
            else:
                tokens[provider] = token_estimate
    
    # Ensure we have some token usage even if no itinerary was created
    if not tokens:
        tokens = {'openai': 500, 'anthropic': 500}
    
    return tokens

def initialize_system():
    """Initialize the travel planning system"""
    global system
    
    # Use the RL-enhanced system if available
    try:
        # Initialize the base system
        base_system = TravelPlanningSystem()
        
        # Try to enhance with RL
        system = RLEnhancedTravelSystem(base_system, "rl_config.json")
        logger.info("Initialized RL-enhanced travel planning system")
    except Exception as e:
        logger.warning(f"Could not initialize RL system: {e}")
        logger.info("Falling back to base travel planning system")
        system = TravelPlanningSystem()
    
    # Set system info for metrics
    SYSTEM_INFO.info({
        'version': '1.0.0',
        'rl_enabled': str(isinstance(system, RLEnhancedTravelSystem)),
        'start_time': datetime.now().isoformat()
    })

def start_metrics_server():
    """Start the Prometheus metrics server"""
    try:
        start_http_server(8001)
        logger.info("Started metrics server on port 8001")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

# Serve UI static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve the React UI"""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Initialize the system
    initialize_system()
    
    # Start metrics server
    start_metrics_server()
    
    # Ensure UI directory exists
    os.makedirs(os.path.join('ui', 'build'), exist_ok=True)
    
    # Start the API server
    app.run(host='0.0.0.0', port=8000)