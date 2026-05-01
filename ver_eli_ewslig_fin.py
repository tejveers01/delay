# Construction Delay Analysis API with LLM Integration
import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import threading
import time
import pickle
import asyncio

import numpy as np
import pandas as pd
import ibm_boto3
from ibm_botocore.client import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import ModelInference

# --------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

COS_API_KEY = os.getenv("COS_API_KEY")
COS_CRN = os.getenv("COS_SERVICE_INSTANCE_CRN")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
COS_BUCKET = os.getenv("COS_BUCKET_NAME")

# Watsonx credentials
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-70b-instruct")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.1))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 200))

# Cache settings
CACHE_FILE = "project_cache.pkl"
CACHE_TTL_HOURS = 24


# Validate required environment variables
if not all([COS_API_KEY, COS_CRN, COS_ENDPOINT, COS_BUCKET]):
    logger.warning("Missing one or more COS environment variables")

# --------------------------------------------------
# ENUMS & TYPES
# --------------------------------------------------

class QueryType(str, Enum):
    """Available query types for delay analysis."""
    TOWER_WISE = "tower-wise"
    FLOOR_WISE = "floor-wise"
    FLAT_WISE = "flat-wise"
    POUR_WISE = "pour-wise"
    MODULE_WISE = "module-wise"
    ACTIVITY_WISE = "activity-wise"
    ACTIVITY_DETAILS = "activity-details"
    MULTI_TOWER = "multi-tower"
    SUMMARY = "summary"
    COMPARISON = "comparison"

class UserIntent(str, Enum):
    """Possible user intents."""
    GET_DELAYS = "get_delays"
    GET_ACTIVITY_DETAILS = "get_activity_details"
    GET_SUMMARY = "get_summary"
    COMPARE_TOWERS = "compare_towers"
    FIND_CRITICAL = "find_critical_delays"
    GET_STATUS = "get_status"
    GET_PROGRESS = "get_progress"
    LIST_PROJECTS = "list_projects"
    LIST_TOWERS = "list_towers"
    MULTI_PROJECT = "multi_project" 

class CacheStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"
    NEEDS_UPDATE = "needs_update"

# --------------------------------------------------
# PYDANTIC MODELS
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    project: Optional[str] = None
    tower: Optional[str] = None
    use_llm: bool = True

class IntentResponse(BaseModel):
    intent: UserIntent
    confidence: float
    project: Optional[str] = None
    towers: List[str] = []
    query_type: QueryType
    filters: Dict[str, Any] = {}
    activity_id: Optional[str] = None
    parsed_query: Dict[str, Any] = {}

# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Construction Delay Analysis API",
    version="1.0",
    description="API for analyzing construction project delays with LLM-powered query understanding",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --------------------------------------------------
# IBM COS CLIENT
# --------------------------------------------------

cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY,
    ibm_service_instance_id=COS_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
)

# --------------------------------------------------
# LLM SERVICE
# --------------------------------------------------

class LLMService:
    """Service for LLM-powered query understanding."""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        self.init_lock = threading.Lock()
        
    def initialize(self):
        """Initialize the LLM model."""
        if self.initialized:
            return
            
        with self.init_lock:
            if self.initialized:  # Double-check
                return
                
            try:
                # Initialize credentials as dictionary
                credentials = {
                    "url": WATSONX_URL,
                    "apikey": WATSONX_API_KEY
                }
                
                # Initialize model
                self.model = ModelInference(
                    model_id=WATSONX_MODEL_ID,
                    credentials=credentials,
                    project_id=WATSONX_PROJECT_ID,
                    params={
                        "temperature": MODEL_TEMPERATURE, 
                        "max_new_tokens": MODEL_MAX_TOKENS,
                        "min_new_tokens": 1,
                        "decoding_method": "greedy",
                        "repetition_penalty": 1.0
                    }
                )
                self.initialized = True
                logger.info(f"LLM service initialized successfully with model: {WATSONX_MODEL_ID}")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLM service: {e}", exc_info=True)
                self.model = None
    
    async def analyze_query(self, query: str, available_projects: List[str]) -> IntentResponse:
        """Analyze user query using LLM."""
        if not self.initialized:
            self.initialize()
            
            # If still not initialized, use fallback
            if not self.initialized:
                logger.warning("LLM not initialized, using fallback parser")
                return self._fallback_parser(query, available_projects)
        
        try:
            # Create prompt for intent classification
            prompt = self._create_intent_prompt(query, available_projects)
            
            # Get LLM response
            response = await self._get_llm_response(prompt)
            
            # Parse response
            intent_data = self._parse_llm_response(response)
            
            logger.info(f"LLM analysis successful: {intent_data.intent.value} (confidence: {intent_data.confidence})")
            logger.info(f"LLM detected project: {intent_data.project}, towers: {intent_data.towers}, query_type: {intent_data.query_type.value}")
            return intent_data
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
            # Fall back to rule-based parser
            return self._fallback_parser(query, available_projects)
    
    def _create_intent_prompt(self, query: str, available_projects: List[str]) -> str:
        """Create prompt for intent classification."""
        
        # Truncate available projects list to prevent context length exceeded errors
        # If there are too many projects, only show the first 50
        projects_context = ""
        if available_projects:
            if len(available_projects) > 50:
                projects_context = json.dumps(available_projects[:50]) + f" ... and {len(available_projects) - 50} more"
            else:
                projects_context = json.dumps(available_projects)
        else:
            projects_context = 'None available'
        
        prompt = f"""Analyze this construction project delay analysis query and extract structured information.

User Query: "{query}"

Available Projects: {projects_context}

Return ONLY a JSON object with this EXACT structure:
    {{
        "intent": "get_delays|get_activity_details|get_summary|compare_towers|find_critical_delays|get_status|get_progress|list_projects|list_towers|multi_project",
        "confidence": 0.0-1.0,
        "project": "project_name_or_null",
        "towers": ["tower1", "tower2"],
        "query_type": "tower-wise|floor-wise|flat-wise|pour-wise|module-wise|activity-wise|activity-details|multi-tower|summary|comparison",
        "filters": {{}},
        "activity_id": "id_or_null"
    }}
    
    CRITICAL RULES:
    1. If query mentions "all projects", set intent to "multi_project" and leave project as null.
    2. If query mentions "all towers" or implies checking the whole project, set query_type to "multi-tower".
    3. If query mentions specific flat/unit (e.g., "Flat 113", "Unit 404"), set query_type to "activity-wise" and add "flat": "113" to filters.
    4. If query mentions specific floor (e.g., "3F", "3rd Floor"), set query_type to "activity-wise" and add "floor": "3" to filters.
    5. If query mentions specific module (e.g., "M7", "Module 7"), set query_type to "activity-wise" and add "module": "7" to filters.
    6. If query mentions specific pour number, set query_type to "activity-wise" and add "pour": "X" to filters.
    7. If query mentions specific activity ID, set query_type to "activity-details" and set activity_id.
    8. "Status" queries for a specific entity (flat, floor) are "get_delays" or "get_status" with "activity-wise" query_type.
    9. "Progress" queries usually imply "get_progress" intent or "get_summary".
    10. Be robust to word order (e.g., "Tower A delay" == "Delay in Tower A").
    11. If query mentions trade (e.g., "painting", "electrical"), add "trade": "trade_name" to filters.
    12. If query mentions progress (e.g., "less than 50%"), add "progress_op": "lt", "progress_val": 50 to filters.
    13. If query mentions dates (e.g., "in Jan", "last month"), add "month" or "date_period" to filters.
    
    EXAMPLES:
    1. Query: "show me delay for all projects" -> {{
        "intent": "multi_project",
        "confidence": 0.98,
        "project": null,
        "towers": [],
        "query_type": "multi-tower",
        "filters": {{}},
        "activity_id": null
    }}
    2. Query: "analyze delays in tower A and B of Veridia" -> {{
        "intent": "get_delays",
        "confidence": 0.95,
        "project": "Veridia",
        "towers": ["A", "B"],
        "query_type": "multi-tower",
        "filters": {{}},
        "activity_id": null
    }}
    3. Query: "Flat 101 progress in tower C" -> {{
        "intent": "get_status",
        "confidence": 0.95,
        "project": null,
        "towers": ["C"],
        "query_type": "activity-wise",
        "filters": {{"flat": "101"}},
        "activity_id": null
    }}
    4. Query: "how is the pouring going in tower 1?" -> {{
        "intent": "get_status",
        "confidence": 0.90,
        "project": null,
        "towers": ["1"],
        "query_type": "pour-wise",
        "filters": {{}},
        "activity_id": null
    }}
    5. Query: "details for activity 12345" -> {{
        "intent": "get_activity_details",
        "confidence": 0.99,
        "project": null,
        "towers": [],
        "query_type": "activity-details",
        "filters": {{"activity_id": "12345"}},
        "activity_id": "12345"
    }}
    6. Query: "show delayed painting activities with less than 50% progress" -> {{
        "intent": "get_delays",
        "confidence": 0.95,
        "project": null,
        "towers": [],
        "query_type": "activity-wise",
        "filters": {{"trade": "paint", "progress_op": "lt", "progress_val": 50, "status": "delayed"}},
        "activity_id": null
    }}
    
    CRITICAL INSTRUCTION: Return ONLY the JSON object. Do NOT add any explanations, comments, markdown formatting, or additional text. Just the JSON.
    
    Now analyze the query "{query}" and return ONLY valid JSON:
    """
        return prompt
        
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            if not self.model:
                return ""
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate(prompt=prompt)
            )
            
            # Handle different response formats
            if isinstance(response, dict):
                return response.get("results", [{}])[0].get("generated_text", "")
            elif isinstance(response, list):
                return response[0].get("generated_text", "")
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return ""
    
    def _parse_llm_response(self, response: str) -> IntentResponse:
        """Parse LLM response to IntentResponse."""
        try:
            # Clean response - extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON found in LLM response: {response[:200]}")
                return self._create_default_response()
            
            json_str = json_match.group()
            
            # FIX: Replace common LLM typos
            json_str = json_str.replace('nulll', 'null')  # Fix "nulll" typo (triple L)
            json_str = json_str.replace('nul', 'null')    # Fix "nul" typo
            json_str = json_str.replace('NUL', 'null')
            json_str = json_str.replace('None', 'null')   # Fix Python None
            json_str = json_str.replace('NULL', 'null')   # Fix uppercase NULL
            
            # Also fix activity_id specifically
            json_str = re.sub(r'"activity_id"\s*:\s*"nulll?"', '"activity_id": null', json_str, flags=re.IGNORECASE)
            
            data = json.loads(json_str)
            
            # Convert intent string to UserIntent enum
            intent_str = data.get("intent", "get_delays")
            intent = self._map_intent(intent_str)
            
            # Convert query_type string to QueryType enum
            query_type_str = data.get("query_type", "tower-wise")
            query_type = self._map_query_type(query_type_str)
            
            # Handle null/None project properly
            project = data.get("project")
            if project in [None, "null", "None", "NULL"]:
                project = None
            
            # FIX: Clean and validate tower names
            towers = data.get("towers", [])
            if not isinstance(towers, list):
                towers = []
            
            # Filter out invalid tower names
            valid_towers = []
            invalid_keywords = {"IN", "FOR", "ALL", "THE", "AND", "TO", "OF", "WITH", "FROM", "MY", "YOUR", "THIS", "THAT"}
            
            for tower in towers:
                if not isinstance(tower, str):
                    continue
                    
                tower_upper = tower.upper().strip()
                
                # Skip common English words
                if tower_upper in invalid_keywords:
                    logger.warning(f"Filtered out invalid tower name: {tower}")
                    continue
                
                # Skip if it looks like a preposition followed by project name
                if len(tower_upper) <= 3 and tower_upper in ["IN", "FOR"]:
                    logger.warning(f"Filtered out preposition as tower: {tower}")
                    continue
                
                # Tower names should be 1-3 alphanumeric characters
                if 1 <= len(tower_upper) <= 3 and tower_upper.isalnum():
                    valid_towers.append(tower_upper)
                else:
                    logger.warning(f"Filtered out invalid tower format: {tower}")
            
            towers = valid_towers
            
            # DEBUG: Log parsed data
            logger.info(f"Parsed LLM data - project: {project}, towers: {towers}, query_type: {query_type_str}, intent: {intent_str}")
            
            return IntentResponse(
                intent=intent,
                confidence=float(data.get("confidence", 0.5)),
                project=project,
                towers=towers,
                query_type=query_type,
                filters=data.get("filters", {}),
                activity_id=data.get("activity_id"),
                parsed_query=data
            )
            
        except Exception as e:
            logger.error(f"Parse error: {e}, response: {response[:500]}")
            return self._create_default_response()
    
    def _map_intent(self, intent_str: str) -> UserIntent:
        """Map intent string to UserIntent enum."""
        intent_map = {
            "get_delays": UserIntent.GET_DELAYS,
            "get_activity_details": UserIntent.GET_ACTIVITY_DETAILS,
            "get_summary": UserIntent.GET_SUMMARY,
            "compare_towers": UserIntent.COMPARE_TOWERS,
            "find_critical_delays": UserIntent.FIND_CRITICAL,
            "get_status": UserIntent.GET_STATUS,
            "get_progress": UserIntent.GET_PROGRESS,
            "list_projects": UserIntent.LIST_PROJECTS,
            "list_towers": UserIntent.LIST_TOWERS,
            "multi_project": UserIntent.MULTI_PROJECT,
        }
        return intent_map.get(intent_str.lower(), UserIntent.GET_DELAYS)
    
    def _map_query_type(self, query_type_str: str) -> QueryType:
        """Map query type string to QueryType enum."""
        query_type_map = {
            "tower-wise": QueryType.TOWER_WISE,
            "floor-wise": QueryType.FLOOR_WISE,
            "flat-wise": QueryType.FLAT_WISE,
            "pour-wise": QueryType.POUR_WISE,
            "module-wise": QueryType.MODULE_WISE,
            "activity-wise": QueryType.ACTIVITY_WISE,
            "activity-details": QueryType.ACTIVITY_DETAILS,
            "multi-tower": QueryType.MULTI_TOWER,
            "summary": QueryType.SUMMARY,
            "comparison": QueryType.COMPARISON,
        }
        return query_type_map.get(query_type_str.lower(), QueryType.TOWER_WISE)
    
    def _create_default_response(self) -> IntentResponse:
        """Create default response for fallback."""
        return IntentResponse(
            intent=UserIntent.GET_DELAYS,
            confidence=0.5,
            query_type=QueryType.TOWER_WISE,
            parsed_query={"error": "LLM parsing failed"}
        )
    
    def _parse_llm_response(self, response: str) -> IntentResponse:
        """Parse LLM response to IntentResponse."""
        try:
            # Clean response - extract JSON from response
            # Remove markdown code blocks and any text before/after JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.error(f"No JSON found in LLM response: {response[:200]}")
                return self._create_default_response()
            
            json_str = json_match.group()
            
            # Remove any markdown code blocks and trailing text
            json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str)
            
            # Remove any text after the JSON (like "The final answer is:" etc.)
            json_str = re.split(r'(?:The final answer is:|Final answer:|Answer:|Analysis:|```)', json_str)[0]
            
            # Remove any boxed text like $\boxed{}$
            json_str = re.sub(r'\$\\boxed\{\}\$', '', json_str)
            
            # Remove any trailing comments or explanations
            json_str = re.sub(r'//.*', '', json_str)  # Remove JS-style comments
            json_str = re.sub(r'#.*', '', json_str)   # Remove Python-style comments
            json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)  # Remove CSS-style comments
            
            json_str = json_str.strip()
            
            # Handle the specific "nulll" typo (triple L) and other common issues
            json_str = json_str.replace('"nulll"', 'null')
            json_str = json_str.replace("'nulll'", 'null')
            json_str = json_str.replace('"nul"', 'null')
            json_str = json_str.replace("'nul'", 'null')
            json_str = json_str.replace('"None"', 'null')
            json_str = json_str.replace("'None'", 'null')
            json_str = json_str.replace('"NULL"', 'null')
            json_str = json_str.replace("'NULL'", 'null')
            
            # Also handle null values without quotes
            json_str = re.sub(r':\s*nulll?\b', ': null', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r':\s*None\b', ': null', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r':\s*NULL\b', ': null', json_str, flags=re.IGNORECASE)
            
            # Fix activity_id specifically
            json_str = re.sub(r'"activity_id"\s*:\s*"nulll?"', '"activity_id": null', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r"'activity_id'\s*:\s*'nulll?'", '"activity_id": null', json_str, flags=re.IGNORECASE)
            
            # Log the cleaned JSON for debugging
            logger.info(f"Cleaned JSON string: {json_str[:500]}")
            
            data = json.loads(json_str)
            
            # Convert intent string to UserIntent enum
            intent_str = data.get("intent", "get_delays")
            intent = self._map_intent(intent_str)
            
            # Convert query_type string to QueryType enum
            query_type_str = data.get("query_type", "tower-wise")
            query_type = self._map_query_type(query_type_str)
            
            # Handle null project
            project = data.get("project")
            if project in [None, "null", "None", "NULL"]:
                project = None
            
            # Ensure towers is always a list
            towers = data.get("towers", [])
            if not isinstance(towers, list):
                towers = []
            
            logger.info(f"Parsed LLM data successfully: intent={intent_str}, project={project}, query_type={query_type_str}, confidence={data.get('confidence', 0.5)}")
            
            return IntentResponse(
                intent=intent,
                confidence=float(data.get("confidence", 0.5)),
                project=project,
                towers=towers,
                query_type=query_type,
                filters=data.get("filters", {}),
                activity_id=data.get("activity_id"),
                parsed_query=data
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Original response: {response[:500]}")
            logger.error(f"Attempted to parse: {json_str[:500] if 'json_str' in locals() else 'N/A'}")
            return self._create_default_response()
        except Exception as e:
            logger.error(f"Parse error: {e}")
            logger.error(f"Response: {response[:500]}")
            return self._create_default_response()

# Initialize LLM service
llm_service = LLMService()



# --------------------------------------------------
# CACHE MANAGEMENT
# --------------------------------------------------

class ProjectCache:
    """Enhanced cache with tracker version checking."""
    
    def __init__(self):
        self.cache = {}
        self.tracker_versions = {}
        self.lock = threading.Lock()
        self.load_from_disk()
    
    def load_from_disk(self):
        """Load cache from disk."""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.tracker_versions = data.get('tracker_versions', {})
                logger.info(f"Loaded cache with {len(self.cache)} projects")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def save_to_disk(self):
        """Save cache to disk."""
        try:
            with self.lock:
                data = {
                    'cache': self.cache,
                    'tracker_versions': self.tracker_versions,
                    'timestamp': datetime.now().isoformat()
                }
                with open(CACHE_FILE, 'wb') as f:
                    pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def get_cache_status(self, project: str, tower: str = None) -> CacheStatus:
        """Check cache status for a project/tower."""
        with self.lock:
            if project not in self.cache:
                return CacheStatus.MISSING
            
            if tower and tower not in self.cache[project].get('towers', {}):
                return CacheStatus.MISSING
            
            project_data = self.cache[project]
            last_updated = project_data.get('last_updated')
            
            if not last_updated:
                return CacheStatus.STALE
            
            cache_age = datetime.now() - datetime.fromisoformat(last_updated)
            if cache_age > timedelta(hours=CACHE_TTL_HOURS):
                return CacheStatus.STALE
            
            return CacheStatus.FRESH
    
    def get_tracker_version(self, project: str, tower: str) -> Optional[str]:
        """Get cached tracker version for a tower."""
        key = f"{project}:{tower}"
        return self.tracker_versions.get(key)
    
    def update_tracker_version(self, project: str, tower: str, version: str):
        """Update tracker version for a tower."""
        key = f"{project}:{tower}"
        with self.lock:
            self.tracker_versions[key] = version
    
    def needs_update(self, project: str, tower: str, current_version: str) -> bool:
        """Check if tower data needs update based on tracker version."""
        cached_version = self.get_tracker_version(project, tower)
        return cached_version != current_version
    
    def get_project_data(self, project: str) -> Optional[Dict]:
        """Get cached data for a project."""
        with self.lock:
            return self.cache.get(project)
    
    def update_project(self, project: str, data: Dict):
        """Update cache for a project."""
        with self.lock:
            data['last_updated'] = datetime.now().isoformat()
            self.cache[project] = data
    
    def update_tower_data(self, project: str, tower: str, data: Dict, tracker_version: str):
        """Update cache for a specific tower."""
        with self.lock:
            if project not in self.cache:
                self.cache[project] = {'towers': {}, 'available_towers': []}
            
            project_data = self.cache[project]
            if 'towers' not in project_data:
                project_data['towers'] = {}
            
            project_data['towers'][tower] = data
            project_data['last_updated'] = datetime.now().isoformat()
            
            if tower not in project_data['available_towers']:
                project_data['available_towers'].append(tower)
            
            key = f"{project}:{tower}"
            self.tracker_versions[key] = tracker_version
    
    def get_tower_data(self, project: str, tower: str) -> Optional[Dict]:
        """Get cached data for a specific tower."""
        with self.lock:
            project_data = self.cache.get(project)
            if not project_data:
                return None
            return project_data.get('towers', {}).get(tower)
    
    def get_all_projects(self) -> List[str]:
        """Get list of all cached projects."""
        with self.lock:
            return list(self.cache.keys())

project_cache = ProjectCache()

# --------------------------------------------------
# DATA PRE-PROCESSOR
# --------------------------------------------------

class DataPreprocessor:
    """Handle pre-processing of Excel files."""
    
    def __init__(self):
        self.columns = [
            "Pour", "Module", "Floor", "Flat", "Activity ID",
            "Activity Name", "Baseline Finish", "Finish", "% Complete"
        ]

    def is_finishing_tracker_file(self, filename: str) -> bool:
        """Return True only for finishing tracker workbooks with a dated filename."""
        if not filename or not filename.lower().endswith(".xlsx"):
            return False

        if "finishing tracker" not in filename.lower():
            return False

        return re.search(r'\(\d{2}-\d{2}-\d{4}\)\.xlsx$', filename, re.IGNORECASE) is not None
    
    def extract_tracker_info(
        self,
        filename: str,
        project: Optional[str] = None,
        object_key: Optional[str] = None
    ) -> Dict[str, str]:
        """Extract tower and date from tracker filename for multiple project naming styles."""
        search_text = " ".join(
            part for part in [filename, project or "", object_key or ""] if part
        )

        date_match = re.search(r'(\d{2}-\d{2}-\d{4})', search_text)
        date_str = date_match.group(1) if date_match else None

        tower = None

        prefixed_tower_patterns = [
            r'\b(EWS|LIG)\s*TOWER\s*[-_ ]*([A-Z0-9]+)\b',
            r'\bTOWER\s*[-_ ]*([A-Z0-9]+)\s*(EWS|LIG)\b',
            r'\b(EWS|LIG)\s*[-_ ]*([A-Z0-9]+)\b',
        ]
        for pattern in prefixed_tower_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                groups = [g.upper() for g in match.groups() if g]
                prefix = next((g for g in groups if g in {"EWS", "LIG"}), None)
                identifier = next((g for g in groups if g not in {"EWS", "LIG"}), None)
                if prefix and identifier:
                    tower = f"{prefix} Tower {identifier}"
                    break

        if not tower:
            generic_patterns = [
                r'\bTOWER\s*[-_ ]*([A-Z])\b',
                r'\b([A-Z])\s*TOWER\b',
                r'\bTOWER\s*[-_ ]*([0-9]{1,3})\b',
                r'\b([0-9]{1,3})\s*TOWER\b',
                r'\bBLOCK\s*[-_ ]*([A-Z0-9]+)\b',
            ]
            for pattern in generic_patterns:
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    tower = match.group(1).upper()
                    break

        if not tower and project:
            project_lower = project.lower()
            filename_upper = filename.upper()

            if "eligo" in project_lower:
                for letter in ["F", "G", "H"]:
                    if re.search(rf'\b{letter}\b', filename_upper):
                        tower = letter
                        break
            elif "veridia" in project_lower:
                match = re.search(r'\b([4-9]|[1-9][0-9])\b', filename_upper)
                if match:
                    tower = match.group(1)

        return {'tower': tower, 'date': date_str}
    
    def get_tracker_version(self, filename: str) -> str:
        """Get tracker version string (tower + date)."""
        info = self.extract_tracker_info(filename)
        if info['tower'] and info['date']:
            return f"{info['tower']}_{info['date']}"
        return filename
    
    def preprocess_excel(self, file_data: bytes, tower: str, project: Optional[str] = None) -> Optional[Dict]:
        """Pre-process a single Excel file with JSON-safe data."""
        try:
            import io
            
            xls = pd.ExcelFile(io.BytesIO(file_data))
            
            # Find candidate sheets, then choose the one that actually contains tracker data.
            tower_upper = str(tower or "").upper().strip()
            compact_tower = tower_upper.replace(" ", "")
            tower_tokens = {tower_upper, compact_tower}

            if tower_upper.startswith("EWS TOWER ") or tower_upper.startswith("LIG TOWER "):
                prefix, _, tower_num = tower_upper.partition(" TOWER ")
                tower_tokens.update({
                    f"{prefix} TOWER {tower_num}",
                    f"TOWER {tower_num}",
                    tower_num,
                })
            elif tower_upper:
                tower_tokens.update({
                    f"TOWER {tower_upper}",
                    tower_upper.replace("TOWER ", ""),
                })

            candidate_sheets = []
            for sheet in xls.sheet_names:
                sheet_upper = sheet.upper().strip()
                has_tower = any(token and token in sheet_upper for token in tower_tokens)
                has_finish_context = any(
                    keyword in sheet_upper for keyword in ["FINISH", "TRACKER", "STATUS", "PROGRESS"]
                )

                if has_tower and has_finish_context:
                    candidate_sheets.append(sheet)

            if not candidate_sheets:
                for sheet in xls.sheet_names:
                    sheet_upper = sheet.upper().strip()
                    if any(token and token in sheet_upper for token in tower_tokens):
                        candidate_sheets.append(sheet)

            if not candidate_sheets and len(xls.sheet_names) == 1:
                candidate_sheets = [xls.sheet_names[0]]

            if not candidate_sheets:
                return None

            best_df = None
            best_sheet = None
            best_score = -1

            for sheet_name in candidate_sheets:
                try:
                    df_candidate = pd.read_excel(
                        xls,
                        sheet_name=sheet_name,
                        usecols=lambda c: str(c) in self.columns,
                        engine="openpyxl"
                    )

                    present_columns = sum(1 for col in self.columns if col in df_candidate.columns)
                    non_empty_rows = len(df_candidate.dropna(how='all'))

                    score = present_columns * 1000 + non_empty_rows
                    if 'Activity ID' in df_candidate.columns:
                        valid_ids = pd.to_numeric(df_candidate['Activity ID'], errors='coerce').notna().sum()
                        score += int(valid_ids) * 10

                    if score > best_score:
                        best_score = score
                        best_df = df_candidate
                        best_sheet = sheet_name
                except Exception:
                    continue

            if best_df is None:
                return None

            df = best_df
            logger.info(f"Selected sheet '{best_sheet}' for tower {tower}")
            
            # Convert numeric columns to proper types
            if 'Activity ID' in df.columns:
                df['Activity ID'] = pd.to_numeric(df['Activity ID'], errors='coerce').fillna(0).astype(int)
            
            if '% Complete' in df.columns:
                df['% Complete'] = pd.to_numeric(df['% Complete'], errors='coerce')
                # Vectorized percentage adjustment
                mask = (df['% Complete'] <= 1) & (df['% Complete'].notna())
                df.loc[mask, '% Complete'] *= 100
                df['% Complete'] = df['% Complete'].fillna(0)
            
            # Calculate delays efficiently
            if 'Finish' in df.columns and 'Baseline Finish' in df.columns:
                df['Finish'] = pd.to_datetime(df['Finish'], dayfirst=True, errors='coerce')
                df['Baseline Finish'] = pd.to_datetime(df['Baseline Finish'], dayfirst=True, errors='coerce')
                
                # Vectorized delay calculation
                df['Delay_Days'] = (df['Finish'] - df['Baseline Finish']).dt.days.fillna(0).astype(int)
            else:
                df['Delay_Days'] = 0
            
            # Classify severity using vectorization
            conditions = [
                (df['Delay_Days'] < 0),
                (df['Delay_Days'] == 0),
                (df['Delay_Days'] <= 7),
                (df['Delay_Days'] <= 30),
                (df['Delay_Days'] <= 60)
            ]
            choices = ['Early', 'On-Time', 'Low', 'Medium', 'High']
            # Default is Critical
            df['Severity'] = np.select(conditions, choices, default='Critical')
            
            # Calculate metrics with safe values
            metrics = self.calculate_metrics(df)
            
            # Convert DataFrame to JSON-safe records
            records = self._convert_to_json_safe_records(df)

            # Treat empty/invalid parses as failure so they are not cached as real tower data.
            if not records or ('Activity ID' not in df.columns and 'Activity Name' not in df.columns):
                logger.warning(f"No valid tracker rows found for tower {tower} in sheet '{best_sheet}'")
                return None
            
            return {
                'data': records,
                'metrics': metrics,
                'columns': list(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error pre-processing tower {tower}: {e}")
            return None
    
    def _convert_to_json_safe_records(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to JSON-safe records with improved type handling."""
        # Create a copy to avoid modifying original
        df_safe = df.copy()
        
        # Convert Timestamps to ISO format strings
        for col in df_safe.select_dtypes(include=['datetime', 'datetimetz']).columns:
            df_safe[col] = df_safe[col].apply(lambda x: x.isoformat() if pd.notna(x) else None)
            
        # Replace NaN/Inf/NaT with None
        df_safe = df_safe.replace([np.inf, -np.inf, np.nan, pd.NaT], None)
        
        return df_safe.to_dict(orient='records')
    
    def calculate_delay(self, finish_date, baseline_date):
        """Calculate delay in days."""
        try:
            if pd.isna(finish_date) or pd.isna(baseline_date):
                return 0
            
            finish = pd.to_datetime(finish_date)
            baseline = pd.to_datetime(baseline_date)
            delay_days = (finish - baseline).days
            return int(delay_days) if not pd.isna(delay_days) else 0
        except:
            return 0
    
    def classify_severity(self, delay_days):
        """Classify delay severity."""
        if delay_days is None:
            return "On-Time"
        
        # Ensure delay_days is a number
        try:
            delay = float(delay_days)
        except:
            return "On-Time"
            
        if delay <= 0:
            return "On-Time" if delay == 0 else "Early"
        elif delay <= 7:
            return "Low"
        elif delay <= 30:
            return "Medium"
        elif delay <= 60:
            return "High"
        else:
            return "Critical"
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary metrics with JSON-safe values."""
        try:
            # Ensure Delay_Days exists
            if 'Delay_Days' not in df.columns:
                return {
                    'total_activities': 0,
                    'delayed_count': 0,
                    'on_time_count': 0,
                    'critical_count': 0,
                    'high_count': 0,
                    'medium_count': 0,
                    'low_count': 0,
                    'max_delay': 0,
                    'avg_delay': 0.0
                }
            
            # Ensure Delay_Days is numeric
            if not pd.api.types.is_numeric_dtype(df['Delay_Days']):
                df_delay = pd.to_numeric(df['Delay_Days'], errors='coerce').fillna(0)
            else:
                df_delay = df['Delay_Days']
            
            # Calculate metrics
            total_activities = len(df)
            delayed_mask = df_delay > 0
            delayed_count = int(delayed_mask.sum())
            on_time_count = total_activities - delayed_count
            
            if delayed_count > 0:
                delayed_days = df_delay[delayed_mask]
                max_delay = int(delayed_days.max())
                avg_delay = round(float(delayed_days.mean()), 2)
            else:
                max_delay = 0
                avg_delay = 0.0
            
            # Count severities
            # Use value_counts on existing Severity column if possible
            if 'Severity' in df.columns:
                severity_counts = df['Severity'].value_counts().to_dict()
            else:
                severity_counts = {}
            
            return {
                'total_activities': int(total_activities),
                'delayed_count': delayed_count,
                'on_time_count': int(on_time_count),
                'critical_count': int(severity_counts.get('Critical', 0)),
                'high_count': int(severity_counts.get('High', 0)),
                'medium_count': int(severity_counts.get('Medium', 0)),
                'low_count': int(severity_counts.get('Low', 0)),
                'max_delay': max_delay,
                'avg_delay': avg_delay
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'total_activities': 0,
                'delayed_count': 0,
                'on_time_count': 0,
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
                'max_delay': 0,
                'avg_delay': 0.0
            }


# Initialize data preprocessor
preprocessor = DataPreprocessor()
# --------------------------------------------------
# COS OPERATIONS
# --------------------------------------------------

def normalize_project_token(value: str) -> str:
    """Normalize project names for fuzzy matching."""
    return re.sub(r'[^a-z0-9]+', '', str(value or "").lower())


def get_project_prefixes(project: str) -> List[str]:
    """Resolve user/LLM project names to known COS prefixes."""
    if not project:
        return []

    normalized_project = normalize_project_token(project)

    explicit_prefix_map = {
        "ewslig": ["EWS LIG P4"],
        "ewsligp4": ["EWS LIG P4"],
        "ews": ["EWS LIG P4"],
        "lig": ["EWS LIG P4"],
        "veridia": ["Veridia"],
        "eligo": ["Eligo"],
    }

    if normalized_project in explicit_prefix_map:
        return explicit_prefix_map[normalized_project]

    try:
        res = cos.list_objects_v2(Bucket=COS_BUCKET, Delimiter="/")
        available_prefixes = [
            p["Prefix"].rstrip("/")
            for p in res.get("CommonPrefixes", [])
        ]
    except Exception as e:
        logger.error(f"Error listing prefixes for project resolution: {e}")
        available_prefixes = []

    if not available_prefixes:
        return [project]

    resolved = []
    for prefix in available_prefixes:
        normalized_prefix = normalize_project_token(prefix)

        if normalized_prefix == normalized_project:
            resolved.append(prefix)
            continue

        if normalized_project and normalized_project in normalized_prefix:
            resolved.append(prefix)

    if resolved:
        return list(dict.fromkeys(resolved))

    return [project]

def list_projects_from_cos() -> List[str]:
    """List all projects from COS."""
    try:
        projects = []
        continuation_token = None

        while True:
            kwargs = {
                "Bucket": COS_BUCKET,
                "Delimiter": "/",
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            res = cos.list_objects_v2(**kwargs)
            projects.extend(
                p["Prefix"].rstrip("/")
                for p in res.get("CommonPrefixes", [])
            )

            if not res.get("IsTruncated"):
                break
            continuation_token = res.get("NextContinuationToken")

        return list(dict.fromkeys(projects))
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        return []

def list_project_files(project: str) -> List[Dict]:
    """List all files in a project with metadata."""
    try:
        files = []

        for prefix in get_project_prefixes(project):
            continuation_token = None

            while True:
                kwargs = {
                    "Bucket": COS_BUCKET,
                    "Prefix": f"{prefix}/",
                }
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token

                res = cos.list_objects_v2(**kwargs)

                for obj in res.get("Contents", []):
                    if obj["Key"].endswith("/"):
                        continue

                    filename = obj["Key"].split('/')[-1]
                    if not preprocessor.is_finishing_tracker_file(filename):
                        continue

                    info = preprocessor.extract_tracker_info(
                        filename,
                        project=project,
                        object_key=obj["Key"]
                    )

                    files.append({
                        'key': obj["Key"],
                        'filename': filename,
                        'tower': info['tower'],
                        'date': info['date'],
                        'last_modified': obj.get("LastModified"),
                        'size': obj.get("Size", 0)
                    })

                if not res.get("IsTruncated"):
                    break
                continuation_token = res.get("NextContinuationToken")
        
        return files
    except Exception as e:
        logger.error(f"Error listing files for {project}: {e}")
        return []

def get_latest_tracker_files(project: str) -> Dict[str, Dict]:
    """Get latest tracker file for each tower in a project."""
    files = list_project_files(project)
    latest_trackers = {}
    
    for file_info in files:
        tower = file_info['tower']
        date_str = file_info['date']
        
        if not tower or not date_str:
            continue
        
        if tower not in latest_trackers:
            latest_trackers[tower] = file_info
        else:
            try:
                current_date = datetime.strptime(date_str, "%d-%m-%Y")
                existing_date = datetime.strptime(latest_trackers[tower]['date'], "%d-%m-%Y")
                if current_date > existing_date:
                    latest_trackers[tower] = file_info
            except:
                pass
    
    return latest_trackers

def download_file(key: str) -> bytes:
    """Download file from COS."""
    try:
        obj = cos.get_object(Bucket=COS_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception as e:
        logger.error(f"Error downloading {key}: {e}")
        raise

# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------

class DataLoader:
    """Load data with smart cache management."""
    
    @staticmethod
    def load_project_data(project: str, force_refresh: bool = False) -> Dict:
        """Load project data, checking for new trackers."""
        try:
            latest_trackers = get_latest_tracker_files(project)
            
            if not latest_trackers:
                cached_data = project_cache.get_project_data(project)
                if cached_data:
                    return cached_data
                raise ValueError(f"No tracker files found for project {project}")
            
            towers_data = {}
            
            for tower, file_info in latest_trackers.items():
                try:
                    tracker_version = preprocessor.get_tracker_version(file_info['filename'])
                    cached_tower = project_cache.get_tower_data(project, tower)
                    cached_records = cached_tower.get('data', []) if cached_tower else []
                    cached_columns = cached_tower.get('columns', []) if cached_tower else []
                    cached_invalid = (
                        cached_tower is not None and
                        (
                            not cached_records or
                            ('Activity ID' not in cached_columns and 'Activity Name' not in cached_columns)
                        )
                    )
                    
                    needs_update = force_refresh or cached_invalid or project_cache.needs_update(
                        project, tower, tracker_version
                    )
                    
                    if needs_update:
                        logger.info(f"Loading tower {tower} from COS (new tracker: {tracker_version})")
                        
                        file_data = download_file(file_info['key'])
                        processed = preprocessor.preprocess_excel(file_data, tower, project=project)
                        
                        if processed:
                            project_cache.update_tower_data(
                                project, tower, processed, tracker_version
                            )
                            towers_data[tower] = processed
                        else:
                            cached = project_cache.get_tower_data(project, tower)
                            if cached:
                                towers_data[tower] = cached
                    else:
                        cached = project_cache.get_tower_data(project, tower)
                        if cached:
                            towers_data[tower] = cached
                        else:
                            logger.info(f"Cache miss for {project}-{tower}")
                            file_data = download_file(file_info['key'])
                            processed = preprocessor.preprocess_excel(file_data, tower, project=project)
                            
                            if processed:
                                project_cache.update_tower_data(
                                    project, tower, processed, tracker_version
                                )
                                towers_data[tower] = processed
                            
                except Exception as e:
                    logger.error(f"Error loading tower {tower}: {e}")
            
            if towers_data:
                project_cache.save_to_disk()
                
                project_data = {
                    'towers': towers_data,
                    'available_towers': list(towers_data.keys()),
                    'total_towers': len(towers_data),
                    'last_updated': datetime.now().isoformat()
                }
                project_cache.update_project(project, project_data)
                
                return project_data
            
            cached_data = project_cache.get_project_data(project)
            if cached_data:
                return cached_data
            
            raise ValueError(f"No data could be loaded for project {project}")
            
        except Exception as e:
            logger.error(f"Error loading project {project}: {e}")
            raise
    
    @staticmethod
    def load_tower_data(project: str, tower: str) -> Optional[Dict]:
        """Load data for a specific tower."""
        try:
            cached_data = project_cache.get_tower_data(project, tower)
            if cached_data:
                return cached_data
            
            project_data = DataLoader.load_project_data(project)
            return project_data.get('towers', {}).get(tower)
            
        except Exception as e:
            logger.error(f"Error loading tower {project}-{tower}: {e}")
            return None


def strip_severity_from_response(obj: Any) -> Any:
    """Recursively remove row-level severity fields from API responses."""
    if isinstance(obj, dict):
        return {
            key: strip_severity_from_response(value)
            for key, value in obj.items()
            if key != "Severity"
        }
    if isinstance(obj, list):
        return [strip_severity_from_response(item) for item in obj]
    return obj


def is_ews_lig_project(project: Optional[str]) -> bool:
    """Return True when the project represents the shared EWS/LIG finishing prefix."""
    normalized = normalize_project_token(project or "")
    return normalized in {"ewslig", "ewsligp4"} or "ewslig" in normalized


def format_completed_value(value: Any) -> Optional[str]:
    """Format completion values consistently as percentages for compact responses."""
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.endswith("%"):
            return stripped
        try:
            value = float(stripped)
        except Exception:
            return stripped

    if isinstance(value, (int, float)):
        pct = float(value)
        if 0 < pct <= 1:
            pct *= 100
        if pct.is_integer():
            return f"{int(pct)}%"
        return f"{round(pct, 2)}%"

    return str(value)


def shape_ews_lig_activity_rows(
    rows: List[Dict[str, Any]],
    root_tower: Optional[str],
    include_context_fields: List[str]
) -> List[Dict[str, Any]]:
    """Limit EWS/LIG activity rows to the requested business columns."""
    shaped_rows = []

    for row in rows or []:
        tower_value = row.get("tower") or root_tower
        shaped = {
            "tower": tower_value,
            "activity_name": row.get("activity_name"),
            "baseline_finish": row.get("baseline_finish"),
            "finish_date": row.get("actual_finish") or row.get("finish_date"),
            "delay_days": row.get("delay_days"),
            "completed": format_completed_value(
                row.get("completed", row.get("percent_complete"))
            ),
        }

        for field_name in include_context_fields:
            shaped[field_name] = row.get(field_name)

        shaped_rows.append(shaped)

    return shaped_rows


def shape_special_activity_entry(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Trim special activity rows to tower name and schedule fields only."""
    if not isinstance(entry, dict):
        return entry

    tower_name = entry.get("found_in_tower") or entry.get("tower")
    return {
        "tower": tower_name,
        "baseline_finish": entry.get("baseline_finish"),
        "actual_finish": entry.get("actual_finish") or entry.get("finish_date"),
        "delay_days": entry.get("delay_days"),
    }


def detect_requested_context_fields(
    query_lower: str,
    filters_applied: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Infer which compact context columns should be included in EWS/LIG activity rows."""
    filters_applied = filters_applied or {}
    include_fields = []

    field_rules = [
        ("pour", ["pour"]),
        ("module", ["module"]),
        ("floor", ["floor", "flr", "level", "first", "second", "third", "fourth", "fifth"]),
        ("flat", ["flat", "unit", "apartment"]),
    ]

    for field_name, keywords in field_rules:
        if field_name in filters_applied or any(keyword in query_lower for keyword in keywords):
            include_fields.append(field_name)

    return include_fields


def shape_special_project_response(
    result: Dict[str, Any],
    original_query: str,
    filters_applied: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Apply compact response shaping for project delay queries requested by the user."""
    if not isinstance(result, dict):
        return result

    query_lower = (original_query or "").lower()
    project = result.get("project")
    normalized_project = normalize_project_token(project or "")

    supported_projects = {"ewslig", "ewsligp4", "veridia", "eligo"}
    if normalized_project not in supported_projects and not is_ews_lig_project(project):
        return result

    project_mentioned = False
    if is_ews_lig_project(project):
        project_mentioned = ("ews" in query_lower or "lig" in query_lower)
    elif normalized_project == "veridia":
        project_mentioned = "veridia" in query_lower
    elif normalized_project == "eligo":
        project_mentioned = "eligo" in query_lower

    if not project_mentioned:
        return result

    if "activity_details" not in result:
        return result

    include_context_fields = detect_requested_context_fields(query_lower, filters_applied)
    root_tower = result.get("tower")
    result["activity_details"] = shape_ews_lig_activity_rows(
        result.get("activity_details", []),
        root_tower,
        include_context_fields
    )

    shaped_highlights = None
    for special_key in ["activity_id_1", "activity_id_2"]:
        if special_key in result:
            result[special_key] = shape_special_activity_entry(result.get(special_key))

    if "activity_highlights" in result:
        shaped_highlights = [
            shape_special_activity_entry(item)
            for item in result.get("activity_highlights", [])
            if isinstance(item, dict)
        ]
        result["activity_highlights"] = shaped_highlights

    # For broad EWS/LIG queries like "show me delay in ews" or "show me delay in lig",
    # return only the special activity block rather than the full activity list.
    broad_scope_query = (
        is_ews_lig_project(project)
        and not include_context_fields
        and "tower" not in query_lower
        and "project" not in query_lower
        and ("activity_id_1" in result or "activity_id_2" in result)
    )
    if broad_scope_query:
        result.pop("activity_details", None)
        result.pop("activity_highlights", None)
        result.pop("tower", None)
        result.pop("activity_id_2", None)

    # For broad Veridia/Eligo project queries like "show me delay in veridia project",
    # return only the per-tower special activity details under activity_id_2.
    broad_project_query = (
        normalized_project in {"veridia", "eligo"}
        and
        "project" in query_lower
        and not include_context_fields
        and shaped_highlights is not None
    )
    if broad_project_query:
        result["activity_id_2"] = shaped_highlights
        result.pop("activity_details", None)
        result.pop("activity_highlights", None)
        result.pop("tower", None)

    for extra_key in ["project_summary", "tower_results"]:
        result.pop(extra_key, None)

    return result

# --------------------------------------------------
# QUERY PROCESSOR
# --------------------------------------------------

class QueryProcessor:
    """Process user queries with LLM assistance."""
    
    @staticmethod
    async def process_query(query_request: QueryRequest) -> Dict[str, Any]:
        start_time = time.time()
        
        available_projects = QueryProcessor.get_available_projects()
        
        if not available_projects:
            raise HTTPException(status_code=404, detail="No projects available")
        
        logger.info(f"User query: '{query_request.query}'")
        
        # Define query_lower for pattern detection
        query_lower = query_request.query.lower()
        
        # Check for "all projects" query BEFORE anything else
        is_all_projects_query = "all projects" in query_lower or "all project" in query_lower
        
        # Define tower_patterns here so it's available throughout the method
        tower_patterns = [
            r'tower[-\s]+([a-z0-9]+[a-z]?)',
            r'\btower\s*([a-z0-9]+[a-z]?)\b',
            r'\b([a-z0-9]+[a-z]?)\s+tower\b',
            r'for\s+tower\s+([a-z0-9]+[a-z]?)',  # "for tower 6"
            r'in\s+tower\s+([a-z0-9]+[a-z]?)',    # "in tower 6"
            r'block[-\s]+([a-z0-9]+[a-z]?)',
            r'\bblock\s*([a-z0-9]+[a-z]?)\b',
        ]
        
        # Step 1: Extract specific filters from query BEFORE anything else
        extracted_filters = QueryProcessor._extract_filters_from_query(query_request.query)
        
        # Step 2: Check if this is a flat-specific query
        # If we found a flat number in filters, it's NOT a tower number
        is_flat_query = 'flat' in extracted_filters
        
        # Detect ALL query patterns early
        is_all_towers_query = any(phrase in query_lower for phrase in [
            "all tower", 
            "all towers", 
            "tower wise", 
            "tower-wise",
            "every tower",
            "each tower"
        ])
        
        # If it's an "all projects" query, handle it immediately
        if is_all_projects_query:
            logger.info(f"Processing 'all projects' query")
            return await QueryProcessor._handle_all_projects_analysis(
                available_projects, start_time
            )
        
        # Detect specific query types
        is_activity_wise_query = "activity wise" in query_lower or "activity-wise" in query_lower or "activities" in query_lower
        is_tower_wise_query = ("tower wise" in query_lower or "tower-wise" in query_lower) and not is_all_towers_query
        is_floor_wise_query = "floor wise" in query_lower or "floor-wise" in query_lower
        is_flat_wise_query = "flat wise" in query_lower or "flat-wise" in query_lower
        is_pour_wise_query = "pour wise" in query_lower or "pour-wise" in query_lower
        is_module_wise_query = "module wise" in query_lower or "module-wise" in query_lower
        is_summary_query = "summary" in query_lower or "overview" in query_lower or "summarize" in query_lower
        is_comparison_query = "compare" in query_lower or "comparison" in query_lower
        
        # Step 3: Extract project and tower BEFORE LLM for better accuracy
        detected_projects = []
        detected_tower = None
        
        # Try to extract project names from the query
        for project in available_projects:
            if project.lower() in query_lower:
                detected_projects.append(project)

        if not detected_projects and ("ews" in query_lower or "lig" in query_lower):
            for project in available_projects:
                if is_ews_lig_project(project):
                    detected_projects.append(project)
                    break
        
        # If multiple projects detected, handle as multi-project analysis immediately
        if len(detected_projects) > 1:
            logger.info(f"Detected multiple projects: {detected_projects}")
            return await QueryProcessor._handle_all_projects_analysis(
                detected_projects, start_time
            )
            
        detected_project = detected_projects[0] if detected_projects else None
        
        # IMPORTANT: Don't extract tower if this is a flat query
        # Towers should be 1-3 characters max, flats can be longer
        detected_towers_list = []
        if not is_flat_query:
            detected_towers_set = set()
            for pattern in tower_patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    
                    if match:
                        tower = match.upper().strip()
                        # Skip invalid tower names
                        if tower not in ["IN", "FOR", "ALL", "THE", "AND", "WITH", "BETWEEN", "OF"] and 1 <= len(tower) <= 3:
                            detected_towers_set.add(tower)
            
            if detected_towers_set:
                detected_towers_list = list(detected_towers_set)
                detected_tower = detected_towers_list[0] # Set primary for fallback logic
            
            # Also look for standalone tower numbers like "tower 6" or just "6"
            if not detected_towers_list:
                # Pattern for "tower 6" or "tower6"
                standalone_matches = re.findall(r'tower[-\s]*([0-9]+)', query_lower, re.IGNORECASE)
                for tower_num in standalone_matches:
                    # Towers are usually 1-3 digits
                    if len(tower_num) <= 3:
                        detected_towers_list.append(tower_num.upper())
                
                if detected_towers_list:
                    detected_tower = detected_towers_list[0]
                else:
                    # Look for just a number that might be a tower
                    # But exclude if it's likely a flat number or percentage
                    # Negative lookahead to ensure it's not followed by % or "percent" or "progress"
                    number_matches = re.findall(r'\b([0-9]+)\b(?!\s*(?:%|percent|progress|complete))', query_lower)
                    for potential_tower in number_matches:
                        # Towers are typically 1-3 digits, flats can be longer
                        if len(potential_tower) <= 3:
                            detected_towers_list.append(potential_tower.upper())
                    
                    if detected_towers_list:
                        detected_tower = detected_towers_list[0]
        
        # Step 3: Use LLM for intent detection
        if query_request.use_llm and (WATSONX_URL and WATSONX_API_KEY):
            try:
                intent_response = await llm_service.analyze_query(
                    query_request.query, 
                    available_projects
                )
            except Exception as e:
                logger.error(f"LLM processing failed: {e}")
                intent_response = llm_service._create_default_response()
        else:
            intent_response = llm_service._create_default_response()
        
        # Step 4: Apply extracted filters to intent_response
        if extracted_filters:
            # Merge extracted filters with LLM filters
            intent_response.filters.update(extracted_filters)
            
            # IMPORTANT FIX: If we have status filters (delayed/on_time), override to activity-wise
            if 'status' in extracted_filters:
                logger.info(f"Status filter detected ({extracted_filters['status']}), forcing query_type to ACTIVITY_WISE")
                intent_response.query_type = QueryType.ACTIVITY_WISE
            
            # If we have specific filters (flat, floor, module, pour, activity_id, trade) but query type is not activity-wise,
            # override to activity-wise to show filtered activities
            elif any(key in extracted_filters for key in ['flat', 'floor', 'module', 'pour', 'activity_id', 'trade']):
                if intent_response.query_type != QueryType.ACTIVITY_WISE and intent_response.query_type != QueryType.ACTIVITY_DETAILS:
                    intent_response.query_type = QueryType.ACTIVITY_WISE
            
            # Ensure activity_id from filters is set in intent_response and force intent
            if 'activity_id' in extracted_filters:
                intent_response.activity_id = extracted_filters['activity_id']
                intent_response.intent = UserIntent.GET_ACTIVITY_DETAILS
                intent_response.query_type = QueryType.ACTIVITY_DETAILS

        has_ews_word = re.search(r"\bews\b", query_lower) is not None
        has_lig_word = re.search(r"\blig\b", query_lower) is not None

        if has_ews_word and not has_lig_word:
            intent_response.filters["tower_scope"] = "EWS"
        elif has_lig_word and not has_ews_word:
            intent_response.filters["tower_scope"] = "LIG"
        
        # Step 5: Use direct query type detection from text (highest priority)
        detected_query_type = QueryProcessor._detect_query_type_from_text(query_request.query)
        if detected_query_type and intent_response.query_type != detected_query_type:
            logger.warning(f"Overriding LLM query_type {intent_response.query_type.value} to {detected_query_type.value}")
            intent_response.query_type = detected_query_type
        
        # Step 6: Override query types based on explicit keywords in query (as backup)
        if is_activity_wise_query and intent_response.query_type != QueryType.ACTIVITY_WISE:
            logger.warning(f"Overriding query_type to ACTIVITY_WISE for query: {query_request.query}")
            intent_response.query_type = QueryType.ACTIVITY_WISE
        
        elif is_tower_wise_query and intent_response.query_type != QueryType.TOWER_WISE:
            logger.warning(f"Overriding query_type to TOWER_WISE for query: {query_request.query}")
            intent_response.query_type = QueryType.TOWER_WISE
        
        elif is_floor_wise_query and intent_response.query_type != QueryType.FLOOR_WISE:
            logger.warning(f"Overriding query_type to FLOOR_WISE for query: {query_request.query}")
            intent_response.query_type = QueryType.FLOOR_WISE
        
        elif is_flat_wise_query and intent_response.query_type != QueryType.FLAT_WISE:
            logger.warning(f"Overriding query_type to FLAT_WISE for query: {query_request.query}")
            intent_response.query_type = QueryType.FLAT_WISE
        
        elif is_pour_wise_query and intent_response.query_type != QueryType.POUR_WISE:
            logger.warning(f"Overriding query_type to POUR_WISE for query: {query_request.query}")
            intent_response.query_type = QueryType.POUR_WISE
        
        elif is_module_wise_query and intent_response.query_type != QueryType.MODULE_WISE:
            logger.warning(f"Overriding query_type to MODULE_WISE for query: {query_request.query}")
            intent_response.query_type = QueryType.MODULE_WISE
        
        elif is_summary_query and intent_response.query_type != QueryType.SUMMARY:
            logger.warning(f"Overriding query_type to SUMMARY for query: {query_request.query}")
            intent_response.query_type = QueryType.SUMMARY
        
        elif is_comparison_query and intent_response.query_type != QueryType.COMPARISON:
            logger.warning(f"Overriding query_type to COMPARISON for query: {query_request.query}")
            intent_response.query_type = QueryType.COMPARISON
        
        elif is_all_towers_query and intent_response.query_type != QueryType.MULTI_TOWER:
            logger.warning(f"Overriding query_type to MULTI_TOWER for 'all towers' query")
            intent_response.query_type = QueryType.MULTI_TOWER
        
        # Step 7: Determine project - IMPROVED LOGIC
        project = None
        
        # Priority 1: User specified project
        if query_request.project:
            project = query_request.project
            logger.info(f"Using user-specified project: {project}")
        
        # Priority 2: Project detected from query text (BEFORE LLM)
        elif detected_project:
            project = detected_project
            logger.info(f"Using project detected from query text: {project}")
        
        # Priority 3: LLM detected project
        elif intent_response.project:
            project = intent_response.project
            logger.info(f"Using LLM detected project: {project}")
        
        # Priority 4: Try to infer project based on available towers
        if not project and detected_tower:
            # Try to find which project has this tower
            for proj in available_projects:
                try:
                    project_data = DataLoader.load_project_data(proj)
                    available_towers = project_data.get('available_towers', [])
                    
                    if detected_tower in available_towers:
                        project = proj
                        logger.info(f"Found project '{project}' that has tower '{detected_tower}'")
                        break
                except Exception as e:
                    logger.warning(f"Could not load project {proj} to check towers: {e}")
                    continue
        
        # Priority 5: Use first available project that can be loaded
        if not project and available_projects:
            for candidate in available_projects:
                try:
                    # quick check if we can load it
                    DataLoader.load_project_data(candidate)
                    project = candidate
                    logger.info(f"Using first valid available project: {project}")
                    break
                except Exception as e:
                    logger.warning(f"Skipping project {candidate} as it cannot be loaded: {e}")
            
            if not project:
                logger.warning("No valid projects could be loaded.")
                # Fallback to first one anyway to let it fail with proper error later
                project = available_projects[0]
    
        # Step 8: Handle towers - IMPROVED LOGIC
        towers = []
        
        # If it's an "all towers" query or no specific tower mentioned, use all towers
        if is_all_towers_query or (not detected_tower and not query_request.tower and project):
            # For any query type without a specific tower, analyze all towers in the project
            logger.info(f"No specific tower mentioned, will analyze all towers in project {project}")
            towers = []  # Empty list = all towers
        
        # Priority 1: User specified tower
        elif query_request.tower:
            towers = [query_request.tower]
            logger.info(f"Using user-specified tower: {towers[0]}")
        
        # Priority 2: Tower detected from query text (BEFORE LLM)
        elif detected_towers_list:
            towers = detected_towers_list
            logger.info(f"Using towers detected from query text: {towers}")
        
        # Priority 3: LLM detected towers (after filtering)
        elif intent_response.towers:
            towers = intent_response.towers
            logger.info(f"Using LLM detected towers: {towers}")
        
        # Try to extract tower from query (fallback pattern matching)
        if not towers and not is_all_towers_query:
            found_towers = set()
            for pattern in tower_patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    
                    if match:
                        tower = match.upper().strip()
                        # Skip invalid tower names
                        if tower not in ["IN", "FOR", "ALL", "THE", "AND", "WITH", "BETWEEN"] and 1 <= len(tower) <= 3:
                            found_towers.add(tower)
            
            if found_towers:
                towers = list(found_towers)
                logger.info(f"Found towers in query (fallback): {towers}")
        
        # FIX: For ANY query type without specific tower, use all towers
        if not towers and not is_all_towers_query and project:
            try:
                project_data = DataLoader.load_project_data(project)
                available_towers = project_data.get('available_towers', [])
                if available_towers:
                    # For ANY query type without specific tower, analyze all towers
                    towers = []
                    logger.info(f"No tower specified for {intent_response.query_type.value} query, analyzing all {len(available_towers)} towers")
                else:
                    logger.warning(f"No towers available in project {project}")
                    towers = []
            except Exception as e:
                logger.error(f"Error getting available towers for project {project}: {e}")
                towers = []

        tower_scope = str(intent_response.filters.get("tower_scope", "")).upper()

        # For the shared EWS/LIG project, a query like "ews tower 3" should resolve to
        # the scoped tower name instead of matching either EWS Tower 3 or LIG Tower 3.
        if towers and project and is_ews_lig_project(project) and tower_scope in {"EWS", "LIG"}:
            scoped_towers = []
            for tower in towers:
                tower_text = str(tower or "").strip()
                tower_upper = tower_text.upper()

                if tower_upper.startswith(f"{tower_scope} TOWER "):
                    scoped_towers.append(tower_text)
                    continue

                if tower_upper.startswith("TOWER "):
                    tower_id = tower_upper.replace("TOWER ", "", 1).strip()
                    scoped_towers.append(f"{tower_scope} Tower {tower_id}")
                    continue

                if re.fullmatch(r"[A-Z0-9]{1,3}", tower_upper):
                    scoped_towers.append(f"{tower_scope} Tower {tower_upper}")
                    continue

                scoped_towers.append(tower_text)

            towers = scoped_towers
            logger.info(f"Applied tower scope '{tower_scope}' to requested towers: {towers}")
        
        # Step 9: Validate that tower exists in the selected project
        if towers and project:
            try:
                project_data = DataLoader.load_project_data(project)
                available_towers = project_data.get('available_towers', [])
                candidate_towers = available_towers

                if is_ews_lig_project(project) and tower_scope in {"EWS", "LIG"}:
                    scoped_available_towers = [
                        av_tower for av_tower in available_towers
                        if tower_scope in str(av_tower).upper()
                    ]
                    if scoped_available_towers:
                        candidate_towers = scoped_available_towers
                        logger.info(
                            f"Restricting tower validation to {tower_scope} towers: {candidate_towers}"
                        )
                
                validated_towers = []
                for tower in towers:
                    # Normalization and Matching Logic
                    match_found = False
                    
                    # 1. Direct match
                    if tower in candidate_towers:
                        validated_towers.append(tower)
                        match_found = True
                        continue
                        
                    # 2. Case-insensitive match
                    for av_tower in candidate_towers:
                        if av_tower.lower() == tower.lower():
                            validated_towers.append(av_tower)
                            match_found = True
                            break
                    if match_found: continue

                    # 3. Fuzzy match (Tower X, Block X)
                    # Common prefixes/suffixes
                    candidates = [
                        f"Tower {tower}", f"Tower-{tower}", 
                        f"Block {tower}", f"Block-{tower}",
                        f"{tower} Tower", f"{tower} Block"
                    ]
                    
                    for candidate in candidates:
                        for av_tower in candidate_towers:
                            if av_tower.lower() == candidate.lower():
                                validated_towers.append(av_tower)
                                match_found = True
                                break
                        if match_found: break
                    if match_found: continue
                    
                    # 4. Numeric/Short match (e.g., "6" matches "Tower 6")
                    # Check if 'tower' is a substring of av_tower (carefully)
                    for av_tower in candidate_towers:
                        # Split available tower into parts to avoid partial matches (e.g. "1" matching "Tower 11")
                        parts = re.split(r'[\s\-_]+', av_tower.lower())
                        if tower.lower() in parts:
                            validated_towers.append(av_tower)
                            match_found = True
                            break
                    if match_found: continue

                    if not match_found:
                        logger.warning(f"Tower {tower} not found in project {project}. Available: {candidate_towers}")
                        
                        # If we're using a default tower and it doesn't exist, try the first available tower
                        if tower == "F" and candidate_towers:
                            logger.info(f"Default tower F not found, using first available tower: {candidate_towers[0]}")
                            validated_towers.append(candidate_towers[0])
                        else:
                            # Try to find the correct project for this tower
                            correct_project = None
                            for proj in available_projects:
                                try:
                                    proj_data = DataLoader.load_project_data(proj)
                                    proj_towers = proj_data.get('available_towers', [])
                                    if tower in proj_towers:
                                        correct_project = proj
                                        break
                                except:
                                    continue
                            
                            if correct_project:
                                logger.info(f"Found tower {tower} in project {correct_project}, switching project")
                                project = correct_project
                                # Re-validate in new project (simplified: just add it and hope)
                                validated_towers.append(tower) 
                            else:
                                # If we can't find the tower anywhere, use first available tower in current project
                                if candidate_towers:
                                    # Only add fallback if we don't have ANY valid towers yet
                                    # But here we are iterating list. 
                                    # Let's add it if strictly required or just skip it?
                                    # Original logic replaced towers list.
                                    # We'll stick to original behavior: use first available.
                                    validated_towers.append(candidate_towers[0])
                                    logger.info(f"Tower {tower} not found in any project, using first available: {candidate_towers[0]}")
                
                # Update towers list with validated ones
                if validated_towers:
                    towers = list(dict.fromkeys(validated_towers)) # Remove duplicates
                
            except Exception as e:
                logger.error(f"Error validating tower in project: {e}")
                # Continue anyway, will get error later if tower doesn't exist
        
        # For tower-wise queries, ensure we have a tower
        # FIX: Don't force single tower for tower-wise queries if none specified. 
        # Let it fall through to "all towers" logic in handler.
        # if intent_response.query_type == QueryType.TOWER_WISE and not towers and project:
        #     try:
        #         project_data = DataLoader.load_project_data(project)
        #         available_towers = project_data.get('available_towers', [])
        #         if available_towers:
        #             towers = [available_towers[0]]
        #             logger.info(f"Tower-wise query but no tower specified, using first available: {towers[0]}")
        #     except Exception as e:
        #         logger.error(f"Error getting towers for project {project}: {e}")
        #         towers = ["F"]
        #         logger.warning(f"Tower-wise query but no tower specified, using default: {towers[0]}")
        
        # For activity-wise queries with no tower, analyze all towers
        if intent_response.query_type == QueryType.ACTIVITY_WISE and not towers and project:
            try:
                project_data = DataLoader.load_project_data(project)
                available_towers = project_data.get('available_towers', [])
                if available_towers:
                    # Empty towers list means analyze all towers
                    towers = []
                    logger.info(f"Activity-wise query but no tower specified, analyzing all {len(available_towers)} towers")
                else:
                    towers = []
                    logger.warning(f"No towers available in project {project} for activity-wise query")
            except Exception as e:
                logger.error(f"Error getting towers for project {project}: {e}")
                towers = []
                logger.warning(f"Activity-wise query but no tower specified, analyzing all available towers")
        
        logger.info(f"Final - Project: '{project}', Towers: {towers}, QueryType: {intent_response.query_type.value}, Filters: {intent_response.filters}")
        
        # FIX: Handle case where intent is GET_ACTIVITY_DETAILS but no ID is present
        # This happens when LLM misclassifies "show activities" as "get details"
        if intent_response.intent == UserIntent.GET_ACTIVITY_DETAILS and not intent_response.activity_id:
            logger.warning("Intent is GET_ACTIVITY_DETAILS but no activity_id found - switching to GET_DELAYS")
            intent_response.intent = UserIntent.GET_DELAYS
            
            # Also ensure query type is appropriate for list/analysis
            if intent_response.query_type == QueryType.ACTIVITY_DETAILS:
                intent_response.query_type = QueryType.ACTIVITY_WISE

        # Step 10: Handle different intents
        if intent_response.intent == UserIntent.LIST_PROJECTS:
            return await QueryProcessor._handle_list_projects()
        
        elif intent_response.intent == UserIntent.LIST_TOWERS:
            project_for_towers = project or available_projects[0]
            return await QueryProcessor._handle_list_towers(project_for_towers)
        
        elif intent_response.intent == UserIntent.GET_ACTIVITY_DETAILS:
            return await QueryProcessor._handle_activity_details(
                intent_response, available_projects, project
            )
        
        else:
            # For delay analysis queries
            return await QueryProcessor._handle_delay_analysis(
                intent_response, available_projects, project, towers, start_time
            )
        
    
    
    @staticmethod
    def _detect_query_type_from_text(query: str) -> Optional[QueryType]:
        """Detect query type directly from query text."""
        query_lower = query.lower()
        
        query_type_patterns = {
            QueryType.ACTIVITY_WISE: ["activity wise", "activity-wise", "activities", "list activities", "show activities"],
            QueryType.TOWER_WISE: ["tower wise", "tower-wise"],
            QueryType.FLOOR_WISE: ["floor wise", "floor-wise"],
            QueryType.FLAT_WISE: ["flat wise", "flat-wise"],
            QueryType.POUR_WISE: ["pour wise", "pour-wise"],
            QueryType.MODULE_WISE: ["module wise", "module-wise"],
            QueryType.MULTI_TOWER: ["all tower", "all towers", "multi tower", "multi-tower"],
            QueryType.SUMMARY: ["summary", "overview", "summarize", "overall status", "project status"],
            QueryType.COMPARISON: ["compare", "comparison"],
        }
        
        for query_type, patterns in query_type_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    logger.info(f"Detected query type '{query_type.value}' from pattern: {pattern}")
                    return query_type
        
        logger.info(f"No pattern matched for: {query_lower}")
        logger.info(f"Available patterns for SUMMARY: {query_type_patterns.get(QueryType.SUMMARY)}")
        
        # Detect "all projects" query
        if "all projects" in query_lower or "all project" in query_lower:
            logger.info(f"Detected 'all projects' query")
            return QueryType.MULTI_TOWER  # Or create a new query type for multi-project analysis
        
        # If no specific query type pattern found, check for specific filters
        # If query mentions specific flat/floor/module/pour numbers, it's activity-wise with filters
        # We use _extract_filters_from_query to be robust (handles "5 flr", "#101", etc.)
        filters = QueryProcessor._extract_filters_from_query(query)
        if any(k in filters for k in ['flat', 'floor', 'module', 'pour', 'activity_id']):
             logger.info(f"Detected specific filter query (filters found: {list(filters.keys())}), defaulting to activity-wise")
             return QueryType.ACTIVITY_WISE
        
        return None
    
    @staticmethod
    def _extract_filters_from_query(query: str) -> Dict[str, Any]:
        """Extract filters from query text."""
        filters = {}
        query_lower = query.lower()
        
        # Helper to mask matches to prevent double-counting
        def mask_match(text, match):
            start, end = match.span()
            return text[:start] + ' ' * (end - start) + text[end:]
            
        # 0. Mask Tower patterns first (to prevent Tower 6 -> Module 6)
        # We don't extract tower here (it's done in process_query), but we must mask it
        tower_patterns = [
            r'tower[-\s]+([a-z0-9]+[a-z]?)',
            r'\btower\s*([a-z0-9]+[a-z]?)\b',
            r'\b([a-z0-9]+[a-z]?)\s+tower\b',
            r'for\s+tower\s+([a-z0-9]+[a-z]?)',
            r'in\s+tower\s+([a-z0-9]+[a-z]?)',
            r'block[-\s]+([a-z0-9]+[a-z]?)',
            r'\bblock\s*([a-z0-9]+[a-z]?)\b',
        ]
        
        for pattern in tower_patterns:
            # Use finditer to mask ALL tower occurrences
            matches = list(re.finditer(pattern, query_lower, re.IGNORECASE))
            for match in matches:
                query_lower = mask_match(query_lower, match)
        
        # Extract top/bottom N (BEFORE other number extractions)
        top_pattern = r'top\s+(\d+)'
        match = re.search(top_pattern, query_lower)
        if match:
            filters['limit'] = int(match.group(1))
            filters['sort_order'] = 'desc' # Descending delay
            logger.info(f"Extracted limit filter: top {filters['limit']}")
            query_lower = mask_match(query_lower, match)
        
        # Extract Progress Conditions (moved up to avoid module/tower conflicts)
        progress_patterns = [
            (r'less\s+than\s+(\d+)%?', 'lt'),
            (r'<\s*(\d+)%?', 'lt'),
            (r'more\s+than\s+(\d+)%?', 'gt'),
            (r'>\s*(\d+)%?', 'gt'),
            (r'zero\s+progress', 'zero'),
            (r'no\s+progress', 'zero'),
            (r'not\s+started', 'zero'),
            (r'(\d+)%?\s*complete', 'eq'),
            (r'(\d+)%?\s*progress', 'eq')
        ]
        
        for pattern, op in progress_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters['progress_op'] = op
                if op == 'zero':
                    filters['progress_val'] = 0
                else:
                    filters['progress_val'] = int(match.group(1))
                logger.info(f"Extracted progress filter: {op} {filters['progress_val']}")
                query_lower = mask_match(query_lower, match) # Mask progress
                break
        
        # Extract flat number (e.g., "Flat 113", "flat 113", "Flat No 113", "Flat No. 113")
        flat_patterns = [
            r'flat[-\s]*no[\.\s]*(\d+)',
            r'flat[-\s]*(\d+)',
            r'flat\s+(\d+)',
            r'\bflat\s*(\d+)\b',
            r'unit[-\s]*(\d+)',  # Also check for "Unit 113"
            r'apartment[-\s]*(\d+)',  # Also check for "Apartment 113"
            r'#(\d+)', # Check for #101
            r'no[\.\s]*(\d+)', # Check for No. 101 (careful, might match other things)
        ]
        
        for pattern in flat_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # Validation: if just "No. 101", make sure it's not "Tower No. 1" (though tower is masked)
                filters['flat'] = match.group(1)
                logger.info(f"Extracted flat filter: {filters['flat']}")
                query_lower = mask_match(query_lower, match) # Mask flat to avoid module detection
                break
        
        # Extract floor number
        floor_patterns = [
            r'\b(\d+)(?:st|nd|rd|th)?\s*floor\b',
            r'floor[-\s]*(\d+)',
            r'\bfloor\s+(\d+)\b',
            r'level[-\s]*(\d+)',
            r'\blevel\s+(\d+)\b',
            r'\b(\d+)(?:st|nd|rd|th)?\s*level\b',
            r'\b(\d+)f\b',
            r'\bfloor[-\s]*(\d+)f\b',
            r'\b(\d+)\s*f\b',
            r'\bl(\d+)\b', # L1, L2 etc.
            r'\b(\d+)\s*flr\b', # 5 flr
        ]
        
        for pattern in floor_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                floor_num = match.group(1)
                filters['floor'] = floor_num
                logger.info(f"Extracted floor filter: {filters['floor']}")
                
                full_match = re.search(r'\b(\d+(?:st|nd|rd|th)?\s*(?:floor|level|f))\b', query_lower, re.IGNORECASE)
                if full_match:
                    filters['floor_raw'] = full_match.group(1).upper()
                    logger.info(f"Extracted floor raw format: {filters['floor_raw']}")
                
                query_lower = mask_match(query_lower, match) # Mask floor
                break
        
        # Extract activity ID (BEFORE Module to prevent "Activity 2" -> "Module 2")
        activity_patterns = [
            r'activity[-\s]*id[-\s]*(\d+)',
            r'activity[-\s]*(\d+)',
            r'activity\s+id\s+(\d+)',
            r'activity\s+(\d+)',
        ]
        
        for pattern in activity_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                filters['activity_id'] = match.group(1)
                logger.info(f"Extracted activity_id filter: {filters['activity_id']}")
                query_lower = mask_match(query_lower, match) # Mask activity ID
                break

        # Extract Date Comparison (before/after/by) - Moved BEFORE Module extraction to avoid year as module
        months_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
            'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }
        
        # Sort by length descending to match longer names first
        sorted_keys = sorted(months_map.keys(), key=len, reverse=True)
        month_regex = r'\b(' + '|'.join(sorted_keys) + r')\b'
        
        # Look for "before Feb 2025" or "before Feb"
        date_op_patterns = [
            (r'before\s+' + month_regex + r'(?:\s+(\d{4}))?', 'lt'),
            (r'by\s+' + month_regex + r'(?:\s+(\d{4}))?', 'lt'),
            (r'until\s+' + month_regex + r'(?:\s+(\d{4}))?', 'lt'),
            (r'after\s+' + month_regex + r'(?:\s+(\d{4}))?', 'gt'),
            (r'starting\s+from\s+' + month_regex + r'(?:\s+(\d{4}))?', 'gt')
        ]
        
        for pattern, op in date_op_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                month_str = match.group(1).lower()
                year_str = match.group(2)
                
                month_num = months_map[month_str]
                current_year = datetime.now().year
                year_num = int(year_str) if year_str else current_year
                
                filters['date_op'] = op
                filters['date_val'] = datetime(year_num, month_num, 1)
                
                logger.info(f"Extracted date filter: {op} {year_num}-{month_num}-01")
                query_lower = mask_match(query_lower, match)
                break

        # Extract Date/Time Period (Simple Month)
        months = {
            'jan': 1, 'january': 1, 
            'feb': 2, 'february': 2, 
            'mar': 3, 'march': 3, 
            'apr': 4, 'april': 4, 
            'may': 5, 
            'jun': 6, 'june': 6, 
            'jul': 7, 'july': 7, 
            'aug': 8, 'august': 8, 
            'sep': 9, 'september': 9, 
            'oct': 10, 'october': 10, 
            'nov': 11, 'november': 11, 
            'dec': 12, 'december': 12
        }
        
        for month_name, month_num in months.items():
            match = re.search(r'\b' + month_name + r'\b', query_lower, re.IGNORECASE)
            if match:
                filters['month'] = month_num
                logger.info(f"Extracted month filter: {month_name} ({month_num})")
                query_lower = mask_match(query_lower, match) # Mask month
                break
        
        if 'this month' in query_lower:
            filters['date_period'] = 'current_month'
        elif 'last month' in query_lower:
            filters['date_period'] = 'last_month'
        elif 'next month' in query_lower:
            filters['date_period'] = 'next_month'

        # Mask Tower to avoid misinterpreting tower number as module
        # We don't extract tower here (process_query does that), but we mask it
        # to prevent "Tower 6" from becoming module="6"
        tower_mask_patterns = [
            r'tower[-\s]+([a-z0-9]+[a-z]?)',
            r'\btower\s*([a-z0-9]+[a-z]?)\b',
            r'\b([a-z0-9]+[a-z]?)\s+tower\b',
        ]
        for pattern in tower_mask_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # Just mask, don't store in filters
                query_lower = mask_match(query_lower, match)
                logger.info(f"Masked tower pattern to protect module extraction")
                break

        # Extract module
        module_patterns = [
            r'module[-\s]*([a-z]?\d+[a-z]?)',  # Allows letter first (like G3) or just number
            r'\bmodule\s+([a-z]?\d+[a-z]?)\b',
            r'\bm?([a-z]?\d+[a-z]?)\b',  # Allows optional M prefix
            r'\bm[-\s]*([a-z]?\d+[a-z]?)\b',
            r'\b([a-z]?\d+[a-z]?)\s*module\b',
            r'\b([a-z]\d+)\b',  # Specifically for patterns like G3, A5, etc.
            r'\b(g\d+)\b',  # Specifically for G3, G4, etc. (case-insensitive)
        ]
        
        for pattern in module_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # IMPORTANT: If it's a bare number (pattern 3), ensure it's not a leftover part of something else
                # or verify it looks like a module
                module_value = match.group(1).upper()
                filters['module'] = module_value
                logger.info(f"Extracted module filter: {filters['module']}")
                
                full_match = re.search(r'\b(?:module[-\s]*)?(m?\d+[a-z]?)\b', query_lower, re.IGNORECASE)
                if full_match:
                    module_raw = full_match.group(1).upper()
                    if module_raw.startswith('M'):
                        filters['module_raw'] = module_raw
                    else:
                        filters['module_raw'] = f"M{module_raw}"
                    logger.info(f"Extracted module raw format: {filters['module_raw']}")
                
                query_lower = mask_match(query_lower, match) # Mask module
                break
        
        # Extract pour
        pour_patterns = [
            r'pour[-\s]*(\d+)',
            r'pour\s+(\d+)',
            r'\bpour\s*(\d+)\b',
        ]
        
        for pattern in pour_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                filters['pour'] = match.group(1)
                logger.info(f"Extracted pour filter: {filters['pour']}")
                query_lower = mask_match(query_lower, match) # Mask pour
                break

        # Extract Trade/Activity Type
        trade_keywords = {
            'plaster': ['plaster', 'plastering'],
            'paint': ['paint', 'painting', 'painter'],
            'tile': ['tile', 'tiling', 'tiler'],
            'electric': ['electric', 'electrical', 'wiring'],
            'plumbing': ['plumbing', 'plumber', 'pipe'],
            'flooring': ['flooring', 'floor finish'],
            'finish': ['finish work', 'finishing work'],
            'block': ['block', 'block work', 'brick'],
            'door': ['door', 'frame'],
            'window': ['window', 'glazing'],
            'kitchen': ['kitchen', 'cabinet'],
            'waterproofing': ['waterproofing', 'water proofing']
        }
        
        for trade, keywords in trade_keywords.items():
            if any(k in query_lower for k in keywords):
                filters['trade'] = trade
                logger.info(f"Extracted trade filter: {filters['trade']}")
                break

        # Progress Conditions extraction moved up

        # Extract Aggregation Intent (Which floor, which area, etc.)
        agg_match = re.search(r'which\s+(floor|tower|module|flat|area|activity|trade)', query_lower)
        if agg_match:
            entity = agg_match.group(1)
            filters['aggregate_by'] = entity
            filters['sort_order'] = 'desc' # Implicitly looking for "most/highest" usually
            if not filters.get('limit'):
                filters['limit'] = 5 # Default to top 5 if not specified
            logger.info(f"Extracted aggregation: by {entity}")

        # Activity ID extraction moved up

        # Extract activity Name (including typo handling for 'activtiy')
        # We look for 'activity' or 'activtiy' followed by text that is NOT 'id', 'wise', or just a number
        # We capture until the end of the string or a stop word (like 'in', 'for' if they start a new clause)
        activity_name_pattern = r'\bactiv(?:ity|tiy)\s+(?!id\b)(?!wise\b)(?!\d+\b)(.*?)(?:\s+(?:in|on|at|for|with|details|status|delay|delays|is|are)\b|$)'
        match = re.search(activity_name_pattern, query_lower, re.IGNORECASE)
        if match:
             potential_name = match.group(1).strip()
             
             # Clean masked parts
             potential_name = re.sub(r'MASKED_MATCH_\d+', '', potential_name).strip()
             
             if potential_name:
                 filters['activity_name'] = potential_name
                 logger.info(f"Extracted activity_name filter: {filters['activity_name']}")

        # Extract status filters (delayed, on time, on-time, completed)
        # Check for "delay" which covers "delay", "delayed", "delays"
        # Also include "delya" (common typo from user request)
        if any(w in query_lower for w in ['delay', 'delya', 'behind', 'late', 'stuck', 'slow']):
            filters['status'] = 'delayed'
            logger.info(f"Extracted status filter: delayed (from keyword)")
        elif any(w in query_lower for w in ['on time', 'on-time', 'ontime', 'completed', 'done', 'finished', 'complete']):
            filters['status'] = 'on_time'
            logger.info(f"Extracted status filter: on_time")
        elif any(w in query_lower for w in ['in progress', 'ongoing', 'running', 'started', 'working']):
            filters['status'] = 'in_progress'
            logger.info(f"Extracted status filter: in_progress")
        
        return filters
    
    @staticmethod
    def _filter_activity_details(activities: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Filter activity details based on filters with flexible matching."""
        if not filters:
            return activities
        
        filtered_activities = []
        
        for activity in activities:
            include = True
            
            # Apply flat filter
            if 'flat' in filters:
                activity_flat = str(activity.get('flat', '')).strip()
                filter_flat = str(filters['flat']).strip()
                if activity_flat != filter_flat:
                    include = False
            
            # Apply floor filter with flexible matching
            if include and 'floor' in filters:
                activity_floor = str(activity.get('floor', '')).strip().upper()
                filter_floor = str(filters['floor']).strip()
                
                # Try different matching strategies
                floor_match = False
                
                # 1. Direct numeric match
                if filter_floor.isdigit():
                    activity_floor_nums = re.findall(r'\d+', activity_floor)
                    if activity_floor_nums and activity_floor_nums[0] == filter_floor:
                        floor_match = True
                    elif activity_floor == filter_floor:
                        floor_match = True
                
                # 2. Raw format match
                if not floor_match and 'floor_raw' in filters:
                    if activity_floor == filters['floor_raw']:
                        floor_match = True
                
                # 3. Normalize and compare
                if not floor_match:
                    normalized_activity = re.sub(r'[^A-Z0-9]', '', activity_floor)
                    if filter_floor.isdigit():
                        if normalized_activity.startswith(filter_floor):
                            floor_match = True
                
                if not floor_match:
                    include = False
            
            # Apply module filter with flexible matching
            if include and 'module' in filters:
                activity_module = str(activity.get('module', '')).strip().upper()
                filter_module = str(filters['module']).strip().upper()
                
                module_match = False
                
                # 1. Direct match
                if activity_module == filter_module:
                    module_match = True
                
                # 2. Match with M prefix
                if not module_match:
                    if filter_module.isdigit() and activity_module == f"M{filter_module}":
                        module_match = True
                    elif activity_module.isdigit() and filter_module == f"M{activity_module}":
                        module_match = True
                
                # 3. Raw format match
                if not module_match and 'module_raw' in filters:
                    if activity_module == filters['module_raw']:
                        module_match = True
                
                # 4. Normalize and compare
                if not module_match:
                    normalized_activity = re.sub(r'[^A-Z0-9]', '', activity_module)
                    normalized_filter = re.sub(r'[^A-Z0-9]', '', filter_module)
                    
                    if not normalized_activity.startswith('M'):
                        normalized_activity = f"M{normalized_activity}"
                    if not normalized_filter.startswith('M'):
                        normalized_filter = f"M{normalized_filter}"
                    
                    if normalized_activity == normalized_filter:
                        module_match = True
                
                if not module_match:
                    include = False
            
            # Apply pour filter
            if include and 'pour' in filters:
                activity_pour = str(activity.get('pour', '')).strip()
                filter_pour = str(filters['pour']).strip()
                if activity_pour != filter_pour:
                    include = False
            
            # Apply activity_id filter
            if include and 'activity_id' in filters:
                activity_id = str(activity.get('activity_id', '')).strip()
                filter_activity_id = str(filters['activity_id']).strip()
                if activity_id != filter_activity_id:
                    # logger.info(f"Filtering out activity_id {activity_id} != {filter_activity_id}")
                    include = False
                else:
                    logger.info(f"Match found for activity_id {activity_id}")
            
            # Apply activity_name filter (fuzzy/substring match)
            if include and 'activity_name' in filters:
                activity_name = str(activity.get('activity_name', '')).strip().lower()
                filter_name = str(filters['activity_name']).strip().lower()
                
                # Simple substring match
                if filter_name not in activity_name:
                    include = False
            
            # NEW: Apply status filter (delayed/on_time)
            if include and 'status' in filters:
                delay_days = activity.get('delay_days', 0)
                
                # Ensure delay_days is a number
                if isinstance(delay_days, str):
                    try:
                        delay_days = int(float(delay_days))
                    except:
                        delay_days = 0
                elif not isinstance(delay_days, (int, float)):
                    delay_days = 0
                
                if filters['status'] == 'delayed':
                    if delay_days <= 0:  # Not delayed
                        include = False
                elif filters['status'] == 'on_time':
                    if delay_days > 0:  # Delayed, not on time
                        include = False
            
            # Apply trade filter
            if include and 'trade' in filters:
                activity_name = str(activity.get('activity_name', '')).lower()
                activity_trade = str(activity.get('trade', '')).lower()
                trade = filters['trade']
                # Define keywords mapping (should match extraction)
                trade_keywords = {
                    'plaster': ['plaster', 'plastering'],
                    'paint': ['paint', 'painting', 'painter'],
                    'tile': ['tile', 'tiling', 'tiler'],
                    'electric': ['electric', 'electrical', 'wiring'],
                    'plumbing': ['plumbing', 'plumber', 'pipe'],
                    'flooring': ['flooring', 'floor finish'],
                    'finish': ['finish work', 'finishing work'],
                    'block': ['block', 'block work', 'brick'],
                    'door': ['door', 'frame'],
                    'window': ['window', 'glazing'],
                    'kitchen': ['kitchen', 'cabinet'],
                    'waterproofing': ['waterproofing', 'water proofing']
                }
                keywords = trade_keywords.get(trade, [trade])
                
                # Check both activity name and trade field
                match_name = any(k in activity_name for k in keywords)
                match_trade = any(k in activity_trade for k in keywords) or (trade in activity_trade)
                
                if not (match_name or match_trade):
                    include = False

            # Apply progress filter
            if include and 'progress_op' in filters:
                pct = activity.get('%_complete') or activity.get('percent_complete') or 0
                # Handle potential string format "50%" or None
                if pct is None: pct = 0
                if isinstance(pct, str):
                    try: pct = float(pct.replace('%', '').strip())
                    except: pct = 0
                elif not isinstance(pct, (int, float)):
                    pct = 0
                
                op = filters['progress_op']
                val = filters['progress_val']
                
                if op == 'lt' and not (pct < val): include = False
                elif op == 'gt' and not (pct > val): include = False
                elif op == 'eq' and not (abs(pct - val) < 1.0): include = False 
                elif op == 'zero' and not (pct == 0): include = False

            # Apply date filter
            if include and ('month' in filters or 'date_period' in filters or 'date_op' in filters):
                finish_date_str = str(activity.get('actual_finish') or activity.get('finish') or activity.get('end_date') or '').strip()
                parsed_date = None
                
                if finish_date_str and finish_date_str.lower() != 'none':
                    # Try parsing date
                    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d', '%d-%b-%y', '%d-%b-%Y']:
                        try:
                            parsed_date = datetime.strptime(finish_date_str[:10], fmt)
                            break
                        except: continue
                
                if parsed_date:
                    if 'month' in filters and parsed_date.month != filters['month']:
                        include = False
                    
                    if 'date_period' in filters:
                        now = datetime.now()
                        if filters['date_period'] == 'current_month':
                            if not (parsed_date.month == now.month and parsed_date.year == now.year):
                                include = False
                        elif filters['date_period'] == 'last_month':
                            last_month = now.month - 1 if now.month > 1 else 12
                            last_year = now.year if now.month > 1 else now.year - 1
                            if not (parsed_date.month == last_month and parsed_date.year == last_year):
                                include = False
                    
                    if 'date_op' in filters:
                        op = filters['date_op']
                        val = filters['date_val']
                        if op == 'lt' and not (parsed_date < val): include = False
                        elif op == 'gt' and not (parsed_date > val): include = False
                else:
                    # If we can't parse date but have date filter, usually safe to exclude
                    include = False

            if include:
                filtered_activities.append(activity)
        
        # Apply sorting and limiting if requested
        if 'limit' in filters:
            # Default to sorting by delay_days descending for "top" queries
            reverse = filters.get('sort_order', 'desc') == 'desc'
            
            # Helper to get delay days safely
            def get_delay(x):
                d = x.get('delay_days', 0)
                if isinstance(d, (int, float)):
                    return d
                return 0

            # Sort by delay_days
            filtered_activities.sort(key=get_delay, reverse=reverse)
            
            # Apply limit
            limit = filters['limit']
            filtered_activities = filtered_activities[:limit]
            
        return filtered_activities
    
    
    

    
    @staticmethod
    async def _handle_list_projects() -> Dict[str, Any]:
        """Handle list projects intent."""
        projects = QueryProcessor.get_available_projects()
        return {
            "status": "success",
            "intent": "list_projects",
            "projects": projects,
            "total": len(projects),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    async def _handle_list_towers(project: str) -> Dict[str, Any]:
        """Handle list towers intent."""
        try:
            project_data = DataLoader.load_project_data(project)
            towers = project_data.get('available_towers', [])
            
            return {
                "status": "success",
                "intent": "list_towers",
                "project": project,
                "towers": towers,
                "total": len(towers),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @staticmethod
    async def _handle_activity_details(
        intent_response: IntentResponse,
        available_projects: List[str],
        user_project: Optional[str]
    ) -> Dict[str, Any]:
        """Handle activity details intent."""
        project = user_project or intent_response.project or available_projects[0]
        
        if not intent_response.activity_id:
            raise HTTPException(status_code=400, detail="Activity ID not specified")
        
        project_data = DataLoader.load_project_data(project)
        activity_details = []
        
        for tower, tower_data in project_data.get('towers', {}).items():
            # The data is already cleaned by DataPreprocessor
            for record in tower_data.get('data', []):
                try:
                    record_id = record.get('Activity ID') or record.get('activity_id')
                    target_id = int(intent_response.activity_id)
                    
                    if record_id == target_id:
                        activity_details.append({
                            'project': project,
                            'tower': tower,
                            'activity_id': record_id, # Use the ID from record (usually int)
                            'activity_name': record.get('Activity Name') or record.get('activity_name'),
                            'floor': record.get('Floor') or record.get('floor'),
                            'flat': record.get('Flat') or record.get('flat'),
                            'module': record.get('Module') or record.get('module'),
                            'pour': record.get('Pour') or record.get('pour'),
                            'baseline_finish': QueryProcessor._format_date(record.get('Baseline Finish') or record.get('baseline_finish')),
                            'actual_finish': QueryProcessor._format_date(record.get('Finish') or record.get('end_date')),
                            'delay_days': record.get('Delay_Days') or record.get('delay_days'),
                            'percent_complete': record.get('% Complete') or record.get('%_complete'),
                            'status': 'Delayed' if (record.get('Delay_Days') or record.get('delay_days', 0)) > 0 else 'On Time'
                        })
                except (ValueError, KeyError, TypeError):
                    continue
        
        if not activity_details:
            raise HTTPException(
                status_code=404,
                detail=f"Activity ID {intent_response.activity_id} not found in project {project}"
            )
        
        return {
            "status": "success",
            "intent": "get_activity_details",
            "activity_id": intent_response.activity_id,
            "project": project,
            "activity_details": activity_details,
            "total_found": len(activity_details),
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    async def _handle_multi_project_analysis(
        intent_response: IntentResponse,
        available_projects: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle multi-project analysis intent."""
        project = intent_response.project or available_projects[0]

        project_data = DataLoader.load_project_data(project)
        multi_project_results = []

        for tower, tower_data in project_data.get('towers', {}).items():
            # The data is already cleaned by DataPreprocessor
            tower_analysis = await QueryProcessor._analyze_tower_delays(
                tower_data, intent_response.filters
            )
            multi_project_results.append({
                "tower": tower,
                "analysis": tower_analysis
            })

        return {
            "status": "success",
            "intent": "multi_project_analysis",
            "project": project,
            "results": multi_project_results,
            "total_found": len(multi_project_results),
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    async def _handle_all_projects_analysis(
        available_projects: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle analysis for all projects."""
        all_results = []
        total_summary = {
            "total_projects": 0,
            "total_towers": 0,
            "total_activities": 0,
            "total_delayed": 0
        }
        
        for project in available_projects:
            try:
                project_data = DataLoader.load_project_data(project)
                towers_data = project_data.get('towers', {})
                
                project_summary = {
                    "project": project,
                    "towers_analyzed": len(towers_data),
                    "total_activities": 0,
                    "delayed_activities": 0,
                    "towers": []
                }
                
                for tower, tower_data in towers_data.items():
                    # Get tower summary
                    tower_summary = QueryProcessor._create_tower_summary(tower_data, tower)
                    
                    # Get activity ID 2 for this tower
                    activity_id_2_info = QueryProcessor._get_activity_id_2_for_tower(tower_data, tower)
                    
                    project_summary["towers"].append({
                        "tower": tower,
                        "summary": tower_summary,
                        "activity_id_2": activity_id_2_info
                    })
                    
                    # Update project totals
                    project_summary["total_activities"] += tower_summary.get("total_activities", 0)
                    project_summary["delayed_activities"] += tower_summary.get("delayed_count", 0)
                
                # Update overall totals
                total_summary["total_projects"] += 1
                total_summary["total_towers"] += project_summary["towers_analyzed"]
                total_summary["total_activities"] += project_summary["total_activities"]
                total_summary["total_delayed"] += project_summary["delayed_activities"]
                
                all_results.append(project_summary)
                
            except Exception as e:
                logger.error(f"Error analyzing project {project}: {e}")
                continue
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "intent": "multi_project_analysis",
            "timestamp": datetime.now().isoformat(),
            "query_type": "multi-project",
            "total_summary": total_summary,
            "processing_time_ms": round(float(processing_time), 2),
            "projects_analyzed": all_results,
            "query_info": {
                "note": "Analysis of all projects"
            }
        }

    @staticmethod
    def _format_date(value: Any) -> Optional[str]:
        """Format date to dd-mm-yyyy string."""
        if value is None:
            return None
        
        try:
            # If it's already a datetime or Timestamp
            if hasattr(value, 'strftime'):
                return value.strftime("%d-%m-%Y")
            
            # If it's a string, try to parse
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
                
                # Remove time portion if present
                if "T" in value:
                    value = value.split("T")[0]
                if " " in value:
                    value = value.split(" ")[0]
                
                # Try common formats
                for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"]:
                    try:
                        dt = datetime.strptime(value, fmt)
                        return dt.strftime("%d-%m-%Y")
                    except ValueError:
                        continue
            
            return None
        except Exception:
            return None

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by replacing NaN/Inf values with None."""
        # Replace NaN, NaT, Inf, -Inf with None
        df_clean = df.where(pd.notnull(df), None)
        
        # For numeric columns, ensure values are JSON serializable
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Replace any remaining NaN/Inf
                df_clean[col] = df_clean[col].apply(
                    lambda x: None if pd.isna(x) or (isinstance(x, float) and not np.isfinite(x)) else x
                )
          
        return df_clean
    
    @staticmethod
    def _get_special_activity_id_for_tower_name(tower: str) -> int:
        """Use Activity ID 1 for EWS/LIG towers and Activity ID 2 for other projects."""
        tower_upper = str(tower or "").upper()
        if "EWS" in tower_upper or "LIG" in tower_upper:
            return 1
        return 2

    @staticmethod
    def _get_special_activity_key_for_tower_name(tower: str) -> str:
        """Return the response key name for the tower's special activity."""
        special_id = QueryProcessor._get_special_activity_id_for_tower_name(tower)
        return f"activity_id_{special_id}"

    @staticmethod
    def _get_activity_id_2_for_tower(tower_data: Dict, tower: str) -> Optional[Dict[str, Any]]:
        """Get details for the tower's special activity ID with improved matching."""
        try:
            target_activity_id = QueryProcessor._get_special_activity_id_for_tower_name(tower)

            # Data is already cleaned by DataPreprocessor
            for record in tower_data.get('data', []):
                # Get activity ID from record with flexible parsing
                # Try multiple possible keys for Activity ID
                activity_id_raw = record.get('Activity ID') or record.get('activity_id')
                activity_id = None
                
                # Handle different possible formats
                if isinstance(activity_id_raw, (int, float)):
                    activity_id = int(activity_id_raw)
                elif isinstance(activity_id_raw, str):
                    try:
                        # Try to convert string to int
                        activity_id = int(float(activity_id_raw))
                    except:
                        try:
                            # Try direct string conversion
                            activity_id = int(activity_id_raw.strip())
                        except:
                            activity_id = None
                
                # Check if activity ID matches the tower-specific special ID
                if activity_id == target_activity_id:
                    # FIX: Also check activity name to ensure it's the right record
                    activity_name = record.get('Activity Name', '').strip()
                    
                    # Debug logging
                    logger.info(f"Found Activity ID {target_activity_id} in tower {tower}: {activity_name}")
                    
                    # FIX: Convert delay_days to integer safely
                    delay_days_raw = record.get('Delay_Days', 0)
                    delay_days = 0
                    
                    # Handle string, int, float, or None values
                    if isinstance(delay_days_raw, (int, float)):
                        delay_days = int(delay_days_raw)
                    elif isinstance(delay_days_raw, str):
                        try:
                            delay_days = int(float(delay_days_raw))
                        except:
                            delay_days = 0
                    elif delay_days_raw is None:
                        delay_days = 0
                    
                    # FIX: Determine status safely
                    status = 'Delayed' if delay_days > 0 else 'On Time'
                    
                    # FIX: Handle percent_complete safely
                    percent_complete_raw = record.get('% Complete') or record.get('%_complete') or record.get('percent_complete')
                    percent_complete = None
                    
                    if isinstance(percent_complete_raw, (int, float)):
                        percent_complete = float(percent_complete_raw)
                    elif isinstance(percent_complete_raw, str):
                        try:
                            percent_complete = float(percent_complete_raw)
                        except:
                            percent_complete = None
                    
                    return {
                        "activity_id": target_activity_id,
                        "activity_name": activity_name,
                        "floor": record.get('Floor'),
                        "flat": record.get('Flat'),
                        "module": record.get('Module'),
                        "pour": record.get('Pour'),
                        "baseline_finish": QueryProcessor._format_date(record.get('Baseline Finish')),
                        "actual_finish": QueryProcessor._format_date(record.get('Finish')),
                        "delay_days": delay_days,
                        "percent_complete": percent_complete,
                        "status": status,
                        "found_in_tower": tower
                    }
            
            # If not found, log debug information
            logger.warning(f"Activity ID {target_activity_id} not found in tower {tower}")
            
            return {
                "activity_id": target_activity_id,
                "status": "Not Found",
                "message": f"Activity ID {target_activity_id} not found in data for tower {tower}",
                "found_in_tower": tower
            }
            
        except Exception as e:
            target_activity_id = QueryProcessor._get_special_activity_id_for_tower_name(tower)
            logger.error(f"Error getting Activity ID {target_activity_id} for tower {tower}: {e}")
            return {
                "activity_id": target_activity_id,
                "status": "Error",
                "message": str(e),
                "found_in_tower": tower
            }
    
    @staticmethod
    def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame."""
        filtered_df = df.copy()
        
        if filters.get("min_delay_days"):
            filtered_df = filtered_df[filtered_df['Delay_Days'] >= filters["min_delay_days"]]
        
        if filters.get("module"):
            filtered_df = filtered_df[filtered_df['Module'] == filters["module"]]
        
        if filters.get("floor"):
            filtered_df = filtered_df[filtered_df['Floor'] == filters["floor"]]
        
        if filters.get("flat"):
            filtered_df = filtered_df[filtered_df['Flat'] == filters["flat"]]
        
        return filtered_df
    
    @staticmethod
    def _create_tower_summary(tower_data: Dict, tower: str) -> Dict:
        """Create tower summary."""
        try:
            data = tower_data.get('data', [])
            metrics = tower_data.get('metrics', {})
            
            # If metrics are available, use them
            if metrics:
                return {
                    "group_name": tower,
                    "total_activities": metrics.get('total_activities', 0),
                    "delayed_count": metrics.get('delayed_count', 0),
                    "on_time_count": metrics.get('on_time_count', 0),
                    "max_delay_days": metrics.get('max_delay', 0),
                    "avg_delay_days": metrics.get('avg_delay', 0.0)
                }
            else:
                # Calculate from raw data if metrics not available
                total_activities = len(data)
                delayed_count = 0
                max_delay = 0
                total_delay = 0
                
                for record in data:
                    delay_raw = record.get('Delay_Days', 0)
                    delay = 0
                    
                    # Convert delay safely
                    if isinstance(delay_raw, (int, float)):
                        delay = int(delay_raw)
                    elif isinstance(delay_raw, str):
                        try:
                            delay = int(float(delay_raw))
                        except:
                            delay = 0
                    
                    if delay > 0:
                        delayed_count += 1
                        total_delay += delay
                        if delay > max_delay:
                            max_delay = delay
                
                avg_delay = total_delay / delayed_count if delayed_count > 0 else 0
                on_time_count = total_activities - delayed_count
                
                return {
                    "group_name": tower,
                    "total_activities": total_activities,
                    "delayed_count": delayed_count,
                    "on_time_count": on_time_count,
                    "max_delay_days": max_delay,
                    "avg_delay_days": round(float(avg_delay), 2)
                }
                
        except Exception as e:
            logger.error(f"Error creating tower summary: {e}")
            return {
                "group_name": tower,
                "total_activities": 0,
                "delayed_count": 0,
                "on_time_count": 0,
                "max_delay_days": 0,
                "avg_delay_days": 0.0
            }
    
    @staticmethod
    def _group_by_column(tower_data: Dict, tower: str, column: str, label: str, filters: Dict[str, Any] = None) -> List[Dict]:
        """Group by a specific column with flexible value matching and optional filtering."""
        results = []
        
        try:
            data = tower_data.get('data', [])
            
            if not data:
                return results
            
            # Convert raw data to activity details format
            activity_details = QueryProcessor._get_all_activity_details(tower_data, filters)
            
            # Group data by the specified column with normalized values
            groups = {}
            for activity in activity_details:
                value = activity.get(column.lower())  # Use lowercase key
                if value is not None:
                    # Normalize the value for grouping
                    normalized_value = QueryProcessor._normalize_column_value(value, column)
                    
                    if normalized_value not in groups:
                        groups[normalized_value] = []
                    groups[normalized_value].append(activity)
            
            for normalized_value, group in groups.items():
                # Calculate metrics for this group with safe type handling
                delayed_count = 0
                max_delay = 0
                total_delay = 0
                
                for r in group:
                    delay_days = r.get('delay_days', 0)
                    
                    # Convert delay to int safely
                    if isinstance(delay_days, (int, float)):
                        delay_days_int = int(delay_days)
                    elif isinstance(delay_days, str):
                        try:
                            delay_days_int = int(float(delay_days))
                        except:
                            delay_days_int = 0
                    else:
                        delay_days_int = 0
                    
                    if delay_days_int > 0:
                        delayed_count += 1
                        total_delay += delay_days_int
                        if delay_days_int > max_delay:
                            max_delay = delay_days_int
                
                avg_delay = total_delay / len(group) if group and delayed_count > 0 else 0
                
                # Get display value
                display_value = normalized_value
                if group:
                    # Count frequency of original values
                    value_counts = {}
                    for r in group:
                        orig_value = r.get(column.lower())  # Use lowercase key
                        if orig_value:
                            value_counts[orig_value] = value_counts.get(orig_value, 0) + 1
                    
                    # Use the most frequent original value
                    if value_counts:
                        display_value = max(value_counts.items(), key=lambda x: x[1])[0]
                
                results.append({
                    "group_name": f"{tower} - {label} {display_value}",
                    label.lower(): str(display_value),
                    "normalized_value": normalized_value,
                    "total_activities": len(group),
                    "delayed_count": delayed_count,
                    "on_time_count": len(group) - delayed_count,
                    "max_delay_days": int(max_delay),
                    "avg_delay_days": round(float(avg_delay), 2)
                })
        except Exception as e:
            logger.error(f"Error in group_by_column for {column}: {e}")
        
        return results
    
    @staticmethod
    def _normalize_column_value(value: Any, column: str) -> str:
        """Normalize column values for consistent grouping."""
        if value is None:
            return ""
        
        str_value = str(value).strip().upper()
        
        if column == 'Floor':
            # Normalize floor values: extract number, remove suffixes
            # "3F" -> "3", "3rd Floor" -> "3", "Floor 3" -> "3"
            match = re.search(r'(\d+)', str_value)
            if match:
                return match.group(1)
            return str_value
        
        elif column == 'Module':
            # Normalize module values: ensure consistent format
            # "M7" -> "M7", "Module 7" -> "M7", "7" -> "M7"
            # "G3" -> "G3", "Module G3" -> "G3"
            # Remove non-alphanumeric
            clean_value = re.sub(r'[^A-Z0-9]', '', str_value)
            
            if not clean_value:
                return ""
            
            # Check if it already starts with a letter (like G3, M7, A1, etc.)
            if clean_value[0].isalpha():
                # Already has letter prefix
                return clean_value
            
            # If it's just numbers, add M prefix (preserving your original logic)
            if clean_value.isdigit():
                return f"M{clean_value}"
            
            # For mixed patterns like "7A" -> "M7A" (if you want M prefix for these)
            # Or keep as-is: "7A" -> "7A"
            return f"M{clean_value}"  # Or just return clean_value if you prefer no prefix
        
        elif column == 'Flat':
            # Normalize flat values: just the number
            match = re.search(r'(\d+)', str_value)
            if match:
                return match.group(1)
            return str_value
        
        elif column == 'Pour':
            # Normalize pour values: just the number
            match = re.search(r'(\d+)', str_value)
            if match:
                return match.group(1)
            return str_value
        
        return str_value
    
    @staticmethod
    def _get_all_activity_details(tower_data: Dict, filters: Dict[str, Any] = None) -> List[Dict]:
        """Get detailed activity information for ALL activities, with optional filtering."""
        results = []
        
        try:
            data = tower_data.get('data', [])
            
            logger.info(f"Processing {len(data)} activities for activity-wise analysis")
            
            for record in data:
                try:
                    # FIX: Convert delay_days to integer safely
                    delay_days_raw = record.get('Delay_Days', 0)
                    delay_days = 0
                    
                    # Handle string, int, float, or None values
                    if isinstance(delay_days_raw, (int, float)):
                        delay_days = int(delay_days_raw)
                    elif isinstance(delay_days_raw, str):
                        try:
                            delay_days = int(float(delay_days_raw))
                        except:
                            delay_days = 0
                    elif delay_days_raw is None:
                        delay_days = 0
                    
                    # FIX: Convert activity_id to integer safely, keep as string if not numeric
                    activity_id_raw = record.get('Activity ID')
                    activity_id = None
                    
                    if isinstance(activity_id_raw, (int, float)):
                        activity_id = int(activity_id_raw)
                    elif isinstance(activity_id_raw, str):
                        try:
                            activity_id = int(float(activity_id_raw))
                        except:
                            activity_id = activity_id_raw.strip() # Keep original string
                    
                    # FIX: Convert percent_complete safely
                    percent_complete_raw = record.get('% Complete')
                    percent_complete = None
                    
                    if isinstance(percent_complete_raw, (int, float)):
                        percent_complete = float(percent_complete_raw)
                    elif isinstance(percent_complete_raw, str):
                        try:
                            percent_complete = float(percent_complete_raw)
                        except:
                            percent_complete = None
                    
                    # Extract all relevant fields with safe conversions
                    result = {
                        "activity_id": activity_id,
                        "activity_name": record.get('Activity Name'),
                        "floor": record.get('Floor'),
                        "flat": record.get('Flat'),
                        "module": record.get('Module'),
                        "pour": record.get('Pour'),
                        "trade": record.get('Trade') or record.get('trade'),
                        "tower": record.get('Tower') or record.get('tower'),
                        "baseline_finish": QueryProcessor._format_date(
                            record.get('Baseline Finish') or record.get('baseline_finish') or record.get('Planned Finish')
                        ),
                        "actual_finish": QueryProcessor._format_date(
                            record.get('Finish') or record.get('actual_finish') or record.get('Actual Finish') or record.get('End Date')
                        ),
                        "delay_days": delay_days,
                        "percent_complete": percent_complete,
                        "status": "Delayed" if delay_days > 0 else "On Time"
                    }
                    
                    # Clean None values
                    cleaned_result = {}
                    for key, value in result.items():
                        if value is not None:
                            cleaned_result[key] = value
                        else:
                            cleaned_result[key] = None
                    
                    results.append(cleaned_result)
                    
                except Exception as e:
                    logger.warning(f"Error processing activity record: {e}")
                    continue
            
            # Apply filters if provided
            if filters:
                results = QueryProcessor._filter_activity_details(results, filters)
                                
        except Exception as e:
            logger.error(f"Error getting all activity details: {e}")
        
        return results
    
    @staticmethod
    def _group_data(tower_data: Dict, tower: str, query_type: QueryType, filters: Dict[str, Any] = None) -> Dict:
        """Group data based on query type."""
        result = {}
        
        # For activity-wise queries, include ALL activity details (with optional filtering)
        # Compare by value to avoid Enum identity issues
        if query_type.value == QueryType.ACTIVITY_WISE.value:
            result["activity_details"] = QueryProcessor._get_all_activity_details(tower_data, filters)
            # Also include a summary
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
            # Also include the tower-specific special activity
            special_key = QueryProcessor._get_special_activity_key_for_tower_name(tower)
            result[special_key] = QueryProcessor._get_activity_id_2_for_tower(tower_data, tower)
        
        # For tower-wise queries, include special activity and summary
        elif query_type == QueryType.TOWER_WISE:
            special_key = QueryProcessor._get_special_activity_key_for_tower_name(tower)
            result[special_key] = QueryProcessor._get_activity_id_2_for_tower(tower_data, tower)
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
        
        # For multi-tower queries
        elif query_type == QueryType.MULTI_TOWER:
            special_key = QueryProcessor._get_special_activity_key_for_tower_name(tower)
            result[special_key] = QueryProcessor._get_activity_id_2_for_tower(tower_data, tower)
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
        
        # For floor-wise queries (pass filters)
        elif query_type == QueryType.FLOOR_WISE:
            floor_wise_data = QueryProcessor._group_by_column(tower_data, tower, 'Floor', 'Floor', filters)
            if floor_wise_data:
                result["floor_wise"] = floor_wise_data
            else:
                result["floor_wise"] = []
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
        
        # For flat-wise queries (pass filters)
        elif query_type == QueryType.FLAT_WISE:
            flat_wise_data = QueryProcessor._group_by_column(tower_data, tower, 'Flat', 'Flat', filters)
            if flat_wise_data:
                result["flat_wise"] = flat_wise_data
            else:
                result["flat_wise"] = []
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
        
        # For pour-wise queries (pass filters)
        elif query_type == QueryType.POUR_WISE:
            pour_wise_data = QueryProcessor._group_by_column(tower_data, tower, 'Pour', 'Pour', filters)
            if pour_wise_data:
                result["pour_wise"] = pour_wise_data
            else:
                result["pour_wise"] = []
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
        
        # For module-wise queries (pass filters)
        elif query_type == QueryType.MODULE_WISE:
            module_wise_data = QueryProcessor._group_by_column(tower_data, tower, 'Module', 'Module', filters)
            if module_wise_data:
                result["module_wise"] = module_wise_data
            else:
                result["module_wise"] = []
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
        
        # For summary queries
        elif query_type == QueryType.SUMMARY:
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
            # Add the tower-specific special activity for summary queries as well
            special_key = QueryProcessor._get_special_activity_key_for_tower_name(tower)
            result[special_key] = QueryProcessor._get_activity_id_2_for_tower(tower_data, tower)
            # Add additional summary metrics
            metrics = tower_data.get('metrics', {})
            result["detailed_summary"] = {
                "max_delay": metrics.get('max_delay', 0),
                "avg_delay": metrics.get('avg_delay', 0.0)
            }
        
        # For comparison queries
        elif query_type == QueryType.COMPARISON:
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
            result["note"] = "Comparison analysis requires multiple towers. Specify towers to compare."
        
        # For activity details queries
        elif query_type == QueryType.ACTIVITY_DETAILS:
            result["summary"] = QueryProcessor._create_tower_summary(tower_data, tower)
        
        return result
    
    @staticmethod
    async def _analyze_single_tower(
        project: str, 
        tower: str, 
        intent_response: IntentResponse,
        start_time: float
    ) -> Dict[str, Any]:
        """Analyze single tower with proper query type handling."""
        tower_data = DataLoader.load_tower_data(project, tower)
        
        if not tower_data:
            raise HTTPException(status_code=404, detail=f"Tower {tower} not found in project {project}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Get grouped data based on query type (passing filters)
        grouped_data = QueryProcessor._group_data(tower_data, tower, intent_response.query_type, intent_response.filters)
        
        metrics = tower_data.get('metrics', {})
        
        response = {
            "status": "success",
            "intent": intent_response.intent.value,
            "timestamp": datetime.now().isoformat(),
            "project": project,
            "tower": tower,
            "query_type": intent_response.query_type.value,
            "filters_applied": intent_response.filters,
            "query_info": {
                "original_query": intent_response.parsed_query,
                "confidence": float(intent_response.confidence)
            }
        }
        
        # Merge grouped data (activity_details, summary, etc.) into response root
        response.update(grouped_data)
        
        # For activity-wise queries, add count of activities returned and limit output
        if intent_response.query_type.value == QueryType.ACTIVITY_WISE.value and "activity_details" in grouped_data:
            details = grouped_data["activity_details"]
            activity_count = len(details)
            response["activity_count"] = activity_count
            
            # Sort and limit activities to avoid huge payload
            # Helper for sorting safely
            def get_delay_value(item):
                try:
                    val = item.get('delay_days', 0)
                    return float(val) if val is not None else 0.0
                except:
                    return 0.0

            # Sort by delay days (descending)
            details.sort(key=get_delay_value, reverse=True)
            
            # Limit to top 50 (configurable)
            LIMIT_ACTIVITIES = 50
            if activity_count > LIMIT_ACTIVITIES:
                response["activity_details"] = details[:LIMIT_ACTIVITIES]
                response["showing_top"] = LIMIT_ACTIVITIES
                response["total_activities_found"] = activity_count
                response["pagination_note"] = f"Showing top {LIMIT_ACTIVITIES} of {activity_count} activities sorted by delay. Use filters to narrow down results."
            
            # Ensure summary exists
            if "summary" not in response:
                 response["summary"] = {}
            response["summary"]["activities_returned"] = len(response["activity_details"])
        
        # For tower-wise queries, note what's included
        elif intent_response.query_type.value == QueryType.TOWER_WISE.value:
            response["note"] = "Tower-wise analysis includes summary and activity ID 2 details"
        
        return response
    
    @staticmethod
    async def _analyze_multiple_towers(
        project: str,
        towers: List[str],
        intent_response: IntentResponse,
        start_time: float
    ) -> Dict[str, Any]:
        """Analyze multiple towers - include activity ID 2 for each tower."""
        all_results = []
        project_summary = {
            "total_activities": 0,
            "delayed_activities": 0,
            "on_time_activities": 0,
            "critical_count": 0,
            "towers_analyzed": 0
        }
        
        for tower in towers:
            try:
                tower_data = DataLoader.load_tower_data(project, tower)
                
                if tower_data:
                    metrics = tower_data.get('metrics', {})
                    
                    # Get tower summary
                    tower_summary = {
                        "tower": tower,
                        "total_activities": metrics.get('total_activities', 0),
                        "delayed_activities": metrics.get('delayed_count', 0),
                        "on_time_activities": metrics.get('on_time_count', 0),
                        "critical_count": metrics.get('critical_count', 0),
                        "max_delay": metrics.get('max_delay', 0),
                        "avg_delay": metrics.get('avg_delay', 0.0)
                    }
                    
                    # Get the tower-specific special activity (ONLY for tower-wise/multi-tower queries)
                    activity_id_2_info = QueryProcessor._get_activity_id_2_for_tower(tower_data, tower)
                    
                    tower_result = {**tower_summary}
                    
                    # Only include the special activity for tower-wise queries
                    if intent_response.query_type == QueryType.MULTI_TOWER:
                        special_key = QueryProcessor._get_special_activity_key_for_tower_name(tower)
                        tower_result[special_key] = activity_id_2_info
                    
                    all_results.append(tower_result)
                    
                    # Update project totals
                    project_summary["total_activities"] += tower_summary["total_activities"]
                    project_summary["delayed_activities"] += tower_summary["delayed_activities"]
                    project_summary["on_time_activities"] += tower_summary["on_time_activities"]
                    project_summary["critical_count"] += tower_summary["critical_count"]
                    project_summary["towers_analyzed"] += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing tower {tower}: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            "status": "success",
            "intent": intent_response.intent.value,
            "timestamp": datetime.now().isoformat(),
            "project": project,
            "query_type": "multi-tower",
            "towers_analyzed": int(project_summary["towers_analyzed"]),
            "project_summary": {
                "total_activities": int(project_summary["total_activities"]),
                "delayed_activities": int(project_summary["delayed_activities"]),
                "on_time_activities": int(project_summary["on_time_activities"]),
                "critical_count": int(project_summary["critical_count"]),
                "towers_analyzed": int(project_summary["towers_analyzed"])
            },
            "processing_time_ms": round(float(processing_time), 2),
            "tower_wise_results": all_results,
            "query_info": {
                "original_query": intent_response.parsed_query,
                "confidence": float(intent_response.confidence)
            }
        }
        
        return response
    
    @staticmethod
    async def _handle_delay_analysis(
        intent_response: IntentResponse,
        available_projects: List[str],
        user_project: Optional[str],
        user_towers: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle delay analysis intent."""
        project = user_project or intent_response.project
        
        # If still no project, use first available
        if not project and available_projects:
            project = available_projects[0]
        
        if not project:
            raise HTTPException(status_code=400, detail="No project specified for analysis")
        
        # Get actual towers from project data if towers list is empty (meaning "all towers")
        if not user_towers:
            try:
                project_data = DataLoader.load_project_data(project)
                available_towers = project_data.get('available_towers', [])
                tower_scope = str(intent_response.filters.get("tower_scope", "")).upper()

                if tower_scope in {"EWS", "LIG"}:
                    available_towers = [
                        tower for tower in available_towers
                        if tower_scope in str(tower).upper()
                    ]
                
                # If no specific towers mentioned, use all towers for ANY query type
                towers = available_towers
                logger.info(f"Analyzing all {len(towers)} towers in project {project} for {intent_response.query_type.value} query")
                
            except Exception as e:
                logger.error(f"Error getting towers for project {project}: {e}")
                towers = []
        else:
            towers = user_towers
        
        logger.info(f"Processing query for project: {project}, towers: {towers}, query_type: {intent_response.query_type.value}")
        
        # If no towers available
        if not towers:
            raise HTTPException(
                status_code=404, 
                detail=f"No towers found in project {project}"
            )
        
        # For activity-wise queries with multiple towers, use special handler
        if intent_response.query_type == QueryType.ACTIVITY_WISE and len(towers) > 1:
            return await QueryProcessor._handle_multi_tower_activity_wise(
                project, towers, intent_response, start_time
            )
        
        # For other query types with multiple towers, use multi-tower analysis
        elif len(towers) > 1:
            # Create a new handler for grouped queries (pour-wise, floor-wise, etc.) across multiple towers
            return await QueryProcessor._handle_multi_tower_grouped_analysis(
                project, towers, intent_response, start_time
            )
        
        # For single tower analysis
        elif len(towers) == 1:
            return await QueryProcessor._analyze_single_tower(
                project, towers[0], intent_response, start_time
            )
        
        else:
            # No towers available
            raise HTTPException(
                status_code=404, 
                detail=f"No towers found in project {project}"
            )
    
    @staticmethod
    async def _handle_multi_tower_activity_wise(
        project: str,
        towers: List[str],
        intent_response: IntentResponse,
        start_time: float
    ) -> Dict[str, Any]:
        """Handle activity-wise queries across multiple towers."""
        all_activity_details = []
        activity_highlights = []
        tower_summaries = []
        project_summary = {
            "total_activities": 0,
            "delayed_activities": 0,
            "on_time_activities": 0,
            "critical_count": 0,
            "towers_analyzed": 0
        }
        
        for tower in towers:
            try:
                tower_data = DataLoader.load_tower_data(project, tower)
                
                if tower_data:
                    # Get all activity details with filters applied
                    tower_activity_details = QueryProcessor._get_all_activity_details(
                        tower_data, intent_response.filters
                    )
                    
                    # Add tower information to each activity
                    for activity in tower_activity_details:
                        activity['tower'] = tower
                        activity['project'] = project
                    
                    all_activity_details.extend(tower_activity_details)
                    
                    # Extract Activity ID 2 for highlight (Requested Feature)
                    act_id_2 = QueryProcessor._get_activity_id_2_for_tower(tower_data, tower)
                    if act_id_2:
                        activity_highlights.append(act_id_2)

                    # Update project totals
                    metrics = tower_data.get('metrics', {})
                    project_summary["total_activities"] += metrics.get('total_activities', 0)
                    project_summary["delayed_activities"] += metrics.get('delayed_count', 0)
                    project_summary["on_time_activities"] += metrics.get('on_time_count', 0)
                    project_summary["critical_count"] += metrics.get('critical_count', 0)
                    project_summary["towers_analyzed"] += 1
                    
                    # Create tower summary for consolidated view
                    tower_summary = QueryProcessor._create_tower_summary(tower_data, tower)
                    tower_summaries.append(tower_summary)
                    
            except Exception as e:
                logger.error(f"Error analyzing tower {tower} for activity-wise query: {e}")
        
        # Sort and limit activities to avoid huge payload
        total_activities_found = len(all_activity_details)
        
        # Helper for sorting safely
        def get_delay_value(item):
            try:
                val = item.get('delay_days', 0)
                return float(val) if val is not None else 0.0
            except:
                return 0.0

        # Sort by delay days (descending)
        all_activity_details.sort(key=get_delay_value, reverse=True)
        
        # Limit to top 50 (configurable)
        LIMIT_ACTIVITIES = 50
        limited_activities = all_activity_details[:LIMIT_ACTIVITIES]
        
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            "status": "success",
            "intent": intent_response.intent.value,
            "timestamp": datetime.now().isoformat(),
            "project": project,
            "query_type": "activity-wise",
            "towers_analyzed": int(project_summary["towers_analyzed"]),
            "project_summary": {
                "total_activities": int(project_summary["total_activities"]),
                "delayed_activities": int(project_summary["delayed_activities"]),
                "on_time_activities": int(project_summary["on_time_activities"]),
                "critical_count": int(project_summary["critical_count"]),
                "towers_analyzed": int(project_summary["towers_analyzed"])
            },
            "filters_applied": intent_response.filters,
            "processing_time_ms": round(float(processing_time), 2),
            "activity_highlights": activity_highlights,
            "activity_details": limited_activities,
            "tower_results": tower_summaries,
            "total_activities_found": total_activities_found,
            "showing_top": len(limited_activities),
            "pagination_note": f"Showing top {len(limited_activities)} of {total_activities_found} activities sorted by delay. Use filters to narrow down results.",
            "query_info": {
                "original_query": intent_response.parsed_query,
                "confidence": float(intent_response.confidence)
            }
        }
        
        return response
    
    @staticmethod
    async def _handle_multi_tower_grouped_analysis(
        project: str,
        towers: List[str],
        intent_response: IntentResponse,
        start_time: float
    ) -> Dict[str, Any]:
        """Handle grouped queries (pour-wise, floor-wise, etc.) across multiple towers."""
        all_grouped_results = []
        project_summary = {
            "total_activities": 0,
            "delayed_activities": 0,
            "on_time_activities": 0,
            "critical_count": 0,
            "towers_analyzed": 0
        }
        
        for tower in towers:
            try:
                tower_data = DataLoader.load_tower_data(project, tower)
                
                if tower_data:
                    # Get grouped data for this tower based on query type
                    grouped_data = QueryProcessor._group_data(
                        tower_data, tower, intent_response.query_type, intent_response.filters
                    )
                    
                    # Add tower information to the results
                    tower_result = {
                        "tower": tower,
                        "results": grouped_data
                    }
                    
                    all_grouped_results.append(tower_result)
                    
                    # Update project totals from tower metrics
                    metrics = tower_data.get('metrics', {})
                    project_summary["total_activities"] += metrics.get('total_activities', 0)
                    project_summary["delayed_activities"] += metrics.get('delayed_count', 0)
                    project_summary["on_time_activities"] += metrics.get('on_time_count', 0)
                    project_summary["critical_count"] += metrics.get('critical_count', 0)
                    project_summary["towers_analyzed"] += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing tower {tower} for grouped query: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            "status": "success",
            "intent": intent_response.intent.value,
            "timestamp": datetime.now().isoformat(),
            "project": project,
            "query_type": intent_response.query_type.value,
            "towers_analyzed": int(project_summary["towers_analyzed"]),
            "project_summary": {
                "total_activities": int(project_summary["total_activities"]),
                "delayed_activities": int(project_summary["delayed_activities"]),
                "on_time_activities": int(project_summary["on_time_activities"]),
                "critical_count": int(project_summary["critical_count"]),
                "towers_analyzed": int(project_summary["towers_analyzed"])
            },
            "filters_applied": intent_response.filters,
            "processing_time_ms": round(float(processing_time), 2),
            "tower_results": all_grouped_results,
            "total_towers_analyzed": len(all_grouped_results),
            "query_info": {
                "original_query": intent_response.parsed_query,
                "confidence": float(intent_response.confidence)
            }
        }
        
        return response
    
    
    @staticmethod
    def get_available_projects() -> List[str]:
        """Get available projects."""
        # Always try to fetch from COS to ensure we have the latest list of projects
        # This fixes the issue where new projects (like eligo) are not detected if cache is populated
        try:
            cos_projects = list_projects_from_cos()
            if cos_projects:
                return cos_projects
        except Exception as e:
            logger.error(f"Error fetching projects from COS: {e}")
            
        # Fallback to cache only if COS fails
        cached_projects = project_cache.get_all_projects()
        if cached_projects:
            return cached_projects
            
        return []



# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.post("/analyze", summary="Analyze construction delays with natural language", tags=["Analysis"])
async def analyze_delay(query_request: QueryRequest):
    try:
        result = await QueryProcessor.process_query(query_request)
        
        # Simplify output - only show required results
        required_keys = [
            "project", 
            "tower",
            "project_summary", 
            "activity_details", 
            "activity_id_1",
            "activity_id_2",
            "tower_results", 
            "activity_highlights",
            "total_summary",
            "projects_analyzed",
        ]
        
        simplified = {k: result[k] for k in required_keys if k in result}
        simplified = shape_special_project_response(
            simplified,
            query_request.query,
            result.get("filters_applied", {})
        )

        return strip_severity_from_response(simplified)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

