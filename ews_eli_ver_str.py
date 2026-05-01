import os
import re
import math
import json
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Dict, Any, Optional
from functools import lru_cache
from enum import Enum

import numpy as np
import pandas as pd
import ibm_boto3
from ibm_botocore.client import Config
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

# --------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load ENV
# --------------------------------------------------
load_dotenv()

COS_API_KEY = os.getenv("COS_API_KEY")
COS_CRN = os.getenv("COS_SERVICE_INSTANCE_CRN")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
COS_BUCKET = os.getenv("COS_BUCKET_NAME")

# Watsonx credentials (for LLM)
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-3-70b-instruct")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.5))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 200))

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Unified Structure Work Tracker API with LLM Query",
    version="5.0"
)


@app.on_event("startup")
async def warmup_services():
    """Warm expensive services once when the API starts."""
    if llm_service and WATSONX_URL and WATSONX_API_KEY:
        logger.info("Warming up LLM service at startup")
        llm_service.initialize()

# --------------------------------------------------
# COS Client
# --------------------------------------------------
cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY,
    ibm_service_instance_id=COS_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
)

# --------------------------------------------------
# CONSTANTS & ENUMS
# --------------------------------------------------

class ProjectType(str, Enum):
    """Supported project types"""
    EWS_LIG = "ews-lig"
    ELIGO = "eligo"
    VERIDIA = "Veridia"

class QueryType(str, Enum):
    """Available query types for delay analysis."""
    TOWER_WISE = "tower-wise"
    FLOOR_WISE = "floor-wise"
    POUR_WISE = "pour-wise"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    DELAY_ANALYSIS = "delay-analysis"
    CRITICAL_DELAYS = "critical-delays"
    MODULE_WISE = "module-wise"

class UserIntent(str, Enum):
    """Possible user intents."""
    GET_DELAYS = "get_delays"
    GET_SUMMARY = "get_summary"
    COMPARE_TOWERS = "compare_towers"
    FIND_CRITICAL = "find_critical_delays"
    GET_STATUS = "get_status"
    LIST_PROJECTS = "list_projects"
    LIST_TOWERS = "list_towers"
    GET_PROJECT_DATA = "get_project_data"

STRUCTURE_PATTERN = re.compile(
    r"(Structure Work Tracker.*|RevisedEWS_LIG.*|EWS_LIG.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx",
    re.IGNORECASE
)

ELIGO_PATTERN = re.compile(
    r"(eligo_.*|Eligo_.*|Eligo.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx",
    re.IGNORECASE
)

VERIDIA_PATTERN = re.compile(
    r"(Structure Work Tracker.*|Veridia.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx",
    re.IGNORECASE
)

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
    parsed_query: Dict[str, Any] = {}

# --------------------------------------------------
# JSON SERIALIZATION HELPER
# --------------------------------------------------

def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types and NaN/Inf values to JSON-serializable types"""
    
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return [convert_to_json_serializable(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

# --------------------------------------------------
# DELAY ANALYSIS FUNCTIONS
# --------------------------------------------------

def calculate_delay_days(actual_date_str: str, baseline_date_str: str) -> int:
    """Calculate delay in days between actual and baseline dates."""
    if not actual_date_str or not baseline_date_str:
        return 0
    
    try:
        actual_date = None
        baseline_date = None
        
        # Helper to parse date string
        def parse_date(date_str):
            if isinstance(date_str, (datetime, pd.Timestamp)):
                return date_str
            
            s = str(date_str).strip()
            # Handle timestamps with time
            if " " in s:
                try:
                    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                except:
                    s = s.split(" ")[0]
            
            formats = [
                "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
                "%d-%b-%y", "%d-%b-%Y", "%d/%b/%y", "%d/%b/%Y"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt)
                except:
                    continue
            return None

        actual_date = parse_date(actual_date_str)
        baseline_date = parse_date(baseline_date_str)
        
        if not actual_date or not baseline_date:
            # logger.warning(f"Could not parse dates: actual='{actual_date_str}', baseline='{baseline_date_str}'")
            return 0
        
        delay_days = (actual_date - baseline_date).days
        return max(0, delay_days)  # Only positive delays
        
    except Exception as e:
        logger.error(f"Error calculating delay: {e}")
        return 0

def calculate_delay_metrics(activities: List[Dict]) -> Dict:
    """Calculate delay metrics from activity data."""
    if not activities:
        return {
            "delayed_count": 0,
            "on_time_count": 0,
            "max_delay": 0,
            "avg_delay": 0.0
        }
    
    total_activities = len(activities)
    delayed_activities = []
    
    for activity in activities:
        delay_days = activity.get("delay_days", 0)
        if delay_days > 0:
            delayed_activities.append(activity)
    
    max_delay = 0
    total_delay = 0
    
    for activity in delayed_activities:
        delay_days = activity.get("delay_days", 0)
        if delay_days > max_delay:
            max_delay = delay_days
        total_delay += delay_days
    
    avg_delay = total_delay / len(delayed_activities) if delayed_activities else 0.0
    
    return {
        "delayed_count": len(delayed_activities),
        "on_time_count": total_activities - len(delayed_activities),
        "max_delay": max_delay,
        "avg_delay": round(avg_delay, 2)
    }

def add_delay_analysis_to_tower_data(tower_data: Dict) -> Dict:
    """Add delay analysis to tower data structure."""
    if "pours" not in tower_data:
        return tower_data
    
    all_activities = []
    
    for pour_name, pour_data in tower_data["pours"].items():
        if isinstance(pour_data, dict):
            for floor_name, floor_data in pour_data.items():
                if isinstance(floor_data, dict):
                    activity = {
                        "tower": tower_data.get("tower", "Unknown"),
                        "pour": pour_name,
                        "floor": floor_name,
                        "baseline_date": floor_data.get("baseline"),
                        "actual_date": floor_data.get("anticipated"),
                        # "days": floor_data.get("days")
                    }
                    
                    delay_days = calculate_delay_days(
                        floor_data.get("anticipated"),
                        floor_data.get("baseline")
                    )
                    
                    activity["delay_days"] = delay_days
                    activity["status"] = "Delayed" if delay_days > 0 else "On Time"
                    
                    all_activities.append(activity)
                    
                    floor_data["delay_days"] = delay_days
                    floor_data["status"] = activity["status"]
    
    tower_data["delay_metrics"] = calculate_delay_metrics(all_activities)
    tower_data["activities"] = all_activities
    
    return tower_data

# --------------------------------------------------
# LLM SERVICE (Similar to app.py)
# --------------------------------------------------

try:
    import importlib
    try:
        ml_module = importlib.import_module("ibm_watson_machine_learning.foundation_models")
        ModelInference = getattr(ml_module, "ModelInference", None)
    except Exception:
        ModelInference = None
    import asyncio
    import threading
    
    class LLMService:
        """Service for LLM-powered query understanding."""
        
        def __init__(self):
            self.model = None
            self.initialized = False
            self.init_attempted = False
            self.init_lock = threading.Lock()
            
        def initialize(self):
            """Initialize the LLM model."""
            if self.initialized:
                return
            if self.init_attempted and self.model is None:
                return
                
            with self.init_lock:
                if self.initialized:
                    return
                if self.init_attempted and self.model is None:
                    return
                
                self.init_attempted = True
                    
                if ModelInference is None:
                    logger.warning("ModelInference not available (import failed). Using fallback parser.")
                    return

                try:
                    credentials = {
                        "url": WATSONX_URL,
                        "apikey": WATSONX_API_KEY
                    }
                    
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
        
        async def analyze_query(self, query: str) -> IntentResponse:
            """Analyze user query using LLM."""
            if not self.initialized:
                self.initialize()
                
                if not self.initialized:
                    logger.warning("LLM not initialized, using fallback parser")
                    return self._fallback_parser(query)
            
            try:
                prompt = self._create_intent_prompt(query)
                response = await self._get_llm_response(prompt)
                intent_data = self._parse_llm_response(response)
                
                logger.info(f"LLM analysis successful: {intent_data.intent.value} (confidence: {intent_data.confidence})")
                return intent_data
                
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}", exc_info=True)
                return self._fallback_parser(query)
        
        def _create_intent_prompt(self, query: str) -> str:
            """Create prompt for intent classification."""
            
            prompt = f"""Analyze this construction project structure analysis query and extract structured information:

User Query: "{query}"

Available Projects: ["EWS LIG P4", "Eligo", "Veridia"]

Return ONLY a JSON object with this EXACT structure:
{{
    "intent": "get_delays|get_summary|compare_towers|find_critical_delays|get_status|list_projects|list_towers|get_project_data",
    "confidence": 0.0-1.0,
    "project": "ews-lig|eligo|veridia|all|null",
    "towers": ["tower1", "tower2"],
    "query_type": "tower-wise|floor-wise|pour-wise|summary|comparison|delay-analysis|critical-delays",
    "filters": {{}},
    "activity_id": "id_or_null"
}}

CRITICAL RULES:
1. If query mentions "EWS LIG" or "EWS", set project to "ews-lig"
2. If query mentions "Eligo", set project to "eligo"
3. If query mentions "Veridia", set project to "veridia"
4. If query mentions "all projects" or doesn't specify, set project to "all"
5. For EWS LIG: towers are like "EWS Tower 1", "LIG Tower 1", "Tower 1", etc.
6. For Eligo: towers are "Tower F", "Tower G", "Tower H"
7. For Veridia: towers are "Tower 1", "Tower 2", or simple numbers "4", "5", "6", "7". Modules are "M4", "M5", "M6", "M7".
8. If query mentions modules (e.g., "M7", "Module 7"), add "module": "7" to filters
9. If query mentions specific floor (e.g., "3F", "3rd Floor"), add "floor": "3" to filters
10. If query mentions specific pour (e.g., "Pour 1", "first pour"), add "pour": "1" to filters
11. For delay-related queries, use query_type "delay-analysis"
12. For critical delays, use query_type "critical-delays"
13. For comparison between towers, use query_type "comparison"

EXAMPLES:
1. Query: "show me delays in EWS LIG project" -> {{
    "intent": "get_delays",
    "confidence": 0.95,
    "project": "ews-lig",
    "towers": [],
    "query_type": "delay-analysis",
    "filters": {{}},
    "activity_id": null
}}
2. Query: "compare Tower F and Tower G in Eligo" -> {{
    "intent": "compare_towers",
    "confidence": 0.95,
    "project": "eligo",
    "towers": ["F", "G"],
    "query_type": "comparison",
    "filters": {{}},
    "activity_id": null
}}
3. Query: "show critical delays in all projects" -> {{
    "intent": "find_critical_delays",
    "confidence": 0.95,
    "project": "all",
    "towers": [],
    "query_type": "critical-delays",
    "filters": {{}},
    "activity_id": null
}}
4. Query: "what's the status of floor 5 in EWS LIG" -> {{
    "intent": "get_status",
    "confidence": 0.95,
    "project": "ews-lig",
    "towers": [],
    "query_type": "floor-wise",
    "filters": {{"floor": "5"}},
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
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(prompt=prompt)
                )
                
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
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    logger.error(f"No JSON found in LLM response: {response[:200]}")
                    return self._create_default_response()
                
                json_str = json_match.group()
                
                # Clean JSON string
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
                json_str = re.sub(r'\s*```$', '', json_str)
                json_str = re.split(r'(?:The final answer is:|Final answer:|Answer:|Analysis:|```)', json_str)[0]
                
                # Fix common issues
                json_str = json_str.replace('"nulll"', 'null')
                json_str = json_str.replace("'nulll'", 'null')
                json_str = json_str.replace('"nul"', 'null')
                json_str = json_str.replace("'nul'", 'null')
                json_str = json_str.replace('"None"', 'null')
                json_str = json_str.replace("'None'", 'null')
                json_str = json_str.replace('"NULL"', 'null')
                json_str = json_str.replace("'NULL'", 'null')
                
                # Fix null values without quotes
                json_str = re.sub(r':\s*nulll?\b', ': null', json_str, flags=re.IGNORECASE)
                json_str = re.sub(r':\s*None\b', ': null', json_str, flags=re.IGNORECASE)
                json_str = re.sub(r':\s*NULL\b', ': null', json_str, flags=re.IGNORECASE)
                
                data = json.loads(json_str)
                
                # Map intent
                intent_str = data.get("intent", "get_delays")
                intent_map = {
                    "get_delays": UserIntent.GET_DELAYS,
                    "get_summary": UserIntent.GET_SUMMARY,
                    "compare_towers": UserIntent.COMPARE_TOWERS,
                    "find_critical_delays": UserIntent.FIND_CRITICAL,
                    "get_status": UserIntent.GET_STATUS,
                    "list_projects": UserIntent.LIST_PROJECTS,
                    "list_towers": UserIntent.LIST_TOWERS,
                    "get_project_data": UserIntent.GET_PROJECT_DATA,
                }
                intent = intent_map.get(intent_str.lower(), UserIntent.GET_DELAYS)
                
                # Map query type
                query_type_str = data.get("query_type", "tower-wise")
                query_type_map = {
                    "tower-wise": QueryType.TOWER_WISE,
                    "floor-wise": QueryType.FLOOR_WISE,
                    "pour-wise": QueryType.POUR_WISE,
                    "summary": QueryType.SUMMARY,
                    "comparison": QueryType.COMPARISON,
                    "delay-analysis": QueryType.DELAY_ANALYSIS,
                    "critical-delays": QueryType.CRITICAL_DELAYS,
                    "module-wise": QueryType.MODULE_WISE,
                }
                query_type = query_type_map.get(query_type_str.lower(), QueryType.TOWER_WISE)
                
                # Handle project
                project = data.get("project")
                if project in [None, "null", "None", "NULL", "all"]:
                    project = "all"
                
                # Handle towers
                towers = data.get("towers", [])
                if not isinstance(towers, list):
                    towers = []
                
                # Filter and clean tower names
                valid_towers = []
                for tower in towers:
                    if isinstance(tower, str):
                        tower_clean = tower.strip()
                        # Allow standard tower names (LIG Tower 1, Tower F, etc.)
                        # Relaxed validation: non-empty, reasonably short
                        if 1 <= len(tower_clean) <= 20:
                            valid_towers.append(tower_clean)
                
                towers = valid_towers
                
                logger.info(f"Parsed LLM data - project: {project}, towers: {towers}, query_type: {query_type_str}, intent: {intent_str}")
                
                return IntentResponse(
                    intent=intent,
                    confidence=float(data.get("confidence", 0.5)),
                    project=project,
                    towers=towers,
                    query_type=query_type,
                    filters=data.get("filters", {}),
                    parsed_query=data
                )
                
            except Exception as e:
                logger.error(f"Parse error: {e}, response: {response[:500]}")
                return self._create_default_response()
        
        def _create_default_response(self) -> IntentResponse:
            """Create default response for fallback."""
            return IntentResponse(
                intent=UserIntent.GET_DELAYS,
                confidence=0.5,
                project="all",
                query_type=QueryType.TOWER_WISE,
                parsed_query={"error": "LLM parsing failed"}
            )
        
        def _fallback_parser(self, query: str) -> IntentResponse:
            """Fallback parser when LLM is not available."""
            query_lower = query.lower()
            
            # Determine project
            project = "all"
            if "eligo" in query_lower:
                project = "eligo"
            elif "ews" in query_lower or "lig" in query_lower:
                project = "ews-lig"
            elif "veridia" in query_lower:
                project = "veridia"
            
            # Determine intent
            intent = UserIntent.GET_DELAYS
            if "summary" in query_lower or "overview" in query_lower:
                intent = UserIntent.GET_SUMMARY
            elif "compare" in query_lower:
                intent = UserIntent.COMPARE_TOWERS
            elif "critical" in query_lower:
                intent = UserIntent.FIND_CRITICAL
            elif "status" in query_lower:
                intent = UserIntent.GET_STATUS
            elif "list" in query_lower and "project" in query_lower:
                intent = UserIntent.LIST_PROJECTS
            elif "list" in query_lower and "tower" in query_lower:
                intent = UserIntent.LIST_TOWERS
            
            # Determine query type
            query_type = QueryType.DELAY_ANALYSIS
            if "floor" in query_lower:
                query_type = QueryType.FLOOR_WISE
            elif "pour" in query_lower:
                query_type = QueryType.POUR_WISE
            elif "tower" in query_lower and "wise" in query_lower:
                query_type = QueryType.TOWER_WISE
            elif "compare" in query_lower:
                query_type = QueryType.COMPARISON
            elif "critical" in query_lower:
                query_type = QueryType.CRITICAL_DELAYS
            elif "summary" in query_lower:
                query_type = QueryType.SUMMARY
            
            # Extract tower names - FIXED VERSION
            towers = []
            
            if project == "ews-lig":
                # Extract tower number from query
                tower_number = None
                
                # Try different patterns to find tower number
                patterns = [
                    r'tower\s*[#]?\s*(\d+)',          # "tower 2"
                    r'ews\s+tower\s*[#]?\s*(\d+)',    # "ews tower 2"
                    r'lig\s+tower\s*[#]?\s*(\d+)',    # "lig tower 2"
                    r'(\d+)(?:st|nd|rd|th)?\s+tower', # "2nd tower"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        tower_number = match.group(1)
                        break
                
                # Also look for standalone numbers after "tower"
                if not tower_number:
                    words = query_lower.split()
                    for i, word in enumerate(words):
                        if word == "tower" and i + 1 < len(words):
                            next_word = words[i + 1]
                            if next_word.isdigit():
                                tower_number = next_word
                                break
                
                # If we found a tower number, determine if it's EWS or LIG
                if tower_number:
                    # Check if query specifically mentions LIG
                    is_lig = "lig" in query_lower or "lig" in query
                    
                    if is_lig:
                        tower_name = f"LIG Tower {tower_number}"
                    else:
                        tower_name = f"EWS Tower {tower_number}"
                    
                    # Add only once, no duplicates
                    if tower_name not in towers:
                        towers.append(tower_name)
            
            elif project == "eligo":
                # Extract Eligo towers
                for letter in ['f', 'g', 'h']:
                    if f'tower {letter}' in query_lower or f'tower-{letter}' in query_lower:
                        tower_name = f"Tower {letter.upper()}"
                        if tower_name not in towers:
                            towers.append(tower_name)

            elif project == "veridia":
                # Extract Veridia towers
                # Look for "Tower X" or just "Tower" followed by number
                tower_match = re.search(r'tower\s*(\d+)', query_lower)
                if tower_match:
                    towers.append(f"Tower {tower_match.group(1)}")
            
            # Extract filters
            filters = {}
            floor_match = re.search(r'floor[-\s]*(\d+)', query_lower)
            if floor_match:
                filters["floor"] = floor_match.group(1)
            
            pour_match = re.search(r'pour[-\s]*(\d+)', query_lower)
            if pour_match:
                filters["pour"] = pour_match.group(1)
            
            module_match = re.search(r'(?:module|m)[-\s]*(\d+)', query_lower)
            if module_match:
                filters["module"] = module_match.group(1)
            
            logger.info(f"Fallback parser - Project: {project}, Towers: {towers}, Query: {query}")
            
            return IntentResponse(
                intent=intent,
                confidence=0.7,
                project=project,
                towers=towers,  # Now without duplicates
                query_type=query_type,
                filters=filters,
                parsed_query={"method": "fallback_parser"}
            )
            
    # Initialize LLM service
    llm_service = LLMService()
    
except ImportError:
    logger.warning("IBM Watson Machine Learning package not available. LLM features disabled.")
    llm_service = None

# --------------------------------------------------
# FILE MANAGEMENT HELPERS
# --------------------------------------------------

@lru_cache(maxsize=50)
def list_files(prefix: str) -> List[str]:
    res = cos.list_objects_v2(Bucket=COS_BUCKET, Prefix=prefix)
    return [
        o["Key"]
        for o in res.get("Contents", []) if not o["Key"].endswith("/")
    ]

def get_latest_file_by_pattern(pattern: re.Pattern, prefixes: List[str] = None, project_type_hint: str = None) -> Optional[Dict]:
    """Get latest file matching a pattern from multiple prefixes"""
    if prefixes is None:
        prefixes = ["", "EWS LIG P4/", "Eligo/"]
    
    latest = None
    latest_date = None
    
    for prefix in prefixes:
        for key in list_files(prefix):
            fname = key.split("/")[-1]
            match = pattern.search(fname)
            if not match:
                continue
            
            try:
                fdate = datetime.strptime(match.group(2), "%d-%m-%Y")
                
                if latest_date is None or fdate > latest_date:
                    if project_type_hint:
                        proj_type = project_type_hint
                    elif "eligo" in fname.lower() or "eligo" in prefix.lower():
                        proj_type = "ELIGO"
                    elif "ews" in fname.lower() or "revisedews" in fname.lower() or "ews" in prefix.lower():
                        proj_type = "EWS_LIG"
                    else:
                        proj_type = project_type_hint if project_type_hint else "EWS_LIG"
                    
                    latest = {
                        "file": fname,
                        "key": key,
                        "date": fdate,
                        "project": proj_type,
                        "prefix": prefix
                    }
                    latest_date = fdate
            except:
                continue
    
    return latest

def get_latest_ews_lig_file() -> Optional[Dict]:
    """Get latest EWS LIG file"""
    ews_lig_pattern = re.compile(r"(RevisedEWS_LIG.*|EWS_LIG.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx", re.IGNORECASE)
    file_info = get_latest_file_by_pattern(ews_lig_pattern, project_type_hint="EWS_LIG")
    
    if not file_info:
        structure_pattern = re.compile(r"(Structure Work Tracker.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx", re.IGNORECASE)
        file_info = get_latest_file_by_pattern(structure_pattern, project_type_hint="EWS_LIG")
    
    return file_info

def get_latest_eligo_file() -> Optional[Dict]:
    """Get latest Eligo file"""
    logger.info("Searching for Eligo files...")
    
    # Try multiple patterns to match your actual file names
    patterns = [
        # Pattern 1: Files in Eligo/ folder with "Structure Work Tracker"
        re.compile(r"(Structure Work Tracker.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx", re.IGNORECASE),
        # Pattern 2: Files with "eligo" in name (original pattern)
        re.compile(r"(eligo_.*|Eligo_.*|Eligo.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx", re.IGNORECASE),
        # Pattern 3: Any .xlsx file in Eligo/ directory
        re.compile(r".*\.xlsx$", re.IGNORECASE)
    ]
    
    latest = None
    latest_date = None
    
    # Check in the Eligo/ directory
    for key in list_files("Eligo/"):
        fname = key.split("/")[-1]
        
        for pattern in patterns:
            match = pattern.search(fname)
            if not match:
                continue
            
            try:
                # Extract date from filename
                date_match = re.search(r'\((\d{2}-\d{2}-\d{4})\)\.xlsx$', fname)
                if date_match:
                    fdate = datetime.strptime(date_match.group(1), "%d-%m-%Y")
                    
                    if latest_date is None or fdate > latest_date:
                        latest = {
                            "file": fname,
                            "key": key,
                            "date": fdate,
                            "project": "ELIGO",
                            "prefix": "Eligo/"
                        }
                        latest_date = fdate
                        logger.info(f"Found Eligo file: {fname} dated {fdate}")
                break  # Found a match, move to next file
            except Exception as e:
                logger.warning(f"Could not parse date from {fname}: {e}")
                continue
    
    # If no files found in Eligo/, check root directory
    if not latest:
        for key in list_files(""):
            fname = key.split("/")[-1]
            if "Eligo" in fname or "eligo" in fname.lower():
                date_match = re.search(r'\((\d{2}-\d{2}-\d{4})\)\.xlsx$', fname)
                if date_match:
                    try:
                        fdate = datetime.strptime(date_match.group(1), "%d-%m-%Y")
                        if latest_date is None or fdate > latest_date:
                            latest = {
                                "file": fname,
                                "key": key,
                                "date": fdate,
                                "project": "ELIGO",
                                "prefix": ""
                            }
                            latest_date = fdate
                    except:
                        continue
    
    if latest:
        logger.info(f"Selected latest Eligo file: {latest['file']} dated {latest['date']}")
    else:
        logger.warning("No Eligo files found with any pattern")
    
    return latest

def get_latest_veridia_file() -> Optional[Dict]:
    """Get latest Veridia file"""
    logger.info("Searching for Veridia files...")
    
    # Try multiple patterns
    patterns = [
        re.compile(r"(Structure Work Tracker.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx", re.IGNORECASE),
        re.compile(r"(Veridia.*)\((\d{2}-\d{2}-\d{4})\)\.xlsx", re.IGNORECASE),
    ]
    
    latest = None
    latest_date = None
    
    # Check in the Veridia/ directory
    for key in list_files("Veridia/"):
        fname = key.split("/")[-1]
        
        for pattern in patterns:
            match = pattern.search(fname)
            if not match:
                continue
            
            try:
                # Extract date from filename
                date_match = re.search(r'\((\d{2}-\d{2}-\d{4})\)\.xlsx$', fname)
                if date_match:
                    fdate = datetime.strptime(date_match.group(1), "%d-%m-%Y")
                    
                    if latest_date is None or fdate > latest_date:
                        latest = {
                            "file": fname,
                            "key": key,
                            "date": fdate,
                            "project": "VERIDIA",
                            "prefix": "Veridia/"
                        }
                        latest_date = fdate
                        logger.info(f"Found Veridia file: {fname} dated {fdate}")
                break
            except Exception as e:
                logger.warning(f"Could not parse date from {fname}: {e}")
                continue
                
    if latest:
        logger.info(f"Selected latest Veridia file: {latest['file']} dated {latest['date']}")
    else:
        logger.warning("No Veridia files found")
    
    return latest

def get_all_latest_files() -> Dict[str, Optional[Dict]]:
    """Get latest files for all projects"""
    return {
        "ews_lig": get_latest_ews_lig_file(),
        "eligo": get_latest_eligo_file(),
        "veridia": get_latest_veridia_file()
    }

# --------------------------------------------------
# DATA EXTRACTION FUNCTIONS
# --------------------------------------------------

def format_date(value):
    """Format date values consistently with proper format detection (DD-MM-YYYY preferred)"""
    if pd.isna(value):
        return None
    
    try:
        # If it's already a datetime/timestamp
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime("%d-%m-%Y")
        
        s = str(value).strip()
        if not s or s.upper() in ["NAN", "NONE", "NAT", "NULL"]:
            return None

        # Handle Excel date strings like "08-Jul-24"
        if "-" in s and any(month in s.lower() for month in 
                               ["jan", "feb", "mar", "apr", "may", "jun", 
                                "jul", "aug", "sep", "oct", "nov", "dec"]):
            try:
                for fmt in ["%d-%b-%y", "%d-%b-%Y", "%d/%b/%y", "%d/%b/%Y"]:
                    try:
                        dt = datetime.strptime(s, fmt)
                        return dt.strftime("%d-%m-%Y")
                    except:
                        continue
            except:
                pass

        # Remove time portion
        if " " in s:
            s = s.split(" ")[0]
        
        # Handle YYYY-MM-DD (convert to DD-MM-YYYY)
        if "-" in s:
            parts = s.split("-")
            if len(parts[0]) == 4: # YYYY-MM-DD
                try:
                    return datetime.strptime(s, "%Y-%m-%d").strftime("%d-%m-%Y")
                except:
                    pass
            elif len(parts[2]) == 4: # DD-MM-YYYY
                try:
                    return datetime.strptime(s, "%d-%m-%Y").strftime("%d-%m-%Y")
                except:
                    pass

        # Try various formats
        date_formats = [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m-%d-%Y"
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.strftime("%d-%m-%Y")
            except:
                continue
        
        # Excel serial date
        try:
            if isinstance(value, (int, float)) or (isinstance(s, str) and s.replace('.', '', 1).isdigit()):
                val_float = float(value)
                if val_float > 20000: # Reasonable Excel date range
                    dt = datetime(1899, 12, 30) + timedelta(days=val_float)
                    return dt.strftime("%d-%m-%Y")
        except:
            pass
        
        return None
        
    except Exception as e:
        # logger.debug(f"Could not parse date '{value}': {e}")
        return None

def extract_pour_section(df: pd.DataFrame, start_col: int, end_col: int, start_row: int, floor_col_idx: int = None, max_row: int = None) -> Dict:
    """Extract pour data from a specific column range for ELIGO format"""
    pour_data = {}
    
    try:
        if start_col >= df.shape[1]:
            logger.warning(f"Column range {start_col}:{end_col} out of bounds")
            return None
            
        # Use provided floor column or default to start_col
        actual_floor_col = floor_col_idx if floor_col_idx is not None else start_col
        
        # Determine data column offsets relative to start_col
        if floor_col_idx is not None and floor_col_idx != start_col:
             # Shared floor: Data starts at start_col (Baseline)
             baseline_offset = 0
             days_offset = 1
             antic_offset = 2
        else:
             # Local floor: Data starts at start_col + 1 (Baseline)
             baseline_offset = 1
             days_offset = 2
             antic_offset = 3

        max_rows_to_check = 100
        limit_row = min(start_row + max_rows_to_check, len(df))
        if max_row:
            limit_row = min(limit_row, max_row)
        
        for r in range(start_row, limit_row):
            # Extract Floor Name from the actual floor column
            if actual_floor_col >= df.shape[1]: continue
            
            floor_val = df.iloc[r, actual_floor_col]
            floor = str(floor_val).strip() if not pd.isna(floor_val) else None
            
            # Validation
            if not floor: continue
            
            floor_upper = floor.upper()
            is_floor = (
                re.match(r'^\d+\s*F$', floor_upper) or
                re.match(r'^F\d+$', floor_upper) or
                re.match(r'^\d+$', floor_upper) or
                floor_upper in ["GF", "TF", "UF", "LG", "UG", "BG", "ROOF", "FOUNDATION", "PLINTH", "SHEAR"] or
                "FLOOR" in floor_upper or "LEVEL" in floor_upper or "WALL" in floor_upper
            )
            
            if not is_floor: continue
            
            floor_data = {
                "floor": floor,
                "baseline": None,
                "anticipated": None
            }
            
            # Extract Data
            # Baseline
            if start_col + baseline_offset < df.shape[1]:
                val = df.iloc[r, start_col + baseline_offset]
                floor_data["baseline"] = format_date(val)
                
            # Days
            if start_col + days_offset < df.shape[1]:
                val = df.iloc[r, start_col + days_offset]
                if not pd.isna(val):
                     try:
                         if isinstance(val, (int, float)):
                             floor_data["days"] = int(val)
                         else:
                             days_str = str(val).strip()
                             if days_str:
                                 days_clean = re.sub(r'[^\d\.\-]', '', days_str)
                                 if days_clean:
                                     floor_data["days"] = int(float(days_clean))
                     except: pass

            # Anticipated
            if start_col + antic_offset < df.shape[1]:
                val = df.iloc[r, start_col + antic_offset]
                floor_data["anticipated"] = format_date(val)
            
            if floor_data["baseline"] or floor_data.get("days") or floor_data["anticipated"]:
                pour_data[floor] = floor_data
        
        logger.info(f"Extracted {len(pour_data)} floors from columns {start_col} (Floor: {actual_floor_col})")
        return pour_data if pour_data else None
        
    except Exception as e:
        logger.error(f"Error extracting pour section at {start_col}: {e}")
        return None

def excel_column_to_index(column_letter: str) -> int:
    """Convert Excel column letter (A, B, C, ...) to zero-based index."""
    column_letter = column_letter.upper()
    result = 0
    for char in column_letter:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1  # Convert to zero-based


def extract_tower_data(df: pd.DataFrame, tower_name: str, tower_start_col: int, start_row: int, max_col: int = None, max_row: int = None) -> Dict:
    """Extract all pours for a tower"""
    print(f"DEBUG: extract_tower_data {tower_name} start_row={start_row} max_row={max_row}")
    tower_data = {
        "tower": tower_name,
        "pours": {}
    }
    
    try:
        # Verify/Adjust start column: find the first column with Floor data
        actual_start_col = tower_start_col
        found_start = False
        
        # Check a wider range of rows for the initial floor column
        check_rows_limit = 30 # Increased to handle gaps
        limit_row_check = min(start_row + check_rows_limit, len(df))
        if max_row:
            limit_row_check = min(limit_row_check, max_row)
        
        for offset in range(5): # Check up to 5 cols from header
            c = tower_start_col + offset
            if c >= df.shape[1]: break
            if max_col and c >= max_col: break
            
            # Check if this column has floor data
            has_floor_data = False
            for r in range(start_row, limit_row_check):
                val = str(df.iloc[r, c]).strip()
                if (re.match(r'^\d+\s*F$', val, re.IGNORECASE) or 
                    re.match(r'^F\d+$', val, re.IGNORECASE) or
                    re.match(r'^\d+$', val) or
                    val.upper() in ["GF", "TF", "UF", "LG", "UG", "BG", "ROOF", "FOUNDATION", "PLINTH", "SHEAR"]):
                    has_floor_data = True
                    break
            
            if has_floor_data:
                actual_start_col = c
                found_start = True
                break
        
        if not found_start:
            logger.warning(f"Could not find Floor data column for {tower_name} starting at {tower_start_col}")
            # Fallback to original but log warning
            actual_start_col = tower_start_col

        primary_floor_col = actual_start_col
        pour_starts = [actual_start_col]
        
        # Look for additional pours
        # We scan columns to the right. 
        
        col_idx = actual_start_col + 3  # Start looking after the first pour (min 3 cols)
        
        limit_col = max_col if max_col else df.shape[1]
        
        while col_idx < limit_col - 3:
            has_data = False
            header_found = False
            
            # Check header row area for "Pour" or "Baseline"
            header_row_approx = max(0, start_row - 5)
            for r in range(header_row_approx, start_row + 5):
                 if r < len(df) and (not max_row or r < max_row):
                     val = str(df.iloc[r, col_idx]).upper()
                     if "POUR" in val or "BASELINE" in val:
                         header_found = True
                         break

            # Check for Floor markers (if local floor)
            for check_row in range(start_row, limit_row_check):
                if col_idx >= limit_col: break
                cell_val = df.iloc[check_row, col_idx]
                if pd.isna(cell_val): continue
                
                cell_str = str(cell_val).strip().upper()
                if (re.match(r'^\d+\s*F$', cell_str) or 
                    re.match(r'^F\d+$', cell_str) or
                    cell_str in ["GF", "TF", "UF", "FOUNDATION", "PLINTH"]):
                    has_data = True
                    break
            
            if header_found or has_data:
                 pour_starts.append(col_idx)
                 if has_data: # Local floor likely
                     col_idx += 4
                 else:
                     col_idx += 3
            else:
                 col_idx += 1
        
        # Remove duplicate pour starts and sort
        pour_starts = sorted(list(set(pour_starts)))
        
        # Filter too-close starts
        filtered_starts = []
        if pour_starts:
            filtered_starts.append(pour_starts[0])
            for p in pour_starts[1:]:
                if p - filtered_starts[-1] >= 3: 
                    filtered_starts.append(p)
        pour_starts = filtered_starts
        
        # Extract data for each pour
        pour_num = 1
        for start_col in pour_starts:
            # Check if we exceeded max_col (double check)
            if max_col and start_col >= max_col:
                continue

            # Determine if this pour has a local floor column
            is_local_floor = False
            
            # Check header
            for r in range(max(0, start_row - 5), start_row + 5):
                if r < len(df):
                     val = str(df.iloc[r, start_col]).upper()
                     if "FLOOR" in val:
                         is_local_floor = True
                         break
            
            # Check data if header ambiguous
            if not is_local_floor:
                for r in range(start_row, min(start_row + 20, len(df))):
                    val = str(df.iloc[r, start_col]).strip()
                    if re.match(r'^\d+F$', val) or val.upper() in ["GF", "TF", "FOUNDATION"]:
                        is_local_floor = True
                        break
            
            floor_col_to_use = start_col if is_local_floor else primary_floor_col
            width = 4 if is_local_floor else 3
            
            # Ensure we don't read past limit
            if max_col and start_col + width > max_col:
                 # Adjust width if strictly needed, or skip?
                 # Usually if start_col is valid, width should be fine unless it overlaps next tower
                 pass

            pour_name = f"Pour_{pour_num}"
            pour_data = extract_pour_section(df, start_col, start_col + width, start_row, floor_col_to_use, max_row=max_row)

            
            if pour_data:
                tower_data["pours"][pour_name] = pour_data
                logger.info(f"Found {pour_name} for {tower_name} at col {start_col} (Floor: {floor_col_to_use})")
            
            pour_num += 1
        
        return tower_data if tower_data["pours"] else None
        
    except Exception as e:
        logger.error(f"Error extracting tower data for {tower_name}: {e}")
        tower_data["error"] = str(e)
        return tower_data

def extract_ews_lig_schedule_data(df: pd.DataFrame) -> Dict:
    """Extract structured schedule data from EWS LIG sheets - DYNAMIC VERSION"""
    
    result = {
        "towers": {},
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns) if hasattr(df, 'columns') else 0
        }
    }
    
    try:
        # 1. Find the header row (look for "EWS Tower" or similar)
        header_row = None
        for i in range(min(50, len(df))):
            row_str = " ".join([str(x).strip() for x in df.iloc[i] if not pd.isna(x)])
            if "EWS TOWER" in row_str.upper() or ("TOWER" in row_str.upper() and "POUR" in row_str.upper()):
                header_row = i
                break
        
        if header_row is None:
            header_row = 0
            
        logger.info(f"EWS LIG: Found header at row {header_row}")
        
        data_start_row = header_row + 1
        
        # 2. Find LIG Start Row (look for "LIG" separator)
        lig_start_row = None
        for i in range(header_row + 10, min(header_row + 100, len(df))):
            row_str = " ".join([str(x).strip() for x in df.iloc[i] if not pd.isna(x)])
            if "LIG" in row_str.upper() and ("TOWER" in row_str.upper() or "POUR" in row_str.upper()):
                lig_start_row = i
                break
        
        if lig_start_row is None:
            lig_start_row = data_start_row + 25 # Fallback
            logger.warning("EWS LIG: Could not find LIG start row, using fallback")
        else:
            logger.info(f"EWS LIG: Found LIG start at row {lig_start_row}")
        
        towers_found = {"EWS": [], "LIG": []}
        
        # 3. Find EWS Towers by Header Scan
        # Look in header_row and surrounding rows
        scan_rows = range(max(0, header_row - 2), header_row + 3)
        
        for col in range(df.shape[1]):
            # Check for "EWS Tower X" or just "Tower X" in EWS section
            found_tower = False
            for r in scan_rows:
                if r >= len(df): continue
                val = str(df.iloc[r, col]).strip().upper()
                if "EWS TOWER" in val or ("TOWER" in val and "LIG" not in val):
                    towers_found["EWS"].append(col)
                    found_tower = True
                    break
            
            # If header scanning fails, check for Pour headers which usually follow towers
            if not found_tower:
                 # Logic: if we find "Pour 1" and it's not part of an existing tower block...
                 pass

        # If scan failed (e.g. merged cells or weird layout), fall back to data scanning
        if not towers_found["EWS"]:
            logger.info("EWS LIG: No EWS Tower headers found, trying data scan")
            col = 0
            while col < df.shape[1]:
                col_data = df.iloc[data_start_row:min(data_start_row+3, len(df)), col]
                if not col_data.isna().all():
                    # Heuristic: Check if it looks like floor data
                    sample = str(col_data.iloc[0]).strip().upper()
                    if "F" in sample or sample.isdigit():
                        towers_found["EWS"].append(col)
                        col += 12 # Skip at least some columns
                    else:
                        col += 1
                else:
                    col += 1

        # 4. Find LIG Towers by Header Scan
        scan_rows_lig = range(lig_start_row, lig_start_row + 5)
        for col in range(df.shape[1]):
            for r in scan_rows_lig:
                if r >= len(df): continue
                val = str(df.iloc[r, col]).strip().upper()
                if "LIG TOWER" in val or ("TOWER" in val and "EWS" not in val):
                    towers_found["LIG"].append(col)
                    break
        
        # Remove duplicates and sort
        towers_found["EWS"] = sorted(list(set(towers_found["EWS"])))
        towers_found["LIG"] = sorted(list(set(towers_found["LIG"])))
        
        # Filter out columns that are too close (merged cells often trigger multiple hits)
        def filter_close_cols(cols, min_dist=4):
            if not cols: return []
            res = [cols[0]]
            for c in cols[1:]:
                if c - res[-1] >= min_dist:
                    res.append(c)
            return res

        towers_found["EWS"] = filter_close_cols(towers_found["EWS"])
        towers_found["LIG"] = filter_close_cols(towers_found["LIG"])

        logger.info(f"EWS LIG: Towers found - EWS: {towers_found['EWS']}, LIG: {towers_found['LIG']}")

        # Extract EWS towers
        sorted_ews_cols = sorted(towers_found["EWS"])
        for i, tower_col in enumerate(sorted_ews_cols):
            tower_num = i + 1
            tower_name = f"EWS Tower {tower_num}"
            # Try to get actual name from header if possible
            try:
                header_val = str(df.iloc[header_row, tower_col]).strip()
                if "TOWER" in header_val.upper():
                    tower_name = header_val
            except:
                pass
            
            # Determine max_col
            max_col = None
            if i < len(sorted_ews_cols) - 1:
                max_col = sorted_ews_cols[i+1]
            else:
                max_col = df.shape[1]
                
            tower_data = extract_tower_data(df, tower_name, tower_col, data_start_row, max_col=max_col, max_row=lig_start_row)
            if tower_data and tower_data.get("pours"):
                tower_key = tower_name.replace(" ", "_")
                tower_data_with_delay = add_delay_analysis_to_tower_data(tower_data)
                result["towers"][tower_key] = tower_data_with_delay
        
        # Extract LIG towers
        sorted_lig_cols = sorted(towers_found["LIG"])
        for i, tower_col in enumerate(sorted_lig_cols):
            tower_num = i + 1
            tower_name = f"LIG Tower {tower_num}"
            try:
                header_val = str(df.iloc[lig_start_row, tower_col]).strip()
                if "TOWER" in header_val.upper():
                    tower_name = header_val
            except:
                pass

            # Determine max_col
            max_col = None
            if i < len(sorted_lig_cols) - 1:
                max_col = sorted_lig_cols[i+1]
            else:
                max_col = df.shape[1]

            tower_data = extract_tower_data(df, tower_name, tower_col, lig_start_row + 1, max_col=max_col) # +1 for data start
            if tower_data and tower_data.get("pours"):
                tower_key = tower_name.replace(" ", "_")
                tower_data_with_delay = add_delay_analysis_to_tower_data(tower_data)
                result["towers"][tower_key] = tower_data_with_delay
        
        result["summary"]["towers_extracted"] = len(result["towers"])
        result["summary"]["lig_start_row"] = lig_start_row
        result["summary"]["towers_scanned"] = towers_found
        
        # Calculate overall project metrics
        all_activities = []
        for tower_data in result["towers"].values():
            if "activities" in tower_data:
                all_activities.extend(tower_data["activities"])
        
        result["delay_metrics"] = calculate_delay_metrics(all_activities)
        
    except Exception as e:
        result["error"] = f"Error extracting data: {str(e)}"
        logger.error(f"EWS LIG Extraction Error: {e}", exc_info=True)
    
    return result

def extract_eligo_slab_data(df: pd.DataFrame, sheet_name: str = None) -> Dict:
    """Extract structured slab cycle data from Eligo sheets - DYNAMIC VERSION"""
    
    result = {
        "project": "Eligo",
        "towers": {},
        "sheet": sheet_name or "Unknown",
        "summary": {}
    }
    
    try:
        # 1. Find the header row with "ELIGO SLAB CYCLE" or similar to locate the main table
        header_row = None
        for i in range(min(50, len(df))):
            row_str = " ".join([str(x).strip() for x in df.iloc[i] if not pd.isna(x)])
            if "ELIGO SLAB CYCLE" in row_str.upper() or "TOWER" in row_str.upper():
                header_row = i
                break
        
        if header_row is None:
            header_row = 0
            
        logger.info(f"Starting search from row {header_row}")

        # 2. Find Tower headers dynamically
        # We look for "Tower F", "Tower G", "Tower H" in the first few rows
        tower_cols = {}
        tower_row_idx = -1
        
        # Scan a few rows to find where "Tower" is mentioned
        for i in range(header_row, min(header_row + 10, len(df))):
            for col in range(df.shape[1]):
                cell_val = str(df.iloc[i, col]).strip()
                if "Tower" in cell_val:
                    # Normalize tower name
                    if "F" in cell_val:
                        tower_name = "Tower F"
                    elif "G" in cell_val:
                        tower_name = "Tower G"
                    elif "H" in cell_val:
                        tower_name = "Tower H"
                    else:
                        continue
                        
                    if tower_name not in tower_cols:
                        tower_cols[tower_name] = col
                        tower_row_idx = i
        
        if not tower_cols:
            logger.warning("No Tower headers found. Falling back to hardcoded defaults.")
            # Fallback based on user observation: F=0, G=10, H=26
            tower_cols = {"Tower F": 0, "Tower G": 10, "Tower H": 26} 

        logger.info(f"Found towers at columns: {tower_cols}")

        # 3. For each tower, find Pours
        for tower_name, start_col in tower_cols.items():
            tower_data = {
                "tower": tower_name,
                "pours": {}
            }
            
            # Determine the range of columns for this tower
            sorted_starts = sorted(tower_cols.values())
            try:
                current_idx = sorted_starts.index(start_col)
                if current_idx < len(sorted_starts) - 1:
                    end_col = sorted_starts[current_idx + 1]
                else:
                    end_col = df.shape[1]
            except ValueError:
                end_col = df.shape[1]
                
            logger.info(f"Scanning {tower_name} from col {start_col} to {end_col}")

            # Find "Pour" headers within this range
            pour_map = {} # Pour Name -> Column Index
            
            search_start_row = tower_row_idx + 1 if tower_row_idx != -1 else header_row
            
            # Identify a primary floor column for this tower (shared across pours)
            primary_floor_col = start_col
            floor_col_found = False
            
            # Scan a few columns/rows to find "Floor" header
            for c in range(start_col, min(start_col + 5, df.shape[1])):
                for r in range(search_start_row, min(search_start_row + 5, len(df))):
                    val = str(df.iloc[r, c]).strip().upper()
                    if "FLOOR" in val:
                        primary_floor_col = c
                        floor_col_found = True
                        break
                if floor_col_found: break
            
            if floor_col_found:
                logger.info(f"  Primary floor column for {tower_name}: {primary_floor_col}")
            
            # Scan rows for Pour headers
            for r in range(search_start_row, min(search_start_row + 10, len(df))):
                for c in range(start_col, end_col):
                    cell_val = str(df.iloc[r, c]).strip()
                    if "Pour" in cell_val:
                        # Extract pour number
                        pour_num_match = re.search(r'(\d+)', cell_val)
                        if pour_num_match:
                            pour_num = pour_num_match.group(1)
                            pour_key = f"Pour_{pour_num}"
                            if pour_key not in pour_map:
                                pour_map[pour_key] = {"col": c, "row": r}

            logger.info(f"  Found pours for {tower_name}: {list(pour_map.keys())}")

            # If no pours found, try to infer them based on column spacing (every 4 columns)
            if not pour_map:
                logger.info(f"  No explicit Pour headers found for {tower_name}, trying default spacing")
                # Default logic: Assume pours start at start_col, start_col+4, etc.
                # Adjust based on expected pour counts: F=2, G=3, H=7
                expected_pours = 2 if "F" in tower_name else (3 if "G" in tower_name else 7)
                
                # Check actual structure using "Baseline" headers if possible
                current_c = start_col
                found_pours_count = 0
                
                # Try to find "Baseline" or "Actual" to confirm a pour block exists
                for p_idx in range(expected_pours):
                    # Heuristic: If we find a date-like column or "Baseline" header, it's a pour
                    pour_key = f"Pour_{p_idx + 1}"
                    pour_map[pour_key] = {"col": current_c, "row": search_start_row + 1}
                    
                    # Advance to next pour (usually 4 cols, but check for gaps)
                    # We might need to skip empty columns
                    current_c += 4
                    # Simple adjustment for potential gaps
                    while current_c < end_col and df.iloc[search_start_row:search_start_row+5, current_c].isna().all():
                        current_c += 1

            # Extract data for each pour
            for pour_name, pour_info in pour_map.items():
                pour_col = pour_info["col"]
                pour_header_row = pour_info["row"]
                
                # Identify sub-columns: Floor, Baseline, Anticipated/Actual
                floor_col = pour_col
                baseline_col = pour_col + 1
                days_col = pour_col + 2
                actual_col = pour_col + 3
                
                # Verify headers if possible
                header_check_row = pour_header_row + 1
                if header_check_row < len(df):
                    # Check if current floor_col is actually a floor column
                    # If the cell at floor_col is "Baseline" or "Anticipated", it's likely sharing the primary floor column
                    # or if it's empty
                    
                    is_local_floor_col = False
                    try:
                        # Check header row and a few rows below
                        header_val = str(df.iloc[header_check_row, floor_col]).upper()
                        if "FLOOR" in header_val:
                            is_local_floor_col = True
                        else:
                            # Check data values
                            for r_off in range(1, 4):
                                if header_check_row + r_off < len(df):
                                    val = str(df.iloc[header_check_row + r_off, floor_col]).strip()
                                    if re.match(r'^\d+F$', val) or val in ["GF", "TF", "UG"]:
                                        is_local_floor_col = True
                                        break
                    except:
                        pass
                        
                    if not is_local_floor_col and floor_col_found:
                         # Fallback to primary floor column
                         floor_col = primary_floor_col
                    
                    for offset in range(5):
                        check_c = pour_col + offset
                        if check_c >= df.shape[1]: continue
                        val = str(df.iloc[header_check_row, check_c]).upper()
                        if "BASE" in val: baseline_col = check_c
                        if "ACTUAL" in val or "ANTICIPATED" in val: actual_col = check_c
                        if "DAYS" in val: days_col = check_c

                pour_data = {}
                data_start = header_check_row + 1
                
                # Scan down rows
                row_idx = data_start
                empty_consecutive = 0
                
                while row_idx < len(df):
                    if empty_consecutive > 5: break
                    
                    if floor_col >= df.shape[1]: break
                    
                    floor_val = df.iloc[row_idx, floor_col]
                    if pd.isna(floor_val):
                        empty_consecutive += 1
                        row_idx += 1
                        continue
                    
                    floor_str = str(floor_val).strip()
                    
                    # Validate it's a floor name
                    is_floor = (
                        re.match(r'^\d+\s*F$', floor_str, re.IGNORECASE) or
                        re.match(r'^F\d+$', floor_str, re.IGNORECASE) or
                        re.match(r'^\d+$', floor_str) or
                        floor_str.upper() in ["GF", "TF", "UF", "LG", "UG", "BG", "LOWER GROUND", "ROOF"]
                    )
                    
                    if not is_floor:
                        empty_consecutive += 1 
                        row_idx += 1
                        continue

                    empty_consecutive = 0
                    
                    # Extract dates (Format: dd-mm-yyyy)
                    baseline = None
                    actual = None
                    
                    def format_dmY(val):
                        if pd.isna(val): return None
                        if isinstance(val, (datetime, pd.Timestamp)):
                            return val.strftime("%d-%m-%Y")
                        
                        s = str(val).strip()
                        # Try parsing common formats to normalize
                        try:
                            # Handle 2023-05-13 00:00:00
                            if " " in s:
                                date_part = s.split(" ")[0]
                                if "-" in date_part:
                                    parts = date_part.split("-")
                                    if len(parts[0]) == 4: # YYYY-MM-DD
                                        return datetime.strptime(date_part, "%Y-%m-%d").strftime("%d-%m-%Y")
                            
                            # Handle YYYY-MM-DD
                            if "-" in s:
                                parts = s.split("-")
                                if len(parts[0]) == 4:
                                    return datetime.strptime(s, "%Y-%m-%d").strftime("%d-%m-%Y")
                        except:
                            pass
                        return s

                    if baseline_col < df.shape[1]:
                        baseline = format_dmY(df.iloc[row_idx, baseline_col])
                    
                    if actual_col < df.shape[1]:
                        actual = format_dmY(df.iloc[row_idx, actual_col])


                    # Store if valid
                    if baseline or actual:
                        pour_data[floor_str] = {
                            "floor": floor_str,
                            "baseline": baseline,
                            "anticipated": actual
                            # "days": removed per user request
                        }
                        
                    row_idx += 1
                
                if pour_data:
                    tower_data["pours"][pour_name] = pour_data
                    logger.info(f"    Extracted {len(pour_data)} floors for {pour_name}")

            if tower_data["pours"]:
                 tower_data_with_delay = add_delay_analysis_to_tower_data(tower_data)
                 result["towers"][tower_name.replace(" ", "_")] = tower_data_with_delay
                 logger.info(f"✓ Added {tower_name} with {len(tower_data['pours'])} pours")

        # Calculate summaries
        total_pours = sum(len(t.get("pours", {})) for t in result["towers"].values())
        total_floors = sum(
            sum(len(pour_data) for pour_data in t.get("pours", {}).values())
            for t in result["towers"].values()
        )
        
        result["summary"] = {
            "towers_found": list(result["towers"].keys()),
            "total_towers": len(result["towers"]),
            "total_pours": total_pours,
            "total_floors": total_floors
        }
        
        # Calculate delay metrics
        all_activities = []
        for tower_data in result["towers"].values():
            if "activities" in tower_data:
                all_activities.extend(tower_data["activities"])
        
        result["delay_metrics"] = calculate_delay_metrics(all_activities)
        
        return result

    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
        logger.error(f"Error extracting Eligo data: {e}\n{traceback.format_exc()}")
        return result

# --------------------------------------------------
# HARDCODED VERIDIA CONFIG
# --------------------------------------------------
VERIDIA_CONFIG = {
    "Tower 5": {
        "col_start": 105, # DB
        "sides": {
            "South": {"header_row": 1, "data_start": 2},  # Adjusted to 0-based relative to block or absolute? Using absolute indices from inspection
            "North": {"header_row": 10, "data_start": 11} # Row 11 in Excel is Index 10
        },
        "modules": ["M7", "M6", "M5", "M4", "M3", "M2", "M1"],
        "module_col_offset": 1, 
        "module_width": 4
    },
    "Tower 6": {
        "col_start": 165, # FJ
        "sides": {
            "South": {"header_row": 1, "data_start": 2},
            "North": {"header_row": 10, "data_start": 11}
        },
        "modules": ["M7", "M6", "M5", "M4", "M3", "M2", "M1"],
        "module_col_offset": 1,
        "module_width": 4
    },
    "Tower 7": {
        "col_start": 135, # EF
        "sides": {
            "East": {"header_row": 1, "data_start": 2},
            "West": {"header_row": 10, "data_start": 11}
        },
        "modules": ["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
        "module_col_offset": 1,
        "module_width": 4
    }
}

def extract_veridia_data(df: pd.DataFrame, sheet_name: str = None) -> Dict:
    """Extract structured data from Veridia sheets using semi-dynamic search."""
    result = {
        "project": "Veridia",
        "towers": {},
        "sheet": sheet_name or "Unknown",
        "summary": {}
    }
    
    try:
        # Find Tower Row (Look for "Tower 5", "Tower 6", "Tower 7")
        tower_row_idx = None
        for idx, row in df.iterrows():
            row_str = str(row.values).upper()
            if "TOWER" in row_str and ("5" in row_str or "6" in row_str or "7" in row_str):
                tower_row_idx = idx
                break
        
        if tower_row_idx is None:
            # Fallback to absolute index 1 (Row 2) if search fails
            tower_row_idx = 1
        
        logger.info(f"Veridia extraction: Found Tower header at row index {tower_row_idx}")

        for tower_name, config in VERIDIA_CONFIG.items():
            tower_key = tower_name.replace(" ", "_")
            result["towers"][tower_key] = {"floors": {}, "activities": [], "tower": tower_name}
            
            col_base = config["col_start"]
            
            # Dynamic Side Detection
            # We know the column start. We search down from tower_row_idx for side headers.
            
            # Potential sides to look for
            potential_sides = ["South", "North", "East", "West"]
            
            # Map side -> start_row
            sides_found = {}
            
            # Scan a range of rows (e.g., 50 rows)
            scan_limit = min(tower_row_idx + 100, len(df))
            
            for r in range(tower_row_idx, scan_limit):
                # Check the header row area (around col_base)
                # The side name usually appears in the header row, e.g., "M7 South"
                # We check the first module column or the floor column?
                # Usually it spans across modules.
                
                # Check a few columns around col_base
                row_vals = []
                for c in range(col_base, col_base + 20): # Check first few columns
                    if c < df.shape[1]:
                        row_vals.append(str(df.iloc[r, c]).upper())
                
                row_text = " ".join(row_vals)
                
                for side in potential_sides:
                    if side.upper() in row_text and side not in sides_found:
                         # Found a header for this side
                         # But wait, "M7 South" appears.
                         # We need to distinguish between South and North.
                         # If we find "SOUTH", we record it.
                         sides_found[side] = r
            
            # If no sides found dynamically, fallback to config if available (but we want to fix "only showing South")
            # If "South" is found but "North" is not, maybe we need to look harder.
            
            # Process all sides found dynamically, plus any configured ones (as fallback)
            expected_sides = set(sides_found.keys()) | set(config["sides"].keys())
            
            for side_name in expected_sides:
                header_r = sides_found.get(side_name)
                
                if header_r is None:
                     # Try to find it again with more specific search if missed
                     pass
                     # Fallback to hardcoded offset if dynamic fail
                     if side_name in config["sides"]:
                         offset = config["sides"][side_name]["header_row"]
                         header_r = tower_row_idx + offset
                     else:
                         # Not found and not in config
                         continue
                
                if header_r >= len(df): continue
                
                data_start_r = header_r + 1
                
                logger.info(f"Processing {tower_name} - {side_name} starting at row {data_start_r}")
                
                # Iterate data rows
                r = data_start_r
                while r < len(df):
                    # Check if we hit another header (e.g. North header while processing South)
                    # or empty floor
                    
                    floor_val = df.iloc[r, col_base]
                    floor_str = str(floor_val).strip()
                    
                    # Filter out unwanted milestone rows (starting with OW or containing Unique ID)
                    if floor_str.upper().startswith("OW") or "UNIQUE ID" in floor_str.upper():
                        r += 1
                        continue
                    
                    # Stop if we hit a known side name in the floor column or nearby
                    is_new_header = False
                    for other_side in potential_sides:
                        if other_side != side_name and other_side.upper() in floor_str.upper():
                            is_new_header = True
                            break
                    
                    if is_new_header:
                        break

                    if pd.isna(floor_val) or floor_str == "" or "#######" in floor_str:
                        # Check if it's just an empty row or end of block
                        # If next row is also empty, likely end.
                        # If next row has "North", definitely end.
                        
                        # Peek ahead
                        if r + 1 < len(df):
                            next_val = str(df.iloc[r+1, col_base]).strip()
                            if any(s.upper() in next_val.upper() for s in potential_sides):
                                break
                        
                        # If truly empty
                        if floor_str == "":
                             # Allow 1 empty row gap?
                             # But usually Veridia data is contiguous.
                             break
                    
                    # Also check if floor_val is a valid floor (optional but safer)
                    # Veridia floors: 1F, 2F...
                    
                    floor_name = floor_str
                    
                    # Iterate Modules
                    current_col = col_base + config["module_col_offset"]
                    
                    for module_name in config["modules"]:
                        if current_col + 2 >= df.shape[1]:
                            break
                            
                        baseline_date_val = df.iloc[r, current_col]
                        duration_val = df.iloc[r, current_col + 1]
                        anticipated_date_val = df.iloc[r, current_col + 2]
                        
                        # Skip if dates are empty
                        if pd.isna(baseline_date_val) and pd.isna(anticipated_date_val):
                             current_col += config["module_width"]
                             continue
                             
                        # Parse Dates
                        baseline_date = calculate_delay_days_helper_parse(baseline_date_val)
                        anticipated_date = calculate_delay_days_helper_parse(anticipated_date_val)
                        
                        # Calculate Delay
                        delay_days = 0
                        
                        if baseline_date and anticipated_date:
                            if isinstance(baseline_date, datetime) and isinstance(anticipated_date, datetime):
                                delta = anticipated_date - baseline_date
                                delay_days = delta.days
                        
                        # Create Activity
                        activity_name = f"{module_name} {side_name} {floor_name}"
                        
                        activity = {
                            "tower": tower_name,
                            "floor": floor_name,
                            "module": module_name,
                            "side": side_name,
                            "activity": activity_name,
                            "baseline_date": baseline_date.strftime("%Y-%m-%d") if baseline_date else None,
                            "anticipated_date": anticipated_date.strftime("%Y-%m-%d") if anticipated_date else None,
                            "duration": duration_val if not pd.isna(duration_val) else 0,
                            "delay_days": delay_days,
                            "status": "Delayed" if delay_days > 0 else "On Time"
                        }
                        
                        result["towers"][tower_key]["activities"].append(activity)
                        
                        # Add to floors (Veridia structure)
                        if floor_name not in result["towers"][tower_key].get("floors", {}):
                            if "floors" not in result["towers"][tower_key]:
                                result["towers"][tower_key]["floors"] = {}
                            result["towers"][tower_key]["floors"][floor_name] = {}
                        
                        # We can group by Module in the 'floors' dict
                        module_key = f"{module_name} {side_name}"
                        result["towers"][tower_key]["floors"][floor_name][module_key] = activity
                        
                        current_col += config["module_width"]
                    
                    r += 1

        # Calculate metrics
        for t_key, t_data in result["towers"].items():
            t_data["delay_metrics"] = calculate_delay_metrics(t_data["activities"])

        return result

    except Exception as e:
        logger.error(f"Error extracting Veridia data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        result["error"] = str(e)
        return result

def calculate_delay_days_helper_parse(date_val):
    """Helper to parse dates safely."""
    if pd.isna(date_val):
        return None
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, pd.Timestamp):
        return date_val.to_pydatetime()
    try:
        return pd.to_datetime(date_val).to_pydatetime()
    except:
        return None


def strip_severity_from_response(obj: Any) -> Any:
    """Recursively remove response-only fields without affecting filtering."""
    hidden_keys = {
        "severity",
        "activities",
        "filtered_activities",
        "tower_key",
        "tower_display_name",
    }
    if isinstance(obj, dict):
        return {
            key: strip_severity_from_response(value)
            for key, value in obj.items()
            if key not in hidden_keys
        }
    if isinstance(obj, list):
        return [strip_severity_from_response(item) for item in obj]
    return obj


def shape_simple_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact API response with only user-facing result fields."""
    def simplify_summary(summary: Any) -> Dict[str, Any]:
        """Keep summary compact and expose only avg_delay from delay metrics."""
        if not isinstance(summary, dict):
            return {}

        simplified_summary = dict(summary)
        delay_metrics = simplified_summary.get("delay_metrics")

        if isinstance(delay_metrics, dict):
            simplified_summary["delay_metrics"] = {
                "avg_delay": delay_metrics.get("avg_delay", 0.0)
            }

        return simplified_summary

    simplified = {
        "status": response.get("status", "success"),
        "intent": response.get("intent"),
        "query_type": response.get("query_type"),
        "filters_applied": response.get("filters_applied", {}),
        "results": {},
    }

    raw_results = response.get("results", {})
    if not isinstance(raw_results, dict):
        return simplified

    for project_key, project_data in raw_results.items():
        if not isinstance(project_data, dict):
            continue

        if project_data.get("status") == "error":
            simplified["results"][project_key] = {
                "error": project_data.get("error", "Unknown error")
            }
            continue

        result_block = project_data.get("results", {})
        towers_block = result_block.get("towers", {}) if isinstance(result_block, dict) else {}

        tower_names = []
        if isinstance(towers_block, dict):
            for tower_key, tower_info in towers_block.items():
                if isinstance(tower_info, dict):
                    tower_names.append(tower_info.get("tower", tower_key))
                else:
                    tower_names.append(tower_key)

        simplified["results"][project_key] = {
            "project": project_data.get("project", project_key),
            "towers": tower_names,
            "summary": simplify_summary(result_block.get("summary", {})),
        }

        if "breakdown" in result_block:
            simplified["results"][project_key]["breakdown"] = result_block.get("breakdown")

    return simplified



# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------

def load_dataframe(key: str, sheet_name: str) -> pd.DataFrame:
    """Load DataFrame preserving all data with flexible sheet name matching"""
    obj = cos.get_object(Bucket=COS_BUCKET, Key=key)
    data = obj["Body"].read()
    
    with pd.ExcelFile(BytesIO(data), engine="openpyxl") as xls:
        if sheet_name in xls.sheet_names:
            actual_sheet = sheet_name
        else:
            actual_sheet = None
            for s in xls.sheet_names:
                if s.strip().lower() == sheet_name.strip().lower():
                    actual_sheet = s
                    break
            
            if not actual_sheet:
                for s in xls.sheet_names:
                    if sheet_name.strip().lower() in s.strip().lower():
                        actual_sheet = s
                        break
            
            if not actual_sheet:
                clean_sheet_name = ' '.join(sheet_name.split())
                for s in xls.sheet_names:
                    if clean_sheet_name.strip().lower() in s.strip().lower():
                        actual_sheet = s
                        break
            
            if not actual_sheet:
                raise ValueError(f"Sheet not found: '{sheet_name}'. Available sheets: {xls.sheet_names}")
        
        df = pd.read_excel(
            xls,
            sheet_name=actual_sheet,
            header=None,
            engine='openpyxl'
        )
        
        df = df.dropna(how='all')
        df = df.reset_index(drop=True)
    
    return df

# --------------------------------------------------
# QUERY PROCESSOR (Similar to app.py)
# --------------------------------------------------
def extract_tower_letter(tower_name: str) -> str:
    """Extract tower letter from tower name (for Eligo)."""
    if not tower_name:
        return ""
    
    tower_lower = tower_name.lower().strip()
    
    # Look for f, g, h in the name
    for letter in ['f', 'g', 'h']:
        if letter in tower_lower:
            return letter
    
    # If "tower" in name, get what comes after
    if "tower" in tower_lower:
        parts = tower_lower.split("tower")
        if len(parts) > 1:
            after_tower = parts[1].replace(" ", "").replace("_", "").strip()
            if after_tower and after_tower[0] in ['f', 'g', 'h']:
                return after_tower[0]
    
    return ""


class QueryProcessor:
    """Process user queries with LLM assistance."""
    
    @staticmethod
    def fallback_parse_query(query: str) -> IntentResponse:
        """Fallback parser when LLM is not available."""
        query_lower = query.lower()
        
        # Determine project
        project = "all"
        if "eligo" in query_lower:
            project = "eligo"
        elif "ews" in query_lower or "lig" in query_lower:
            project = "ews-lig"
        elif "veridia" in query_lower or "veridiea" in query_lower or "veridea" in query_lower:
            project = "veridia"
        
        # Determine intent
        intent = UserIntent.GET_DELAYS
        
        # Check for specific data keywords that imply GET_DELAYS even if "list" is present
        data_keywords = ["delayed", "delay", "completed", "pending", "ongoing", "status", "progress", 
                         "floor", "flat", "module", "pour", "slab"]
        has_data_keyword = any(k in query_lower for k in data_keywords)
        
        if "summary" in query_lower or "overview" in query_lower:
            intent = UserIntent.GET_SUMMARY
        elif "compare" in query_lower:
            intent = UserIntent.COMPARE_TOWERS
        elif "critical" in query_lower:
            intent = UserIntent.FIND_CRITICAL
        elif "status" in query_lower and not has_data_keyword: # Only if generic status request
            intent = UserIntent.GET_STATUS
        elif "list" in query_lower and "project" in query_lower:
            intent = UserIntent.LIST_PROJECTS
        elif "list" in query_lower and "tower" in query_lower and not has_data_keyword:
            intent = UserIntent.LIST_TOWERS
        
        # Determine query type
        query_type = QueryType.DELAY_ANALYSIS
        if "module" in query_lower:
             query_type = QueryType.MODULE_WISE
        elif "floor" in query_lower:
            query_type = QueryType.FLOOR_WISE
        elif "pour" in query_lower:
            query_type = QueryType.POUR_WISE
        elif "tower" in query_lower and "wise" in query_lower:
            query_type = QueryType.TOWER_WISE
        elif "compare" in query_lower:
            query_type = QueryType.COMPARISON
        elif "critical" in query_lower:
            query_type = QueryType.CRITICAL_DELAYS
        elif "summary" in query_lower:
            query_type = QueryType.SUMMARY
        
        # Extract tower names
        towers = []
        
        # Check EWS-LIG towers if project is all or ews-lig
        if project == "all" or project == "ews-lig":
            # Extract tower number from query
            tower_number = None
            
            # Try different patterns to find tower number
            patterns = [
                r'tower\s*[#]?\s*(\d+)',          # "tower 2"
                r'ews\s+tower\s*[#]?\s*(\d+)',    # "ews tower 2"
                r'lig\s+tower\s*[#]?\s*(\d+)',    # "lig tower 2"
                r'(\d+)(?:st|nd|rd|th)?\s+tower', # "2nd tower"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    tower_number = match.group(1)
                    break
            
            # Also look for standalone numbers after "tower"
            if not tower_number:
                words = query_lower.split()
                for i, word in enumerate(words):
                    if word == "tower" and i + 1 < len(words):
                        next_word = words[i + 1]
                        if next_word.isdigit():
                            tower_number = next_word
                            break
            
            # If we found a tower number, determine if it's EWS or LIG
            if tower_number:
                # Check if query specifically mentions LIG
                is_lig = "lig" in query_lower or "lig" in query
                
                if is_lig:
                    tower_name = f"LIG Tower {tower_number}"
                else:
                    tower_name = f"EWS Tower {tower_number}"
                
                # Add only once, no duplicates
                if tower_name not in towers:
                    towers.append(tower_name)
        
        # Check Eligo towers if project is all or eligo
        if project == "all" or project == "eligo":
            # Extract Eligo towers
            for letter in ['f', 'g', 'h']:
                if f'tower {letter}' in query_lower or f'tower-{letter}' in query_lower:
                    tower_name = f"Tower {letter.upper()}"
                    if tower_name not in towers:
                        towers.append(tower_name)

        # Check Veridia towers if project is all or veridia
        if project == "all" or project == "veridia":
            # Extract Veridia towers (Tower 1, Tower 2, etc.)
            # We look for "Tower N" pattern
            veridia_matches = re.finditer(r'tower\s*(\d+)', query_lower)
            for match in veridia_matches:
                tower_num = match.group(1)
                tower_name = f"Tower {tower_num}"
                # Only add if not already present (avoid duplicates with EWS logic if needed)
                if tower_name not in towers:
                    towers.append(tower_name)
            
            # Check for "Towers 5, 6, and 7" pattern
            multi_tower_match = re.search(r'towers\s+([\d\s,]+(?:and\s+\d+)?)', query_lower)
            if multi_tower_match:
                numbers_str = multi_tower_match.group(1)
                # Extract all numbers
                numbers = re.findall(r'\d+', numbers_str)
                for num in numbers:
                    tower_name = f"Tower {num}"
                    if tower_name not in towers:
                        towers.append(tower_name)
        
        # Extract filters
        filters = {}
        floor_match = re.search(r'floor[-\s]*(\d+)', query_lower)
        if floor_match:
            filters["floor"] = floor_match.group(1)
        
        pour_match = re.search(r'pour[-\s]*(\d+)', query_lower)
        if pour_match:
            filters["pour"] = pour_match.group(1)

        # Module filter (Veridia)
        module_match = re.search(r'\b[Mm](\d+)\b', query)
        if module_match:
            filters["module"] = module_match.group(1)
            # If explicit module mentioned, default to module-wise if generic
            if query_type == QueryType.DELAY_ANALYSIS:
                query_type = QueryType.MODULE_WISE
        
        module_word_match = re.search(r'module\s*(\d+)', query_lower)
        if module_word_match:
            filters["module"] = module_word_match.group(1)
            if query_type == QueryType.DELAY_ANALYSIS:
                query_type = QueryType.MODULE_WISE

        # Side filter (Veridia)
        side_match = re.search(r'\b(north|south|east|west)\b', query_lower)
        if side_match:
            filters["side"] = side_match.group(1)

        # Date filters
        # Extract dates (YYYY-MM-DD or DD-MM-YYYY)
        date_pattern = r'(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})'
        raw_dates = re.findall(date_pattern, query)
        
        # Normalize dates to YYYY-MM-DD
        dates = []
        for d in raw_dates:
            # If DD-MM-YYYY (e.g. 03-07-2024)
            if re.match(r'\d{2}-\d{2}-\d{4}', d):
                try:
                    # Convert DD-MM-YYYY to YYYY-MM-DD
                    dates.append(datetime.strptime(d, "%d-%m-%Y").strftime("%Y-%m-%d"))
                except:
                    dates.append(d)
            else:
                dates.append(d)
        
        if dates:
            # Check context for date type (baseline vs actual)
            is_actual = "actual" in query_lower or "completed" in query_lower
            
            if "from" in query_lower or "after" in query_lower or "start" in query_lower or "since" in query_lower:
                key = "actual_from" if is_actual else "baseline_from"
                filters[key] = dates[0]
                if len(dates) > 1:
                    key_to = "actual_to" if is_actual else "baseline_to"
                    filters[key_to] = dates[1]
            elif "to" in query_lower or "before" in query_lower or "end" in query_lower or "until" in query_lower:
                key = "actual_to" if is_actual else "baseline_to"
                filters[key] = dates[0]
            elif "between" in query_lower and len(dates) >= 2:
                key_from = "actual_from" if is_actual else "baseline_from"
                key_to = "actual_to" if is_actual else "baseline_to"
                filters[key_from] = dates[0]
                filters[key_to] = dates[1]
            elif len(dates) == 1:
                # Default to looking for activities around this date? 
                # Or just treat as 'from'
                key = "actual_from" if is_actual else "baseline_from"
                filters[key] = dates[0]

        # Status filters
        if "delayed" in query_lower or "delays" in query_lower:
            filters["status"] = "delayed"
        elif "delay" in query_lower and "analysis" not in query_lower:
            # "delay" word implies delayed status unless it's "delay analysis"
            filters["status"] = "delayed"
        elif "completed" in query_lower:
            filters["status"] = "completed"
        elif "pending" in query_lower:
            filters["status"] = "pending"
        elif "ongoing" in query_lower or "in progress" in query_lower:
            filters["status"] = "in progress"
        elif "on time" in query_lower or "on-time" in query_lower or "ontime" in query_lower:
            filters["status"] = "on time"
            
        logger.info(f"Fallback parser - Project: {project}, Towers: {towers}, Query: {query}, Filters: {filters}")
        
        return IntentResponse(
            intent=intent,
            confidence=0.7,
            project=project,
            towers=towers,
            query_type=query_type,
            filters=filters,
            parsed_query={"method": "fallback_parser"}
        )

    @staticmethod
    async def process_query(query_request: QueryRequest) -> Dict[str, Any]:
        start_time = datetime.now()
        query_lower = query_request.query.lower()
        
        # Analyze query with LLM or fallback
        if query_request.use_llm and llm_service and WATSONX_URL and WATSONX_API_KEY:
            try:
                intent_response = await llm_service.analyze_query(query_request.query)
            except Exception as e:
                logger.error(f"LLM processing failed: {e}")
                intent_response = llm_service._create_default_response() if llm_service else IntentResponse(
                    intent=UserIntent.GET_DELAYS,
                    confidence=0.5,
                    project="all",
                    query_type=QueryType.TOWER_WISE
                )
        else:
            # Use fallback parser
            intent_response = QueryProcessor.fallback_parse_query(query_request.query)
        
        # Override with user-specified values if provided
        if query_request.project:
            intent_response.project = query_request.project
            
        # 🔧 FIX: normalize project value (REQUIRED)
        if intent_response.project:
            p = (
                intent_response.project
                .lower()
                .replace(" ", "")
                .replace("_", "")
                .replace("-", "")
            )
        
            if p in ["ewslig", "ews", "lig", "ews-lig", "ews_lig"]:
                intent_response.project = "ews-lig"
            elif p == "eligo":
                intent_response.project = "eligo"
            elif p == "veridia":
                intent_response.project = "veridia"
            elif p == "all":
                intent_response.project = "all"
            else:
                intent_response.project = None

        has_ews_word = re.search(r"\bews\b", query_lower) is not None
        has_lig_word = re.search(r"\blig\b", query_lower) is not None

        if (has_ews_word or has_lig_word) and intent_response.project in [None, "all"]:
            intent_response.project = "ews-lig"
        
        
        
        if query_request.tower:
            intent_response.towers = [query_request.tower]

        # 🔧 FIX: Refine towers for EWS/LIG based on query text
        # This handles cases where LLM returns generic "Tower X" or empty list,
        # but user specifically asked for LIG or EWS.
        if intent_response.project == "ews-lig" or intent_response.project == "all":
            is_lig_explicit = has_lig_word
            is_ews_explicit = has_ews_word

            if is_ews_explicit and not is_lig_explicit:
                intent_response.filters["tower_scope"] = "EWS"
            elif is_lig_explicit and not is_ews_explicit:
                intent_response.filters["tower_scope"] = "LIG"
            
            # If we have towers, try to refine them
            if intent_response.towers:
                new_towers = []
                for tower in intent_response.towers:
                    tower_lower = tower.lower()
                    
                    # If tower is generic (no EWS/LIG prefix)
                    if "ews" not in tower_lower and "lig" not in tower_lower:
                        if is_lig_explicit and not is_ews_explicit:
                            # User said LIG, tower is generic -> make it LIG
                            # Check if it already has "Tower"
                            if "tower" in tower_lower:
                                new_towers.append(f"LIG {tower}")
                            else:
                                new_towers.append(f"LIG Tower {tower}")
                        elif is_ews_explicit and not is_lig_explicit:
                            # User said EWS, tower is generic -> make it EWS
                            if "tower" in tower_lower:
                                new_towers.append(f"EWS {tower}")
                            else:
                                new_towers.append(f"EWS Tower {tower}")
                        else:
                            # Ambiguous or both -> keep as is (will match both)
                            new_towers.append(tower)
                    else:
                        # Tower already has specific type
                        # Filter out mismatches if user was explicit
                        if is_lig_explicit and not is_ews_explicit:
                            if "ews" in tower_lower:
                                continue # Skip EWS tower if user wanted LIG
                        
                        if is_ews_explicit and not is_lig_explicit:
                            if "lig" in tower_lower:
                                continue # Skip LIG tower if user wanted EWS
                                
                        new_towers.append(tower)
                
                intent_response.towers = new_towers
            
            # If no towers found (maybe LLM failed to extract), try to extract from query manually
            # This is a mini-fallback for towers specifically
            if not intent_response.towers and (is_lig_explicit or is_ews_explicit):
                # Try to find tower number
                tower_num_match = re.search(r'tower\s*[#]?\s*(\d+)', query_lower)
                if not tower_num_match:
                     tower_num_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+tower', query_lower)
                
                if tower_num_match:
                    num = tower_num_match.group(1)
                    if is_lig_explicit:
                        intent_response.towers.append(f"LIG Tower {num}")
                    elif is_ews_explicit:
                        intent_response.towers.append(f"EWS Tower {num}")

        logger.info(f"Processing query: '{query_request.query}'")
        logger.info(f"Intent: {intent_response.intent.value}, Project: {intent_response.project}, Towers: {intent_response.towers}")
        
        # Process based on intent
        if intent_response.intent == UserIntent.LIST_PROJECTS:
            return await QueryProcessor._handle_list_projects()
        
        elif intent_response.intent == UserIntent.LIST_TOWERS:
            return await QueryProcessor._handle_list_towers(intent_response.project)
        
        else:
            return await QueryProcessor._handle_delay_analysis(intent_response, start_time)
    
    @staticmethod
    async def _handle_list_projects() -> Dict[str, Any]:
        """Handle list projects intent."""
        return {
            "status": "success",
            "intent": "list_projects",
            "projects": [
                {"name": "EWS LIG P4", "type": "ews-lig", "description": "EWS LIG Structure Work"},
                {"name": "Eligo", "type": "eligo", "description": "Eligo Structure Work"}
            ],
            "total": 2,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    async def _handle_list_towers(project: str) -> Dict[str, Any]:
        """Handle list towers intent."""
        towers = []
        
        if project == "ews-lig" or project == "all":
            # EWS LIG has variable number of towers, but typically 4-6
            for i in range(1, 7):
                towers.append(f"EWS Tower {i}")
                towers.append(f"LIG Tower {i}")
        
        if project == "eligo" or project == "all":
            towers.extend(["Tower F", "Tower G", "Tower H"])
        
        return {
            "status": "success",
            "intent": "list_towers",
            "project": project,
            "towers": towers,
            "total": len(towers),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    async def _handle_delay_analysis(intent_response: IntentResponse, start_time: datetime) -> Dict[str, Any]:
        """Handle delay analysis intent."""
        
        # Determine which projects to analyze
        projects_to_analyze = []
        if not intent_response.project or intent_response.project == "all":
            intent_response.project = "all"
            projects_to_analyze = ["ews-lig", "eligo", "veridia"]
        elif intent_response.project == "ews-lig":
            projects_to_analyze = ["ews-lig"]
        elif intent_response.project == "eligo":
            projects_to_analyze = ["eligo"]
        elif intent_response.project == "veridia":
            projects_to_analyze = ["veridia"]
        else:
            projects_to_analyze = ["ews-lig", "eligo", "veridia"]
        
        results = {}
        
        for project_type in projects_to_analyze:
            try:
                project_data = await QueryProcessor._get_project_data(project_type)
                if project_data and "data" in project_data:
                    data = project_data["data"]
                    
                    # Apply filters and query type
                    filtered_results = QueryProcessor._apply_query_filters(
                        data, intent_response.query_type, intent_response.filters, intent_response.towers
                    )
                    
                    results[project_type] = {
                        "project": project_data.get("project", project_type),
                        "file": project_data.get("file"),
                        "date": project_data.get("date"),
                        "query_type": intent_response.query_type.value,
                        "results": filtered_results,
                        "metrics": data.get("delay_metrics", {})
                    }
            except Exception as e:
                logger.error(f"Error analyzing project {project_type}: {e}")
                results[project_type] = {
                    "error": str(e),
                    "status": "error"
                }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "status": "success",
            "intent": intent_response.intent.value,
            "confidence": intent_response.confidence,
            "timestamp": datetime.now().isoformat(),
            "query_type": intent_response.query_type.value,
            "filters_applied": intent_response.filters,
            "processing_time_ms": round(processing_time, 2),
            "projects_analyzed": list(results.keys()),
            "results": results,
            "query_info": {
                "parsed_query": intent_response.parsed_query
            }
        }
        
        return convert_to_json_serializable(response)
    
    @staticmethod
    async def _get_project_data(project_type: str) -> Optional[Dict]:
        """Get project data by type."""
        if project_type == "ews-lig":
            file_info = get_latest_ews_lig_file()
            if not file_info:
                return None
            
            try:
                df = load_dataframe(file_info["key"], "Revised Baseline 45daysNGT+Rai")
                data = extract_ews_lig_schedule_data(df)
                
                return {
                    "project": "EWS LIG P4",
                    "file": file_info["file"],
                    "date": file_info["date"].strftime("%d-%m-%Y"),
                    "data": data,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error loading EWS LIG data: {e}")
                return None
        
        elif project_type == "eligo":
            file_info = get_latest_eligo_file()
            if not file_info:
                return None
            
            try:
                # Determine sheet name for Eligo
                obj = cos.get_object(Bucket=COS_BUCKET, Key=file_info["key"])
                data_bytes = obj["Body"].read()
                
                with pd.ExcelFile(BytesIO(data_bytes), engine="openpyxl") as xls:
                    sheet_names = xls.sheet_names
                
                target_sheet = None
                for sheet in sheet_names:
                    sheet_lower = sheet.lower()
                    if "25 days" in sheet_lower or "baselines" in sheet_lower:
                        target_sheet = sheet
                        break
                
                if not target_sheet and sheet_names:
                    target_sheet = sheet_names[0]
                
                if not target_sheet:
                    raise ValueError("No sheets found in Eligo file")
                
                df = load_dataframe(file_info["key"], target_sheet)
                extracted_data = extract_eligo_slab_data(df, target_sheet)
                
                return {
                    "project": "Eligo",
                    "file": file_info["file"],
                    "date": file_info["date"].strftime("%d-%m-%Y"),
                    "sheet": target_sheet,
                    "data": extracted_data,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error loading Eligo data: {e}")
                return None
        
        elif project_type == "veridia":
            file_info = get_latest_veridia_file()
            if not file_info:
                return None
            
            try:
                # Load Veridia file (assume first sheet)
                obj = cos.get_object(Bucket=COS_BUCKET, Key=file_info["key"])
                data_bytes = obj["Body"].read()
                
                with pd.ExcelFile(BytesIO(data_bytes), engine="openpyxl") as xls:
                    sheet_names = xls.sheet_names
                    
                target_sheet = None
                if sheet_names:
                    # Prioritize specific Veridia sheet
                    for sheet in sheet_names:
                        if "Revised baseline with 60d NGT" in sheet:
                            target_sheet = sheet
                            break
                    
                    # Fallback to first sheet
                    if not target_sheet:
                        target_sheet = sheet_names[0]
                
                df = load_dataframe(file_info["key"], target_sheet)
                data = extract_veridia_data(df, target_sheet)
                
                return {
                    "project": "Veridia",
                    "file": file_info["file"],
                    "date": file_info["date"].strftime("%d-%m-%Y"),
                    "data": data,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error loading Veridia data: {e}")
                return None

        return None
    
    @staticmethod
    def _apply_query_filters(data: Dict, query_type: QueryType, filters: Dict, towers: List[str]) -> Dict:
        """Apply query filters to project data with comprehensive matching and filtering."""
        result = {
            "towers": {},
            "activities": [],
            "filtered_activities": [],
            "summary": {
                "total_towers": 0,
                "filters_applied": filters.copy()
            }
        }
        
        if "towers" not in data:
            logger.warning("No towers found in data")
            return result
        
        # Filter towers if specified
        target_towers = {}
        available_tower_keys = list(data["towers"].keys())
        tower_scope = str(filters.get("tower_scope", "")).upper()
        
        logger.info(f"Filtering towers - Requested: {towers}, Available: {available_tower_keys}")
        logger.info(f"Filters to apply: {filters}")
        
        # Remove duplicates from requested towers
        unique_towers = []
        for tower in towers:
            if tower not in unique_towers:
                unique_towers.append(tower)
        towers = unique_towers
        
        # Check if this is Eligo project based on data
        is_eligo_project = False
        if "project" in data:
            is_eligo_project = "eligo" in str(data.get("project", "")).lower()
        else:
            # Check tower names for Eligo pattern
            has_eligo_towers = any(
                any(letter in key.lower() for letter in ['f', 'g', 'h'])
                for key in available_tower_keys
            )
            is_eligo_project = has_eligo_towers

        candidate_tower_items = list(data["towers"].items())
        if tower_scope in {"EWS", "LIG"}:
            scoped_items = []
            for tower_key, tower_data in candidate_tower_items:
                tower_display_name = str(tower_data.get("tower", tower_key))
                tower_text = f"{tower_key} {tower_display_name}".upper()
                if tower_scope in tower_text:
                    scoped_items.append((tower_key, tower_data))

            if scoped_items:
                candidate_tower_items = scoped_items
                logger.info(
                    f"Restricted candidate towers to scope '{tower_scope}': "
                    f"{[tower_data.get('tower', tower_key) for tower_key, tower_data in candidate_tower_items]}"
                )
        
        # Helper function for tower matching
        def tower_matches_request(tower_display_name: str, tower_key: str, requested_tower: str) -> bool:
            """Check if a tower matches the requested tower name."""
            requested_lower = requested_tower.lower().strip()
            display_lower = tower_display_name.lower().strip()
            key_lower = tower_key.lower().strip()
            
            # 1. Check for exact match first (most strict)
            exact_match = (
                requested_lower == display_lower or
                requested_lower == key_lower or
                requested_lower.replace(" ", "_") == key_lower
            )
            
            if exact_match:
                logger.info(f"Exact tower match: '{requested_tower}' → '{tower_display_name}'")
                return True
            
            # Check if this is EWS/LIG project based on tower names
            is_ews_tower = "ews" in display_lower or "ews" in key_lower
            is_lig_tower = "lig" in display_lower or "lig" in key_lower
            is_eligo_tower = any(letter in display_lower for letter in ['f', 'g', 'h']) or any(letter in key_lower for letter in ['f', 'g', 'h'])
            
            if is_eligo_tower:
                # Eligo specific matching (towers F, G, H)
                # Extract letter from request
                req_letter = ""
                if "tower" in requested_lower:
                    parts = requested_lower.split("tower")
                    if len(parts) > 1:
                        req_letter = parts[1].replace(" ", "").strip()
                
                # If no letter extracted, try to get single letter
                if not req_letter and len(requested_lower) == 1:
                    req_letter = requested_lower
                
                # Extract letter from tower names
                display_letter = ""
                if "tower" in display_lower:
                    parts = display_lower.split("tower")
                    if len(parts) > 1:
                        display_letter = parts[1].replace(" ", "").replace("_", "").strip()
                
                key_letter = ""
                if "tower" in key_lower:
                    parts = key_lower.split("tower")
                    if len(parts) > 1:
                        key_letter = parts[1].replace(" ", "").replace("_", "").strip()
                
                # If no letters found in names, try to extract from whole string
                if not display_letter:
                    for letter in ['f', 'g', 'h']:
                        if letter in display_lower:
                            display_letter = letter
                            break
                
                if not key_letter:
                    for letter in ['f', 'g', 'h']:
                        if letter in key_lower:
                            key_letter = letter
                            break
                
                # Check for letter match
                if req_letter and (req_letter == display_letter or req_letter == key_letter):
                    logger.info(f"Eligo letter match: '{req_letter}' → '{tower_display_name}'")
                    return True
                
                # Partial match for Eligo
                if req_letter and (req_letter in display_lower or req_letter in key_lower):
                    logger.info(f"Eligo partial match: '{req_letter}' in '{tower_display_name}'")
                    return True
            
            # Check for Veridia specific numeric towers (4, 5, 6, 7)
            elif requested_lower in ["4", "5", "6", "7"] and ("veridia" in display_lower or "tower" in display_lower):
                # Simple number match for Veridia
                req_num = requested_lower
                display_num = ''.join(filter(str.isdigit, display_lower))
                if req_num == display_num:
                    return True

            elif is_ews_tower or is_lig_tower:
                # EWS/LIG matching logic - FIXED VERSION
                # Extract tower number from requested tower
                req_number = ''.join(filter(str.isdigit, requested_lower))
                
                # Extract tower number from display name and key
                display_number = ''.join(filter(str.isdigit, display_lower))
                key_number = ''.join(filter(str.isdigit, key_lower))
                
                # Extract tower type (EWS or LIG) from requested
                is_ews_requested = "ews" in requested_lower
                is_lig_requested = "lig" in requested_lower
                
                # Extract tower type from display/key
                is_ews_display = "ews" in display_lower
                is_lig_display = "lig" in display_lower
                is_ews_key = "ews" in key_lower
                is_lig_key = "lig" in key_lower
                
                # For EWS/LIG, we must check BOTH type AND number
                
                # First, check if the types match
                type_matches = False
                if is_ews_requested and (is_ews_display or is_ews_key):
                    type_matches = True
                elif is_lig_requested and (is_lig_display or is_lig_key):
                    type_matches = True
                elif not is_ews_requested and not is_lig_requested:
                    # If no type specified in request, check if we have a number match
                    # and the tower is one of the expected types
                    if (is_ews_display or is_lig_display or is_ews_key or is_lig_key):
                        type_matches = True
                
                # Then check if numbers match
                number_matches = False
                if req_number:  # If a number was specified in request
                    if req_number == display_number or req_number == key_number:
                        number_matches = True
                else:  # No number specified - match all towers of the type
                    number_matches = True
                
                # Both type AND number must match for EWS/LIG towers
                if type_matches and number_matches:
                    logger.info(f"EWS/LIG match: '{requested_tower}' → '{tower_display_name}' (type_matches={type_matches}, number_matches={number_matches})")
                    return True
            
            return False
        
        # Apply tower filtering
        for tower_key, tower_data in candidate_tower_items:
            tower_display_name = tower_data.get("tower", tower_key)
            
            # If no towers specified, include all
            if not towers:
                target_towers[tower_key] = tower_data
                continue
            
            # Check if this tower matches any requested tower
            matches_any = False
            for requested_tower in towers:
                if tower_matches_request(tower_display_name, tower_key, requested_tower):
                    matches_any = True
                    break
            
            if matches_any:
                target_towers[tower_key] = tower_data
        
        logger.info(f"Filtered to {len(target_towers)} towers: {list(target_towers.keys())}")
        
        # Helper function to apply activity filters
        def apply_activity_filters(activity: Dict) -> bool:
            """Apply all activity-level filters and return True if activity passes all filters."""

            # 1. Status filter
            status_filter = filters.get("status")
            if status_filter:
                status_filter_lower = status_filter.lower()
                activity_status = str(activity.get("status", "")).lower()
                
                # Check delay days directly for robustness
                delay_days = activity.get("delay_days", 0)
                is_delayed = delay_days > 0
                
                if status_filter_lower in ["on time", "on-time", "ontime"]:
                    # Strict check for on time
                    if is_delayed or "delay" in activity_status:
                        return False
                elif status_filter_lower in ["delayed", "delay"]:
                    # Strict check for delayed - must have positive delay days
                    if not is_delayed:
                        return False
                elif status_filter_lower == "completed":
                    # Check if activity is completed (has actual date)
                    actual_date_val = activity.get("actual_date")
                    if not actual_date_val or str(actual_date_val).lower() in ["nan", "nat", "none", ""]:
                        return False
                elif status_filter_lower == "pending":
                    # Check if activity is pending (no actual date and not delayed yet)
                    actual_date_val = activity.get("actual_date")
                    has_actual = actual_date_val and str(actual_date_val).lower() not in ["nan", "nat", "none", ""]
                    
                    if has_actual: # If completed, it's not pending
                        return False
                    
                    # Also check explicit status if available
                    if "pending" in activity_status:
                        return True
                        
                    # Otherwise, if it's delayed, is it pending? 
                    # Usually pending implies not yet done, so delay status overrides unless specifically asking for pending items
                    # Let's assume pending means not completed
                    return True
                elif status_filter_lower == "in progress":
                    # Check if activity is in progress
                    if "progress" in activity_status or "ongoing" in activity_status:
                        return True
                    # If not explicitly marked, maybe infer from lack of completion? 
                    # Safer to stick to explicit status
                    return False
                else:
                    # Generic match
                    if status_filter_lower not in activity_status:
                        return False
            
            # 2. Delay days range filter
            delay_days = activity.get("delay_days", 0)
            
            min_delay = filters.get("min_delay_days")
            if min_delay is not None:
                try:
                    min_delay_int = int(min_delay)
                    if delay_days < min_delay_int:
                        return False
                except:
                    pass
            
            max_delay = filters.get("max_delay_days")
            if max_delay is not None:
                try:
                    max_delay_int = int(max_delay)
                    if delay_days > max_delay_int:
                        return False
                except:
                    pass
            
            # 3. Floor filter (for activities)
            floor_filter = filters.get("floor")
            if floor_filter:
                activity_floor = str(activity.get("floor", "")).lower()
                floor_num = ''.join(filter(str.isdigit, activity_floor))
                if not (floor_filter == floor_num or 
                        floor_filter in activity_floor or
                        f"floor {floor_filter}" in activity_floor or
                        f"{floor_filter}f" in activity_floor):
                    return False
            
            # 4. Pour filter (for activities)
            pour_filter = filters.get("pour")
            if pour_filter:
                activity_pour = str(activity.get("pour", "")).lower()
                pour_num = ''.join(filter(str.isdigit, activity_pour))
                if not (pour_filter == pour_num or 
                        pour_filter in activity_pour or
                        f"pour {pour_filter}" in activity_pour):
                    return False
            
            # 5. Date range filters
            baseline_date = activity.get("baseline_date")
            actual_date = activity.get("actual_date")
            
            def parse_filter_date(d_str):
                if not d_str: return None
                try:
                    return datetime.strptime(d_str, "%Y-%m-%d")
                except ValueError:
                    try:
                        return datetime.strptime(d_str, "%d-%m-%Y")
                    except:
                        return None

            # Filter by baseline date range
            baseline_from = filters.get("baseline_from")
            baseline_to = filters.get("baseline_to")
            if baseline_date and (baseline_from or baseline_to):
                try:
                    baseline_dt = datetime.strptime(baseline_date, "%Y-%m-%d")
                    
                    if baseline_from:
                        from_dt = parse_filter_date(baseline_from)
                        if from_dt and baseline_dt < from_dt:
                            return False
                    
                    if baseline_to:
                        to_dt = parse_filter_date(baseline_to)
                        if to_dt and baseline_dt > to_dt:
                            return False
                except:
                    pass
            
            # Filter by actual date range
            actual_from = filters.get("actual_from")
            actual_to = filters.get("actual_to")
            if actual_date and (actual_from or actual_to):
                try:
                    actual_dt = datetime.strptime(actual_date, "%Y-%m-%d")
                    
                    if actual_from:
                        from_dt = parse_filter_date(actual_from)
                        if from_dt and actual_dt < from_dt:
                            return False
                    
                    if actual_to:
                        to_dt = parse_filter_date(actual_to)
                        if to_dt and actual_dt > to_dt:
                            return False
                except:
                    pass
            
            # 6. Days duration filter
            days_filter = filters.get("days")
            if days_filter:
                activity_days = activity.get("days")
                if activity_days is not None:
                    try:
                        days_int = int(days_filter)
                        if activity_days != days_int:
                            return False
                    except:
                        pass
            
            # 7. Module filter (for activities)
            module_filter = filters.get("module")
            if module_filter:
                # Handle M-prefix in filter if present
                clean_module_filter = module_filter.upper().replace("M", "")
                target_module = f"M{clean_module_filter}"
                
                activity_module = str(activity.get("module", "")).upper()
                
                # Check direct module field
                if activity_module and activity_module == target_module:
                    pass
                else:
                    # Fallback to checking activity name
                    activity_name_upper = str(activity.get("activity", "")).upper()
                    if target_module not in activity_name_upper:
                        return False
            
            # 8. Side filter (Veridia)
            side_filter = filters.get("side")
            if side_filter:
                side_filter_lower = side_filter.lower()
                activity_side = str(activity.get("side", "")).lower()
                
                # Direct check
                if side_filter_lower == activity_side:
                    pass
                # Check activity name
                elif side_filter_lower in str(activity.get("activity", "")).lower():
                    pass
                else:
                    return False
            
            return True
        
        # Apply query type specific logic
        if query_type == QueryType.TOWER_WISE:
            result["towers"] = target_towers
            
            # Collect and filter activities for tower-wise view
            all_activities = []
            filtered_activities = []
            
            for tower_key, tower_data in target_towers.items():
                if "activities" in tower_data:
                    for activity in tower_data["activities"]:
                        all_activities.append(activity)
                        if apply_activity_filters(activity):
                            filtered_activities.append(activity)
            
            result["activities"] = filtered_activities
            result["filtered_activities"] = filtered_activities
            if filtered_activities:
                result["summary"]["delay_metrics"] = calculate_delay_metrics(filtered_activities)
            
            # Add pour information for each tower
            for tower_key, tower_data in result["towers"].items():
                if "pours" in tower_data:
                    result["towers"][tower_key]["pour_summary"] = {
                        "total_pours": len(tower_data["pours"]),
                        "pour_names": list(tower_data["pours"].keys()),
                        "floors_per_pour": {
                            pour_name: len(pour_data) 
                            for pour_name, pour_data in tower_data["pours"].items()
                        }
                    }
        
        elif query_type == QueryType.FLOOR_WISE:
            # First populate standard tower/activity lists
            all_activities = []
            filtered_activities = []
            
            for tower_key, tower_data in target_towers.items():
                if "activities" in tower_data:
                    for activity in tower_data["activities"]:
                        all_activities.append(activity)
                        if apply_activity_filters(activity):
                            filtered_activities.append(activity)
            
            result["towers"] = target_towers
            result["activities"] = filtered_activities
            result["filtered_activities"] = filtered_activities
            if filtered_activities:
                result["summary"]["delay_metrics"] = calculate_delay_metrics(filtered_activities)

            # Create Breakdown by Floor
            floor_breakdown = {}
            for activity in filtered_activities:
                floor_name = activity.get("floor", "Unknown")
                if floor_name not in floor_breakdown:
                    floor_breakdown[floor_name] = []
                floor_breakdown[floor_name].append(activity)
            
            # Sort keys if possible (simple heuristic)
            try:
                sorted_keys = sorted(floor_breakdown.keys(), key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
                floor_breakdown = {k: floor_breakdown[k] for k in sorted_keys}
            except:
                pass # Keep original order if sorting fails

            result["breakdown"] = {
                "type": "floor_wise",
                "groups": floor_breakdown
            }

        elif query_type == QueryType.POUR_WISE:
            # First populate standard tower/activity lists
            all_activities = []
            filtered_activities = []
            
            for tower_key, tower_data in target_towers.items():
                if "activities" in tower_data:
                    for activity in tower_data["activities"]:
                        all_activities.append(activity)
                        if apply_activity_filters(activity):
                            filtered_activities.append(activity)
            
            result["towers"] = target_towers
            result["activities"] = filtered_activities
            result["filtered_activities"] = filtered_activities
            if filtered_activities:
                result["summary"]["delay_metrics"] = calculate_delay_metrics(filtered_activities)

            # Create Breakdown by Pour
            pour_breakdown = {}
            for activity in filtered_activities:
                pour_name = activity.get("pour", "Unknown")
                if pour_name not in pour_breakdown:
                    pour_breakdown[pour_name] = []
                pour_breakdown[pour_name].append(activity)
            
            # Sort keys
            try:
                sorted_keys = sorted(pour_breakdown.keys(), key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
                pour_breakdown = {k: pour_breakdown[k] for k in sorted_keys}
            except:
                pass

            result["breakdown"] = {
                "type": "pour_wise",
                "groups": pour_breakdown
            }

        elif query_type == QueryType.MODULE_WISE:
            # First populate standard tower/activity lists
            all_activities = []
            filtered_activities = []
            
            for tower_key, tower_data in target_towers.items():
                if "activities" in tower_data:
                    for activity in tower_data["activities"]:
                        all_activities.append(activity)
                        if apply_activity_filters(activity):
                            filtered_activities.append(activity)
            
            result["towers"] = target_towers
            result["activities"] = filtered_activities
            result["filtered_activities"] = filtered_activities
            if filtered_activities:
                result["summary"]["delay_metrics"] = calculate_delay_metrics(filtered_activities)

            # Create Breakdown by Module
            module_breakdown = {}
            for activity in filtered_activities:
                module_name = activity.get("module", "Unknown")
                side_name = activity.get("side", "")
                
                # Create a composite key for grouping
                if module_name != "Unknown":
                    key = f"{module_name}"
                    if side_name:
                        key += f" {side_name}"
                else:
                    # Fallback to pour if module not available (e.g. non-Veridia)
                    key = activity.get("pour", "Unknown")
                
                if key not in module_breakdown:
                    module_breakdown[key] = []
                module_breakdown[key].append(activity)
            
            # Sort keys
            try:
                # specific sort for M1, M2 etc
                def sort_key(k):
                    nums = ''.join(filter(str.isdigit, str(k)))
                    return int(nums) if nums else 999
                
                sorted_keys = sorted(module_breakdown.keys(), key=sort_key)
                module_breakdown = {k: module_breakdown[k] for k in sorted_keys}
            except:
                pass

            result["breakdown"] = {
                "type": "module_wise",
                "groups": module_breakdown
            }
        
        elif query_type == QueryType.DELAY_ANALYSIS or query_type == QueryType.CRITICAL_DELAYS:
            # Collect activities based on filters.
            all_activities = []
            filtered_activities = []

            for tower_key, tower_data in target_towers.items():
                if "activities" in tower_data:
                    for activity in tower_data["activities"]:
                        all_activities.append(activity)
                        if apply_activity_filters(activity):
                            filtered_activities.append(activity)

            # Delay-focused queries should only return delayed activities.
            filtered_activities = [
                activity for activity in filtered_activities
                if activity.get("delay_days", 0) > 0
            ]
            
            result["towers"] = target_towers
            result["activities"] = filtered_activities
            result["filtered_activities"] = filtered_activities
            if filtered_activities:
                result["summary"]["delay_metrics"] = calculate_delay_metrics(filtered_activities)

            result["breakdown"] = {
                "type": "status_wise",
                "groups": {"Delayed": filtered_activities}
            }
        
        elif query_type == QueryType.COMPARISON:
            # For comparison, include all specified towers
            result["towers"] = target_towers
            
            # Calculate comparison metrics
            comparison_metrics = {}
            for tower_key, tower_data in target_towers.items():
                if "delay_metrics" in tower_data:
                    comparison_metrics[tower_key] = tower_data["delay_metrics"]
            
            result["summary"]["comparison"] = {
                "towers_compared": len(target_towers),
                "metrics": comparison_metrics
            }
            
            # Include activities if requested
            all_activities = []
            filtered_activities = []
            
            for tower_key, tower_data in target_towers.items():
                if "activities" in tower_data:
                    for activity in tower_data["activities"]:
                        all_activities.append(activity)
                        if apply_activity_filters(activity):
                            filtered_activities.append(activity)
            
            result["activities"] = filtered_activities
            result["filtered_activities"] = filtered_activities
        elif query_type == QueryType.SUMMARY:
            # Collect all activities for summary
            all_activities = []
            filtered_activities = []
            
            for tower_key, tower_data in target_towers.items():
                if "activities" in tower_data:
                    for activity in tower_data["activities"]:
                        all_activities.append(activity)
                        if apply_activity_filters(activity):
                            filtered_activities.append(activity)
            
            result["summary"] = {
                "total_towers": len(target_towers),
                "delay_metrics": calculate_delay_metrics(filtered_activities) if filtered_activities else {},
                "towers_included": list(target_towers.keys()),
                "filters_applied": filters
            }
            
            if filters.get("include_tower_list"):
                result["towers"] = target_towers

        
        # Add tower data to activities for better context (Skip for Veridia as requested)
        is_veridia = str(data.get("project", "")).lower() == "veridia"
        
        if not is_veridia:
            for activity in result.get("activities", []):
                if "tower_key" not in activity:
                    # Find which tower this activity belongs to
                    for tower_key, tower_data in target_towers.items():
                        if "activities" in tower_data:
                            # Create a simple ID for comparison
                            activity_id = f"{activity.get('tower', '')}_{activity.get('pour', '')}_{activity.get('floor', '')}"
                            for tower_activity in tower_data["activities"]:
                                tower_activity_id = f"{tower_activity.get('tower', '')}_{tower_activity.get('pour', '')}_{tower_activity.get('floor', '')}"
                                if activity_id == tower_activity_id:
                                    activity["tower_key"] = tower_key
                                    activity["tower_display_name"] = tower_data.get("tower", tower_key)
                                    break
                            if "tower_key" in activity:
                                break
        
        # Cleanup Veridia output as requested (remove activity name)
        if is_veridia:
            # Helper to clean activity dict
            def clean_activity(act):
                if isinstance(act, dict):
                    act.pop("activity", None) # Remove constructed name
                    # Also ensure tower_key/display are gone if they crept in
                    act.pop("tower_key", None)
                    act.pop("tower_display_name", None)
            
            # Clean lists
            for act in result.get("activities", []):
                clean_activity(act)
            for act in result.get("filtered_activities", []):
                clean_activity(act)
                
            # Clean towers structure
            for t_key, t_data in result.get("towers", {}).items():
                for act in t_data.get("activities", []):
                    clean_activity(act)
                # Clean floors
                for floor_data in t_data.get("floors", {}).values():
                    for module_data in floor_data.values():
                        clean_activity(module_data)
            
            # Clean breakdown if exists
            if "breakdown" in result and "groups" in result["breakdown"]:
                for group in result["breakdown"]["groups"].values():
                    for act in group:
                        clean_activity(act)
        
        # Update summary counts
        result["summary"]["total_towers"] = len(result.get("towers", {}))
        
        # Log final result summary
        logger.info(f"Query result - Towers: {result['summary']['total_towers']}, "
                    f"Query type: {query_type.value}")
        
        return strip_severity_from_response(result)
    
# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
def read_root():
    return {
        "message": "Unified Structure Work Tracker API with LLM Query",
        "version": "5.0",
        "endpoints": {
            "/": "This info page",
            "/health": "Health check",
            "/analyze": "Analyze with natural language query (POST)",
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        cos.list_objects_v2(Bucket=COS_BUCKET, MaxKeys=1)
        llm_status = "initialized" if llm_service and llm_service.initialized else "not_initialized"
        return {
            "status": "healthy",
            "cos_connected": True,
            "llm_service": llm_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "cos_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/analyze")
async def analyze_structure_query(query_request: QueryRequest):
    try:
        result = await QueryProcessor.process_query(query_request)
        return shape_simple_response(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
