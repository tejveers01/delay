import os
import re
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Dict, Optional, Union, Any, Tuple
from functools import lru_cache
import logging

import numpy as np
import pandas as pd
import ibm_boto3
from ibm_botocore.client import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --------------------------------------------------
# Setup logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load ENV
# --------------------------------------------------
load_dotenv()

COS_API_KEY = os.getenv("COS_API_KEY")
COS_CRN = os.getenv("COS_SERVICE_INSTANCE_CRN")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
COS_BUCKET = os.getenv("COS_BUCKET_NAME")

# --------------------------------------------------
# Pydantic Models
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    project: Optional[str] = None

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Construction Delay Analysis API",
    version="1.0"
)

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
# CONSTANTS
# --------------------------------------------------

STRUCTURE_PATTERN = re.compile(
    r"Structure Work Tracker.*\((\d{2}-\d{2}-\d{4})\)\.xlsx",
    re.IGNORECASE
)

COLUMN_MAPPINGS = {
    "eden": [
        "Tower ",
        "Task Name"
        "Activity No.",
        "Structure Task",
        "Baseline Finish",
        "Finish",
        "% Complete"
    ],
    "wave city club": [
        "Block",
        "Part",
        "Domain",
        "Activity ID",
        "Activity Name",
        "Baseline Finish",
        "Finish ()",
        "% Complete"
    ],
}

# Sheet mapping for Wave City Club
WAVE_CITY_SHEETS = {
    "B1": "B1 Banket Hall & Finedine",
    "B1 Banket Hall & Finedine": "B1 Banket Hall & Finedine",
    "B2 & B3": "B2 & B3",
    "B4": "B4",
    "B5": "B5",
    "B6": "B6",
    "B7": "B7",
    "B8": "B8",
    "B9": "B9",
    "B10": "B10",
    "B11": "B11",
}

PROJECT_SHEETS = {
    "eden": ["Master Sheet"],
    "wave city club": list(WAVE_CITY_SHEETS.values()),
}

# Header row configuration per project
PROJECT_HEADER_ROW = {
    "eden": 0,
    "wave city club": 1,
}

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_int(value: Any) -> int:
    """Safely convert any value to integer."""
    if value is None:
        return 0
    try:
        if isinstance(value, str):
            # Remove any non-numeric characters except minus sign
            cleaned = re.sub(r'[^\d\-.]', '', value)
            if cleaned:
                return int(float(cleaned))
            return 0
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            return 0
    except (ValueError, TypeError):
        return 0

def safe_float(value: Any) -> float:
    """Safely convert any value to float."""
    if value is None:
        return 0.0
    try:
        if isinstance(value, str):
            cleaned = re.sub(r'[^\d\-.]', '', value)
            if cleaned:
                return float(cleaned)
            return 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0

def format_completed_value(value: Any) -> Optional[str]:
    """Format completion values consistently as percentages."""
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


def detect_requested_context_fields(
    query_lower: str,
    filters_applied: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Infer which context columns should be included in compact activity rows."""
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


def shape_compact_activity_rows(
    rows: List[Dict[str, Any]],
    root_project: Optional[str],
    include_context_fields: List[str]
) -> List[Dict[str, Any]]:
    """Trim activity rows to the compact API response format."""
    shaped_rows = []

    for row in rows or []:
        tower_value = row.get("Tower") or row.get("Block") or row.get("tower") or row.get("block")
        activity_name = (
            row.get("Activity Name")
            or row.get("Structure Task")
            or row.get("Task Name")
            or row.get("activity_name")
        )
        baseline_finish = row.get("Baseline Finish") or row.get("baseline_finish")
        actual_finish = (
            row.get("Finish")
            or row.get("Finish ()")
            or row.get("Actual Finish")
            or row.get("actual_finish")
            or row.get("finish_date")
        )
        delay_days = row.get("Delay_Days") or row.get("delay_days")
        completed = (
            row.get("% Complete")
            or row.get("percent_complete")
            or row.get("completed")
        )

        shaped = {
            "tower": tower_value,
            "activity_name": activity_name,
            "baseline_finish": baseline_finish,
            "actual_finish": actual_finish,
            "finish_date": actual_finish,
            "delay_days": delay_days,
            "completed": format_completed_value(completed),
        }

        field_map = {
            "pour": ["Pour", "pour"],
            "module": ["Module", "module"],
            "floor": ["Floor", "floor"],
            "flat": ["Flat", "flat"],
        }

        for field_name in include_context_fields:
            value = None
            for candidate in field_map.get(field_name, []):
                if candidate in row:
                    value = row.get(candidate)
                    break
            shaped[field_name] = value

        if root_project:
            shaped["project"] = root_project

        shaped_rows.append(shaped)

    return shaped_rows


def shape_special_project_response(
    result: Dict[str, Any],
    original_query: str,
    filters_applied: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Apply compact response shaping for delay list queries."""
    if not isinstance(result, dict):
        return result

    if "delayed_activities" not in result:
        return result

    query_lower = (original_query or "").lower()
    include_context_fields = detect_requested_context_fields(query_lower, filters_applied)
    project_name = result.get("project")

    result["activity_details"] = shape_compact_activity_rows(
        result.get("delayed_activities", []),
        project_name,
        include_context_fields
    )
    result.pop("delayed_activities", None)
    result.pop("summary", None)

    return result

# --------------------------------------------------
# QUERY PARSING FUNCTIONS
# --------------------------------------------------

# Update the parse_query function to better handle activity name detection:
# Update the parse_query function with better activity name extraction:
def parse_query(query: str) -> Tuple[Optional[str], Optional[str], Optional[str], Dict, bool, bool, bool]:
    """
    Parse natural language query to extract project, tower/block, sheet, and filters.
    Returns: (project, tower/block, sheet, filters, all_projects_flag, tower_wise_flag, consolidate_flag)
    """
    if query is None:
        return None, None, None, {}, False, False, False

    query_lower = query.lower()
    original_query = query  # Keep original for better text extraction
    filters = {}
    all_projects_flag = False
    tower_wise_flag = False
    consolidate_flag = False
    
    # Initialize variables
    project = None
    tower_block = None
    sheet = None
    
    # Check for "consolidate", "summary", "report", "group by"
    if any(x in query_lower for x in ["consolidate", "summary", "report", "overview", "group by"]):
        consolidate_flag = True

    # Check for "tower wise" or "tower-wise" analysis
    if "tower wise" in query_lower or "tower-wise" in query_lower or "by tower" in query_lower:
        tower_wise_flag = True
    
    # Check for "all towers" or "each tower" or "every tower"
    if "all towers" in query_lower or "all tower" in query_lower or "each tower" in query_lower or "every tower" in query_lower:
        tower_wise_flag = True
    
    # Check for "all projects" query
    if "all projects" in query_lower or "all project" in query_lower or "every project" in query_lower:
        all_projects_flag = True
        # Don't return yet, allow filter extraction to happen
    
    # Check for "both projects" or "each project"
    if "both projects" in query_lower or "each project" in query_lower:
        all_projects_flag = True
        # Don't return yet, allow filter extraction to happen
    
    # Detect project
    if "eden" in query_lower:
        project = "Eden"
    elif "wave" in query_lower or "club" in query_lower or "wcc" in query_lower:
        project = "Wave City Club"
    
    # Detect tower/block/sheet patterns
    # For Eden: tower patterns
    tower_patterns = [
        r'tower\s*(\d+|[a-zA-Z])\b', # "tower 4", "tower A"
        r'\bt\s*(\d+|[a-zA-Z])\b',   # "t4", "tA"
        r'\btower\s*([a-z]\d+)',     # "tower b1"
        r'\b(\d+)\s*tower',          # "4 tower"
    ]
    
    # For Wave City Club: block/sheet patterns
    block_patterns = [
        r'b\s*(\d+)',               # "b1", "b 1"
        r'block\s*(\d+)',           # "block 1"
        r'\bb(\d+)\b',              # "b1" standalone
        r'\bbanket',                # "banket hall"
    ]
    
    # Try tower patterns first (for Eden)
    for pattern in tower_patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            tower_block = match.group(1).upper()
            break
    
    # Try block patterns (for Wave City Club)
    if not tower_block:
        for pattern in block_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                if pattern == r'\bbanket':
                    tower_block = "B1"
                else:
                    tower_block = f"B{match.group(1)}"
                
                # Infer project if not set
                if not project:
                    project = "Wave City Club"
                break
    
    # Map block to sheet for Wave City Club
    if project == "Wave City Club" and tower_block:
        sheet = WAVE_CITY_SHEETS.get(tower_block)
    
    # Extract delay threshold
    delay_match = re.search(r'(\d+)\s+days', query_lower)
    if delay_match:
        filters["min_delay_days"] = int(delay_match.group(1))
    
    # Extract activity ID
    activity_id_match = re.search(r'activity\s*(?:number|no\.?|id)?\s*(\d+)', query_lower, re.IGNORECASE)
    if activity_id_match:
        filters["activity_id"] = activity_id_match.group(1)

    # Extract Part (for Wave City Club)
    # Look for "part X" or "X part"
    part_match = re.search(r'\bpart\s+(?!of\b|in\b|for\b)(\w+)\b', query_lower, re.IGNORECASE)
    if not part_match:
        # Try "X part" (e.g., "podium part")
        part_match = re.search(r'\b(\w+)\s+part\b', query_lower, re.IGNORECASE)
    
    if part_match:
        filters["part"] = part_match.group(1).upper()
        
    # Extract Domain (for Wave City Club)
    # Look for "X domain" or "domain X"
    # Prevent capturing "Summary of domain", "Part 1 domain" (where 1 is domain)
    domain_match = re.search(r'\b(?!in\b|of\b|for\b|from\b|summary\b|report\b)(\w+)\s+domain', query_lower, re.IGNORECASE)
    
    # If match found, check if it's a number (likely Part number or Block number leaking in)
    if domain_match and not domain_match.group(1).isdigit():
        filters["domain"] = domain_match.group(1).capitalize()
    else:
        # Avoid capturing "domain in", "domain of", etc.
        domain_match = re.search(r'\bdomain\s+(?!in\b|of\b|for\b|from\b)(\w+)', query_lower, re.IGNORECASE)
        if domain_match:
            filters["domain"] = domain_match.group(1).capitalize()
        
    # Extract % Complete (for Wave City Club)
    percent_match = re.search(r'(\d+)%\s*(?:complete|progress)', query_lower, re.IGNORECASE)
    if percent_match:
        filters["percent_complete"] = percent_match.group(1) + "%"
    
    # Handle "less than X% complete/progress"
    less_percent_match = re.search(r'less\s+than\s+(\d+)%\s*(?:complete|progress)', query_lower, re.IGNORECASE)
    if less_percent_match:
        filters["percent_complete"] = f"<{less_percent_match.group(1)}%"

    # Handle "completed" or "finished" activities (implies 100%)
    if "completed activities" in query_lower or "finished activities" in query_lower:
        filters["percent_complete"] = "100%"

    # Handle "unfinished" or "pending" (implies < 100% or finish=null)
    if "unfinished" in query_lower or "pending" in query_lower:
        filters["percent_complete"] = "<100%"
    
    # IMPROVED ACTIVITY NAME EXTRACTION
    # Look for common activity name patterns in construction
    common_activity_patterns = [
        # Wave City Club activities from the data
        r'excavation',
        r'pcc',
        r'foundation\s+reinforcement',
        r'foundation\s+shuttering',
        r'foundation\s+concreting',
        r'plinth\s+beam\s+reinforcement',
        r'plinth\s+beam\s+shuttering',
        r'plinth\s+beam\s+concreting',
        r'gf\s+colou?mn\s+reinforcement',
        r'gf\s+column\s+shuttering',
        r'gf\s+column\s+casting',
        r'gf\s+roof\s+slab\s+shuttering',
        r'gf\s+roof\s+slab\s+reinforcement',
        r'gf\s+roof\s+slab\s+casting',
        r'ff\s+colou?mn\s+reinforcement',
        r'ff\s+column\s+shuttering',
        r'ff\s+column\s+casting',
        r'ff\s+roof\s+slab\s+shuttering',
        r'ff\s+roof\s+slab\s+reinforcement',
        r'ff\s+roof\s+slab\s+casting',
        r'brick\s+work\s+gf',
        r'brick\s+work\s+ff',
        r'terrace\s+work',
        r'internal\s+plaster\s+gf',
        r'internal\s+plaster\s+ff',
        r'external\s+plaster',
        r'terrace\s+waterproofing',
        # Eden activities
        r'shuttering\s+work',
        r'reinforcement\s+work',
        r'column/shear\s+wall',
        r'layout\s+and\s+starter',
        r'checking\s+&\s+casting\s+work',
        r'beam/slab',
        r'lower\s+basement',
        r'upper\s+basement',
        r'shuttering',
        r'casting\s+work',
        r'checking',
        r'reinforcement\s+binding',
        r'handover',
        r'raft\s+casting',
        r'foundation',
        r'sub\s+structure\s+works',
        r'casting',
        r'plaster',
    ]
    
    # First try to match specific activity patterns
    activity_name_found = None
    for pattern in common_activity_patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            activity_name_found = match.group(0)
            break
    
    # If no pattern matched, try to extract activity name after keywords
    if not activity_name_found:
        # Look for activity after "activity" keyword
        activity_keyword_match = re.search(r'activity\s+(.+?)\s+(?:in|of|for|from)', query_lower, re.IGNORECASE)
        if not activity_keyword_match:
            activity_keyword_match = re.search(r'activity\s+(.+?)$', query_lower, re.IGNORECASE)
        
        if activity_keyword_match:
            activity_name_found = activity_keyword_match.group(1).strip()
            # Remove any project/tower references from the end
            if activity_name_found:
                activity_name_found = re.sub(r'\s+(eden|wave|club|b\d+|tower\s*\d+)$', '', activity_name_found, flags=re.IGNORECASE).strip()
    
    # If we found an activity name, add it to filters
    if activity_name_found:
        # Clean up the activity name
        if activity_name_found:
            activity_name_found = activity_name_found.strip()
        # Remove common prefixes
        activity_name_found = re.sub(r'^(show|get|find|search for|look for)\s+', '', activity_name_found, flags=re.IGNORECASE)
        # Remove common suffixes like "delays"
        activity_name_found = re.sub(r'\s+delays?$', '', activity_name_found, flags=re.IGNORECASE)
        
        if activity_name_found and not activity_name_found.isdigit(): # Don't treat "123" as activity name if we already have activity ID logic
            # Avoid setting activity name if it looks like "number X" and we have an activity_id
            if "activity_id" in filters and re.match(r'^(number|no\.?|id)\s*\d+$', activity_name_found, re.IGNORECASE):
                pass
            else:
                filters["activity_name"] = activity_name_found
    
    # Extract Status (Delayed vs On-Time)
    if "on time" in query_lower or "not delayed" in query_lower or "ontime" in query_lower:
        filters["status"] = "on_time"
    elif any(x in query_lower for x in ["delayed", "late", "delay", "exceeded baseline", "missed baseline", "finished late"]):
        filters["status"] = "delayed"
    elif re.search(r'missed\s+.*baseline', query_lower):
        filters["status"] = "delayed"
    elif "all activities" in query_lower or "all status" in query_lower:
        filters["status"] = "all"
    
    # Check for maximum delays
    if "maximum delay" in query_lower or "most delayed" in query_lower:
        filters["status"] = "delayed" # Implies we are looking for delays
        filters["sort"] = "max_delay" # Hint for sorting (though default is by delay)

    # Extract Limit (Top X)
    limit_match = re.search(r'top\s+(\d+)', query_lower)
    if limit_match:
        filters["limit"] = int(limit_match.group(1))
    else:
        limit_match = re.search(r'limit\s+(\d+)', query_lower)
        if limit_match:
            filters["limit"] = int(limit_match.group(1))

    return project, tower_block, sheet, filters, all_projects_flag, tower_wise_flag, consolidate_flag

# --------------------------------------------------
# CORE DATA LOADING FUNCTIONS
# --------------------------------------------------

def normalize_col(col) -> str:
    return (
        str(col)
        .replace("\xa0", " ")
        .replace("\n", " ")
        .strip()
    )
def filter_columns(df: pd.DataFrame, allowed_columns: list) -> pd.DataFrame:
    allowed_norm = {normalize_col(c) for c in allowed_columns}

    col_map = {
        col: normalize_col(col)
        for col in df.columns
    }

    keep_cols = [
        original
        for original, normalized in col_map.items()
        if normalized in allowed_norm
    ]

    if not keep_cols:
        print(f"WARNING: No matching columns found!")
        print(f"Expected: {allowed_norm}")
        print(f"Available: {set(col_map.values())}")
        return df[[]]

    return df[keep_cols]

@lru_cache(maxsize=50)
def list_files(prefix: str) -> List[str]:
    if not COS_BUCKET:
        logger.error("COS_BUCKET_NAME environment variable is not set")
        return []
        
    try:
        files = []
        continuation_token = None

        while True:
            kwargs = {
                "Bucket": COS_BUCKET,
                "Prefix": prefix,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            res = cos.list_objects_v2(**kwargs)
            files.extend(
                o["Key"]
                for o in res.get("Contents", [])
                if not o["Key"].endswith("/") and o["Key"].lower().endswith(".xlsx")
            )

            if not res.get("IsTruncated"):
                break
            continuation_token = res.get("NextContinuationToken")

        return files
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []

def get_latest_structure_file(prefix: str) -> Dict:
    latest = None
    latest_date = None

    for key in list_files(prefix):
        fname = key.split("/")[-1]
        match = STRUCTURE_PATTERN.search(fname)
        if not match:
            continue

        fdate = datetime.strptime(match.group(1), "%d-%m-%Y")

        if latest_date is None or fdate > latest_date:
            latest = {
                "file": fname,
                "key": key,
                "date": fdate,
            }
            latest_date = fdate

    return latest

# --------------------------------------------------
# DELAY CALCULATION FUNCTIONS
# --------------------------------------------------

def calculate_delay(baseline_date, actual_date) -> int:
    """
    Calculate delay in days between baseline and actual finish dates.
    Returns negative for early completion, 0 for on-time, positive for delays.
    """
    try:
        if pd.isna(baseline_date) or pd.isna(actual_date):
            return 0
        
        # Convert to datetime if strings
        if isinstance(baseline_date, str):
            baseline_date = pd.to_datetime(baseline_date, errors='coerce')
        if isinstance(actual_date, str):
            actual_date = pd.to_datetime(actual_date, errors='coerce')
        
        if pd.isna(baseline_date) or pd.isna(actual_date):
            return 0
        
        delay_days = (actual_date - baseline_date).days
        return int(delay_days) if not pd.isna(delay_days) else 0
    except Exception:
        return 0

def load_dataframe_with_delays(
    key: str,
    project_name: str,
    sheet_name: str,
) -> pd.DataFrame:
    """Load DataFrame with delay calculations and proper tower extraction."""
    if not COS_BUCKET:
        raise ValueError("COS_BUCKET_NAME environment variable is not set")

    obj = cos.get_object(Bucket=COS_BUCKET, Key=key)
    data = obj["Body"].read()

    project_key = project_name.lower().strip()
    allowed_columns = COLUMN_MAPPINGS[project_key]
    header_row = PROJECT_HEADER_ROW.get(project_key, 0)

    with pd.ExcelFile(BytesIO(data), engine="openpyxl") as xls:
        actual_sheet = None
        for s in xls.sheet_names:
            if s.strip().lower() == sheet_name.strip().lower():
                actual_sheet = s
                break

        if not actual_sheet:
            raise ValueError(f"Sheet not found: {sheet_name}")

        # Use project-specific header row
        df = pd.read_excel(
            xls,
            sheet_name=actual_sheet,
            header=header_row
        )

    # Normalize column headers
    df.columns = [normalize_col(c) for c in df.columns]

    # Rename Task Name to Structure Task for Eden
    if project_name.lower() == "eden":
        df.rename(columns={"Task Name": "Structure Task"}, inplace=True)

    # Drop fully empty rows
    df = df.dropna(how="all")

    # Filter only required columns
    df = filter_columns(df, allowed_columns)

    # Replace NaN values with None
    df = df.replace([np.inf, -np.inf, np.nan], None)
    
    # ============================================
    # FIX TOWER COLUMN FOR EDEN PROJECT
    # ============================================
    
    if project_key == "eden":
        # Find the Tower column (might be "Tower" or "Tower " with space)
        tower_col = None
        for col in df.columns:
            if 'tower' in col.lower():
                tower_col = col
                break
        
        if tower_col and tower_col in df.columns:
            # Forward fill to handle merged cells
            df[tower_col] = df[tower_col].ffill()
            
            # Clean the Tower column values
            def clean_tower_value(value):
                if pd.isna(value) or value is None:
                    return None
                
                value_str = str(value).strip()
                
                # If it's already "Tower 4", extract just "4"
                if value_str.upper().startswith('TOWER'):
                    # Extract number after "Tower"
                    match = re.search(r'tower\s*(\d+)', value_str, re.IGNORECASE)
                    if match:
                        return match.group(1)
                    else:
                        # Check if it's like "Tower F" or similar
                        match = re.search(r'tower\s*([a-z])', value_str, re.IGNORECASE)
                        if match:
                            return match.group(1).upper()
                
                # If it's "NTA", try to extract from Task Name
                if value_str.upper() in ['NTA', 'N/A', 'NOT AVAILABLE', '']:
                    # Try to extract from Task Name if available
                    task_name_col = None
                    for col in df.columns:
                        if 'task' in col.lower() or 'name' in col.lower():
                            task_name_col = col
                            break
                    
                    # This will be handled in the main function
                    return None
                
                # Return cleaned value
                return value_str
            
            # Apply cleaning
            df['Cleaned_Tower'] = df[tower_col].apply(clean_tower_value)
            
            # Also create a column with original tower for reference
            df['Original_Tower'] = df[tower_col].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    
    # ============================================
    # DELAY CALCULATION LOGIC
    # ============================================
    
    # Determine date columns based on project
    baseline_col = None
    actual_col = None
    
    # Find baseline date column
    for col in df.columns:
        col_lower = str(col).lower()
        if 'baseline' in col_lower and 'finish' in col_lower:
            baseline_col = col
            break
    
    # Find actual finish date column
    for col in df.columns:
        col_lower = str(col).lower()
        if ('finish' in col_lower and 'baseline' not in col_lower) or 'actual' in col_lower:
            actual_col = col
            break
    
    # Calculate delays if we have both date columns
    if baseline_col and actual_col:
        # Clean date columns
        for col in [baseline_col, actual_col]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate delay days
        df['Delay_Days'] = df.apply(
            lambda row: calculate_delay(row[baseline_col], row[actual_col]),
            axis=1
        )
        
        # Add status column
        df['Status'] = df['Delay_Days'].apply(
            lambda x: 'Delayed' if x > 0 else ('Early' if x < 0 else 'On-Time')
        )
    
    return df

def get_delayed_activities_for_project(
    project_name: str,
    min_delay_days: Optional[int] = None,
    max_delay_days: Optional[int] = None,
    tower: Optional[str] = None,
    block: Optional[str] = None,
    sheet: Optional[str] = None,
    tower_wise: bool = False,
    limit: int = 100,
    activity_id: Optional[str] = None,
    activity_name: Optional[str] = None,
    consolidate: bool = False,
    part: Optional[str] = None,
    domain: Optional[str] = None,
    percent_complete: Optional[str] = None,
    status: str = "delayed",
) -> Dict:
    """Get delayed or on-time activities for a specific project."""
    project_key = project_name.lower().strip()

    if project_key not in COLUMN_MAPPINGS:
        raise HTTPException(status_code=400, detail="Project not supported")

    file_info = get_latest_structure_file(f"{project_name}/")
    if not file_info:
        raise HTTPException(status_code=404, detail=f"No Structure Work Tracker found for {project_name}")

    # For tower-wise analysis, use special function
    # Note: Currently tower-wise analysis defaults to delayed activities only.
    if tower_wise and project_key == "eden":
        return get_eden_tower_wise_analysis(
            project_name=project_name,
            file_info=file_info,
            min_delay_days=min_delay_days,
            max_delay_days=max_delay_days,
        )
    
    all_delayed_activities = []
    sheets_to_process = []
    
    # Determine which sheets to process
    if sheet:
        sheet_str = str(sheet).strip()
        if sheet_str in PROJECT_SHEETS.get(project_key, []):
            sheets_to_process = [sheet_str]
        else:
            mapped_sheet = WAVE_CITY_SHEETS.get(sheet_str.upper())
            if mapped_sheet and mapped_sheet in PROJECT_SHEETS.get(project_key, []):
                sheets_to_process = [mapped_sheet]
            else:
                raise HTTPException(status_code=400, detail=f"Sheet '{sheet_str}' not found for project {project_name}")
    else:
        sheets_to_process = PROJECT_SHEETS.get(project_key, [])
    
    for sheet_name in sheets_to_process:
        try:
            df = load_dataframe_with_delays(
                file_info["key"],
                project_name,
                sheet_name,
            )
            
            # Filter based on status
            if status == "on_time":
                # On-time means Delay_Days <= 0 (includes early)
                delayed_df = df[df['Delay_Days'] <= 0].copy()
            elif status == "all":
                delayed_df = df.copy()
            else:
                # Default to delayed (Delay_Days > 0)
                delayed_df = df[df['Delay_Days'] > 0].copy()
            
            if not delayed_df.empty:
                # Apply filters
                if min_delay_days is not None:
                    delayed_df = delayed_df[delayed_df['Delay_Days'] >= min_delay_days]
                
                if max_delay_days is not None:
                    delayed_df = delayed_df[delayed_df['Delay_Days'] <= max_delay_days]
                
                # Apply activity ID filter if specified
                if activity_id:
                    # Check for Activity ID or Activity No.
                    id_col = None
                    if 'Activity ID' in delayed_df.columns:
                        id_col = 'Activity ID'
                    elif 'Activity No.' in delayed_df.columns:
                        id_col = 'Activity No.'
                    
                    if id_col:
                        delayed_df = delayed_df[
                            delayed_df[id_col].astype(str).str.contains(
                                str(activity_id), case=False, na=False, regex=False
                            )
                        ]
                
                # Apply Part filter (Wave City Club)
                if part and 'Part' in delayed_df.columns:
                    delayed_df = delayed_df[
                        delayed_df['Part'].astype(str).str.contains(
                            str(part), case=False, na=False, regex=False
                        )
                    ]
                
                # Apply Domain filter (Wave City Club)
                if domain and 'Domain' in delayed_df.columns:
                    delayed_df = delayed_df[
                        delayed_df['Domain'].astype(str).str.contains(
                            str(domain), case=False, na=False, regex=False
                        )
                    ]
                
                # Apply % Complete filter (Wave City Club)
                if percent_complete and '% Complete' in delayed_df.columns:
                    target_str = str(percent_complete).replace('%', '').strip()
                    
                    # Try to parse target as float for smart matching
                    try:
                        target_float = float(target_str)
                    except:
                        target_float = None

                    def match_percent_complete(x):
                        if pd.isna(x):
                            return False
                        x_str = str(x).replace('%', '').strip()
                        
                        # Handle inequality
                        try:
                            x_float = float(x_str)
                            # Normalize x to 0-100 range if it's 0-1
                            if x_float <= 1.0 and x_float > 0:
                                x_float_100 = x_float * 100
                            else:
                                x_float_100 = x_float
                            
                            if target_str.startswith('<'):
                                threshold = float(target_str.replace('<', '').strip())
                                # Normalize threshold
                                if threshold <= 1.0 and threshold > 0:
                                    threshold *= 100
                                return x_float_100 < threshold
                            elif target_str.startswith('>'):
                                threshold = float(target_str.replace('>', '').strip())
                                if threshold <= 1.0 and threshold > 0:
                                    threshold *= 100
                                return x_float_100 > threshold
                        except:
                            pass

                        # 1. Simple string match
                        if target_str in x_str:
                            return True
                            
                        # 2. Smart numeric match
                        if target_float is not None:
                            try:
                                x_float = float(x_str)
                                
                                # Exact match
                                if x_float == target_float:
                                    return True
                                    
                                # Handle 1.0 vs 100%
                                # If target is 100, match 1.0
                                if target_float == 100 and x_float == 1.0:
                                    return True
                                # If target is 1.0, match 100
                                if target_float == 1.0 and x_float == 100:
                                    return True
                                    
                                # Handle 0.5 vs 50%
                                if target_float > 1.0 and x_float <= 1.0:
                                    if abs(x_float * 100 - target_float) < 0.1:
                                        return True
                                        
                                if target_float <= 1.0 and x_float > 1.0:
                                    if abs(target_float * 100 - x_float) < 0.1:
                                        return True
                            except:
                                pass
                        
                        return False

                    delayed_df = delayed_df[
                        delayed_df['% Complete'].apply(match_percent_complete)
                    ]

                # Determine activity column name
                activity_col = None
                for col in ['Activity Name', 'Structure Task', 'Task Name']:
                    if col in delayed_df.columns:
                        activity_col = col
                        break

                # Apply activity name filter if specified - IMPROVED VERSION
                if activity_name and activity_col:
                    activity_name_lower = str(activity_name).lower()
                    
                    # Clean the activity name for better matching
                    # Remove common query words that might have been captured
                    activity_name_clean = re.sub(
                        r'\s+(show|get|find|search|look|delay|in|of|for|from|activity)\s+', 
                        ' ', 
                        activity_name_lower
                    )
                    activity_name_clean = activity_name_clean.strip()
                    
                    # If activity_name_clean is too short or generic, try to extract main keywords
                    if len(activity_name_clean) < 5 or activity_name_clean in ['plinth', 'beam', 'reinforcement']:
                        # Extract main construction keywords
                        keywords = ['plinth', 'beam', 'reinforcement', 'foundation', 'shuttering', 
                                   'concreting', 'casting', 'column', 'roof', 'slab']
                        found_keywords = []
                        for keyword in keywords:
                            if keyword in activity_name_lower:
                                found_keywords.append(keyword)
                        
                        if found_keywords:
                            # Use the found keywords for matching
                            mask = pd.Series([False] * len(delayed_df), index=delayed_df.index)
                            for keyword in found_keywords:
                                mask = mask | delayed_df[activity_col].astype(str).str.lower().str.contains(
                                    keyword, case=False, na=False, regex=False
                                )
                            delayed_df = delayed_df[mask]
                    else:
                        # Try exact match first
                        exact_match = delayed_df[
                            delayed_df[activity_col].astype(str).str.lower() == activity_name_clean
                        ]
                        
                        if not exact_match.empty:
                            delayed_df = exact_match
                        else:
                            # Try partial match with the cleaned activity name
                            delayed_df = delayed_df[
                                delayed_df[activity_col].astype(str).str.lower().str.contains(
                                    activity_name_clean, case=False, na=False, regex=False
                                )
                            ]
                            
                            # If still no matches, try with individual keywords
                            if delayed_df.empty:
                                keywords = activity_name_clean.split()
                                if len(keywords) > 1:
                                    # Try matching any of the keywords
                                    mask = pd.Series([False] * len(df[df['Delay_Days'] > 0]), index=df[df['Delay_Days'] > 0].index)
                                    for keyword in keywords:
                                        if len(keyword) > 3:  # Only use meaningful keywords
                                            mask = mask | df[df['Delay_Days'] > 0][activity_col].astype(str).str.lower().str.contains(
                                                keyword, case=False, na=False, regex=False
                                            )
                                    delayed_df = df[df['Delay_Days'] > 0][mask].copy()
                                    
                                    # Reapply other filters
                                    if min_delay_days is not None:
                                        delayed_df = delayed_df[delayed_df['Delay_Days'] >= min_delay_days]
                                    if max_delay_days is not None:
                                        delayed_df = delayed_df[delayed_df['Delay_Days'] <= max_delay_days]
                
                # Apply tower/block filter
                tower_block = tower or block
                if tower_block:
                    tower_block_str = str(tower_block).upper()
                    
                    # For Eden project
                    if project_key == "eden":
                        # Try different tower columns in order
                        tower_columns = ['Cleaned_Tower', 'Original_Tower']
                        
                        filtered = False
                        for tower_col in tower_columns:
                            if tower_col in delayed_df.columns:
                                # Extract just the number from tower block (e.g., "4" from "Tower 4")
                                tower_num = tower_block_str.replace('T', '').replace('TOWER', '').strip()
                                
                                if tower_num:  # If we have a number to filter by
                                    # Filter rows where tower column contains this number
                                    mask = delayed_df[tower_col].astype(str).str.contains(
                                        tower_num, case=False, na=False, regex=False
                                    )
                                    delayed_df = delayed_df[mask]
                                    filtered = True
                                    break
                        
                        if not filtered:
                            # No tower columns found, try any column with "tower" in name
                            for col in delayed_df.columns:
                                if 'tower' in col.lower():
                                    tower_num = tower_block_str.replace('T', '').replace('TOWER', '').strip()
                                    mask = delayed_df[col].astype(str).str.contains(
                                        tower_num, case=False, na=False, regex=False
                                    )
                                    delayed_df = delayed_df[mask]
                                    break
                    
                    # For Wave City Club project
                    elif project_key == "wave city club" and 'Block' in delayed_df.columns:
                        block_num = tower_block_str.replace('B', '').replace('BLOCK', '').strip()
                        delayed_df = delayed_df[
                            delayed_df['Block'].astype(str).str.contains(block_num, case=False, na=False, regex=False)
                        ]
                
                # Sort by delay (largest first)
                delayed_df = delayed_df.sort_values('Delay_Days', ascending=False)
                
                # For Eden, prioritize actual Towers over NTA (None in Cleaned_Tower) if no specific tower requested
                if project_key == "eden" and not tower_block and 'Cleaned_Tower' in delayed_df.columns:
                    delayed_df['Has_Tower'] = delayed_df['Cleaned_Tower'].notna()
                    delayed_df = delayed_df.sort_values(['Has_Tower', 'Delay_Days'], ascending=[False, False])
                    # Remove temporary column
                    if 'Has_Tower' in delayed_df.columns:
                        delayed_df = delayed_df.drop(columns=['Has_Tower'])
                
                # Limit results if NOT consolidating (we need all data for consolidation)
                if not consolidate:
                    delayed_df = delayed_df.head(limit)
                
                # Convert to records
                for _, row in delayed_df.iterrows():
                    record = {}
                    for col in delayed_df.columns:
                        # Skip internal columns in response
                        if col in ['Cleaned_Tower', 'Original_Tower']:
                            continue
                            
                        value = row[col]
                        
                        # Special handling for % Complete
                        if '% Complete' in col:
                            if pd.isna(value):
                                record[col] = None
                            else:
                                try:
                                    val_float = float(value)
                                    if val_float <= 1.0:
                                        record[col] = f"{int(val_float * 100)}%"
                                    else:
                                        record[col] = f"{int(val_float)}%"
                                except:
                                    record[col] = str(value)
                            continue
                            
                        if pd.isna(value):
                            record[col] = None
                        elif isinstance(value, (pd.Timestamp, datetime)):
                            # Return only the date part as requested
                            record[col] = value.strftime("%Y-%m-%d")
                        elif isinstance(value, (np.integer, np.int64)):
                            record[col] = int(value)
                        elif isinstance(value, (np.floating, np.float64)):
                            if np.isfinite(value):
                                record[col] = float(value)
                            else:
                                record[col] = None
                        else:
                            if col == 'Delay_Days':
                                record[col] = safe_int(value)
                            else:
                                record[col] = str(value).strip() if value else None
                    
                    # Add tower information if available
                    if 'Cleaned_Tower' in delayed_df.columns and pd.notna(row.get('Cleaned_Tower')):
                        record['Tower'] = str(row['Cleaned_Tower'])
                    elif 'Original_Tower' in delayed_df.columns and pd.notna(row.get('Original_Tower')):
                        record['Tower'] = str(row['Original_Tower'])
                    
                    record["sheet"] = sheet_name
                    record["project"] = project_name
                    all_delayed_activities.append(record)
                    
        except Exception as e:
            logger.error(f"Error processing sheet {sheet_name}: {e}")
            continue
    
    # CONSOLIDATION LOGIC
    if consolidate and all_delayed_activities:
        # Group by Activity Name and Tower/Block
        consolidated_groups = {}
        
        for act in all_delayed_activities:
            # Create a key based on Activity Name and Tower (if available)
            activity_name = act.get("Activity Name") or act.get("Structure Task") or act.get("Task Name") or "Unknown"
            tower = act.get("Tower") or act.get("Block") or "Unknown"
            
            key = f"{activity_name}|{tower}"
            
            if key not in consolidated_groups:
                consolidated_groups[key] = {
                    "activity_name": activity_name,
                    "location": tower,
                    "count": 0,
                    "total_delay": 0,
                    "max_delay": 0,
                    "examples": []
                }
            
            group = consolidated_groups[key]
            delay = safe_int(act.get("Delay_Days", 0))
            
            group["count"] += 1
            group["total_delay"] += delay
            group["max_delay"] = max(group["max_delay"], delay)
            
            # Keep top 3 examples
            if len(group["examples"]) < 3:
                group["examples"].append(act)
        
        # Convert to list
        consolidated_results = list(consolidated_groups.values())
        
        # Sort by total delay
        consolidated_results.sort(key=lambda x: x["total_delay"], reverse=True)
        
        # Calculate summary metrics for consolidated view
        total_delay_days = sum(item["total_delay"] for item in consolidated_results)
        max_delay = max(item["max_delay"] for item in consolidated_results) if consolidated_results else 0
        
        summary = {
            "total_groups": len(consolidated_results),
            "total_activities_represented": len(all_delayed_activities),
            "total_delay_days": total_delay_days,
            "max_delay": max_delay,
            "average_delay_per_group": round(total_delay_days / len(consolidated_results), 2) if consolidated_results else 0
        }
        
        return {
            "project": project_name,
            "view": "consolidated",
            "summary": summary,
            "consolidated_activities": consolidated_results[:limit] # Limit the groups returned
        }

    # STANDARD LOGIC (Non-consolidated)
    # Sort all activities by delay days
    all_delayed_activities.sort(key=lambda x: safe_int(x.get('Delay_Days', 0)), reverse=True)
    
    # Apply final limit
    final_results = all_delayed_activities[:limit]
    
    # Calculate summary metrics
    if all_delayed_activities:
        total_delay_days = 0
        max_delay = 0
        
        for act in all_delayed_activities:
            delay = safe_int(act.get('Delay_Days', 0))
            total_delay_days += delay
            if delay > max_delay:
                max_delay = delay
        
        summary = {
            "total_delayed_activities": len(all_delayed_activities),
            "total_delay_days": total_delay_days,
            "max_delay": max_delay,
            "average_delay": round(total_delay_days / len(all_delayed_activities), 2) if all_delayed_activities else 0
        }
    else:
        summary = {
            "total_delayed_activities": 0,
            "total_delay_days": 0,
            "max_delay": 0,
            "average_delay": 0
        }

    return {
        "project": project_name,
        "summary": summary,
        "delayed_activities": final_results
    }
    
def get_eden_tower_wise_analysis(
    project_name: str,
    file_info: Dict,
    min_delay_days: Optional[int] = None,
    max_delay_days: Optional[int] = None,
    limit: int = 100
) -> Dict:
    """Get tower-wise delay analysis for Eden project."""
    sheets_to_process = PROJECT_SHEETS.get("eden", [])
    
    # Dictionary to store tower-wise data
    tower_data = {}
    
    for sheet_name in sheets_to_process:
        try:
            df = load_dataframe_with_delays(
                file_info["key"],
                project_name,
                sheet_name,
            )
            
            # Filter for delayed activities (Delay_Days > 0)
            delayed_df = df[df['Delay_Days'] > 0].copy()
            
            if not delayed_df.empty:
                # Apply delay range filters
                if min_delay_days is not None:
                    delayed_df = delayed_df[delayed_df['Delay_Days'] >= min_delay_days]
                
                if max_delay_days is not None:
                    delayed_df = delayed_df[delayed_df['Delay_Days'] <= max_delay_days]
                
                # Group by Cleaned_Tower (or Original_Tower if Cleaned_Tower is not available)
                tower_column = 'Cleaned_Tower' if 'Cleaned_Tower' in delayed_df.columns else 'Original_Tower'
                
                if tower_column in delayed_df.columns:
                    for tower, tower_group in delayed_df.groupby(tower_column):
                        if pd.isna(tower):
                            continue
                        
                        tower_str = str(tower).strip()
                        
                        # Skip NTA or empty values
                        if not tower_str or tower_str.upper() in ['NTA', 'N/A', 'NOT AVAILABLE', 'NONE']:
                            continue
                        
                        # Clean tower string - extract just the number if it's "Tower 4"
                        if tower_str.upper().startswith('TOWER'):
                            match = re.search(r'tower\s*(\d+)', tower_str, re.IGNORECASE)
                            if match:
                                tower_str = match.group(1)
                        
                        # Initialize tower entry if not exists
                        if tower_str not in tower_data:
                            tower_data[tower_str] = {
                                "total_activities": 0,
                                "total_delay_days": 0,
                                "max_delay": 0,
                                "activities": []
                            }
                        
                        # Update tower statistics
                        tower_activities = []
                        for _, row in tower_group.iterrows():
                            # Create activity record
                            activity = {}
                            for col in delayed_df.columns:
                                # Skip internal columns
                                if col in ['Cleaned_Tower', 'Original_Tower']:
                                    continue
                                    
                                value = row[col]
                                if pd.isna(value):
                                    activity[col] = None
                                elif isinstance(value, (pd.Timestamp, datetime)):
                                    activity[col] = value.strftime("%Y-%m-%d")
                                elif isinstance(value, (np.integer, np.int64)):
                                    activity[col] = int(value)
                                elif isinstance(value, (np.floating, np.float64)):
                                    if np.isfinite(value):
                                        activity[col] = float(value)
                                    else:
                                        activity[col] = None
                                else:
                                    if col == 'Delay_Days':
                                        activity[col] = safe_int(value)
                                    else:
                                        activity[col] = str(value).strip() if value else None
                            
                            # Add tower information
                            activity["Tower"] = tower_str
                            activity["sheet"] = sheet_name
                            activity["project"] = project_name
                            tower_activities.append(activity)
                        
                        # Update tower data
                        tower_data[tower_str]["total_activities"] += len(tower_activities)
                        for act in tower_activities:
                            delay = safe_int(act.get('Delay_Days', 0))
                            tower_data[tower_str]["total_delay_days"] += delay
                            tower_data[tower_str]["max_delay"] = max(
                                tower_data[tower_str]["max_delay"], 
                                delay
                            )
                            # Add activities (up to limit)
                            if len(tower_data[tower_str]["activities"]) < limit:
                                tower_data[tower_str]["activities"].append(act)
                    
        except Exception as e:
            logger.error(f"Error processing sheet {sheet_name} for tower-wise analysis: {e}")
            continue
    
    # Prepare tower-wise results
    tower_results = []
    for tower, data in tower_data.items():
        if data["total_activities"] > 0:
            avg_delay = round(data["total_delay_days"] / data["total_activities"], 2)
            
            tower_results.append({
                "tower": tower,
                "total_activities": data["total_activities"],
                "total_delay_days": data["total_delay_days"],
                "average_delay": avg_delay,
                "max_delay": data["max_delay"],
                "sample_activities": data["activities"][:5]  # Show first 5 activities
            })
    
    # Sort towers by total delay days (highest first)
    tower_results.sort(key=lambda x: x["total_delay_days"], reverse=True)
    
    # Calculate overall summary
    if tower_results:
        total_activities = sum(t["total_activities"] for t in tower_results)
        total_delay_days = sum(t["total_delay_days"] for t in tower_results)
        max_delay = max(t["max_delay"] for t in tower_results)
        
        summary = {
            "total_towers": len(tower_results),
            "total_delayed_activities": total_activities,
            "total_delay_days": total_delay_days,
            "max_delay": max_delay,
            "average_delay_per_tower": round(total_delay_days / len(tower_results), 2) if tower_results else 0
        }
    else:
        summary = {
            "total_towers": 0,
            "total_delayed_activities": 0,
            "total_delay_days": 0,
            "max_delay": 0,
            "average_delay_per_tower": 0
        }
    
    return {
        "project": project_name,
        "analysis_type": "tower_wise",
        "summary": summary,
        "tower_wise_results": tower_results
    }    

def analyze_all_projects(filters: Dict) -> Dict:
    """Analyze delays across all projects."""
    all_projects_data = []
    total_summary = {
        "total_projects": 0,
        "total_delayed_activities": 0,
        "total_delay_days": 0,
        "max_delay": 0,
        "projects_analyzed": []
    }
    
    # Clean filters
    clean_filters = {}
    for key, value in filters.items():
        if value is not None:
            clean_filters[key] = value
    
    for project_name in ["Eden", "Wave City Club"]:
        try:
            # Get project data
            project_data = get_delayed_activities_for_project(
                project_name=project_name,
                min_delay_days=clean_filters.get('min_delay_days'),
                max_delay_days=clean_filters.get('max_delay_days'),
                tower=clean_filters.get('tower'),
                block=clean_filters.get('block'),             
            )
            
            # Extract summary
            project_summary = project_data["summary"]
            
            # Prepare project result
            project_result = {
                "project": project_name,
                "total_delayed_activities": project_summary["total_delayed_activities"],
                "total_delay_days": project_summary["total_delay_days"],
                "max_delay": project_summary["max_delay"],
                "average_delay": project_summary["average_delay"],
                "sample_activities": project_data["delayed_activities"][:3] if project_data["delayed_activities"] else []
            }
            
            all_projects_data.append(project_result)
            
            # Update overall totals
            total_summary["total_projects"] += 1
            total_summary["total_delayed_activities"] += project_summary["total_delayed_activities"]
            total_summary["total_delay_days"] += project_summary["total_delay_days"]
            total_summary["max_delay"] = max(total_summary["max_delay"], project_summary["max_delay"])
            
            total_summary["projects_analyzed"].append(project_name)
            
        except Exception as e:
            logger.error(f"Error analyzing project {project_name}: {e}")
            continue
    
    # Calculate overall averages
    if total_summary["total_projects"] > 0:
        total_summary["average_delay_per_project"] = round(
            total_summary["total_delay_days"] / total_summary["total_projects"], 2
        )
        total_summary["average_activities_per_project"] = round(
            total_summary["total_delayed_activities"] / total_summary["total_projects"], 2
        )
    
    return {
        "query_type": "all_projects_analysis",
        "total_summary": total_summary,
        "projects_analyzed": all_projects_data
    }

# --------------------------------------------------
# SINGLE ENDPOINT FOR ALL ANALYSIS
# --------------------------------------------------

# In the analyze_delays endpoint function:
@app.post("/analyze")
def analyze_delays(query_request: QueryRequest):
    """
    Single endpoint for all delay analysis queries.
    """
    try:
        # Handle potential None query
        if query_request.query is None:
            raise HTTPException(status_code=400, detail="Query string is required")
            
        query = query_request.query.strip()
        
        # Parse the query
        project, tower_block, sheet, filters, all_projects_flag, tower_wise_flag, consolidate_flag = parse_query(query)
        
        # Handle "all projects" query
        if all_projects_flag:
            return analyze_all_projects(filters)
        
        # Use provided project or parsed project
        project_name = query_request.project or project
        
        # If no project can be determined
        if not project_name:
            # Try to infer from tower/block
            if tower_block:
                tower_block_str = str(tower_block).upper()
                if tower_block_str.startswith('B'):
                    project_name = "Wave City Club"
                elif tower_block_str.isdigit():
                    project_name = "Eden"
        
        # If still no project, return helpful message
        if not project_name:
            return {
                "query": query,
                "error": "Could not determine project from query. Please specify 'Eden' or 'Wave City Club' or a specific tower/block."
            }
        
        project_key = project_name.lower().strip()
        
        # Prepare parameters
        params = {
            "project_name": project_name,
            "min_delay_days": filters.get('min_delay_days'),
            "max_delay_days": filters.get('max_delay_days'),
            "tower_wise": tower_wise_flag,
            "activity_id": filters.get('activity_id'),
            "activity_name": filters.get('activity_name'),
            "part": filters.get('part'),
            "domain": filters.get('domain'),
            "percent_complete": filters.get('percent_complete'),
            "status": filters.get('status', 'delayed'),
            "consolidate": consolidate_flag,
            "limit": filters.get('limit', 100),
        }
        
        # Add tower/block/sheet filter ONLY if specific tower is mentioned
        # For "all towers", we don't want to filter by a specific tower
        if tower_block and not ("all" in query.lower() and "tower" in query.lower()):
            tower_block_str = str(tower_block)
            if project_key == "eden":
                params['tower'] = tower_block_str
            elif project_key == "wave city club":
                params['block'] = tower_block_str
        
        if sheet:
            params['sheet'] = str(sheet)
        
        # Get project analysis
        result = get_delayed_activities_for_project(**params)
        return shape_special_project_response(result, query, filters)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Construction Delay Analysis API",
        "available_projects": ["Eden", "Wave City Club"],
        "features": ["Natural language query parsing", "Delay calculation", "Multi-project analysis"]
    }
