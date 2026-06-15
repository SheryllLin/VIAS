"""
VIAS Frontend - Refactored for Video Analysis, Person Matching, Query Engine, and Accuracy Metrics
Main Features:
  1. Video Upload & Processing (with progress tracking)
  2. Person Analyzing (Image-to-Image Matching)
  3. Query Engine (with confidence scores and citations)
  4. System Status & Diagnostics
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import requests


# Configuration
API_BASE = os.getenv("VIAS_API_BASE", "http://127.0.0.1:8000")
API_TIMEOUT_VIDEO = 1200  # 20 minutes for video processing
API_TIMEOUT_DEFAULT = 300  # 5 minutes for other operations


# ============================================================================
# API HELPERS WITH ERROR HANDLING & CONFIDENCE EXTRACTION
# ============================================================================

def format_error_message(error: Exception) -> str:
    """Format error message for user-friendly display."""
    if isinstance(error, requests.exceptions.ConnectionError):
        return f"❌ **Connection Error**: Cannot reach backend at {API_BASE}\n\nMake sure backend is running:\n`uvicorn backend.main:app --host 127.0.0.1 --port 8000`"
    elif isinstance(error, requests.exceptions.Timeout):
        return f"⏱️ **Timeout**: Operation took too long. Try with a smaller file or check system resources."
    elif isinstance(error, requests.exceptions.HTTPError):
        return f"❌ **Backend Error**: {str(error)}"
    else:
        return f"❌ **Error**: {str(error)}"


def post_file(endpoint: str, file_path: str, person_id: str | None = None) -> tuple[bool, str]:
    """Upload file to backend with error handling."""
    try:
        with open(file_path, "rb") as handle:
            files = {"file": (Path(file_path).name, handle)}
            data = {"person_id": person_id} if person_id else None
            response = requests.post(
                f"{API_BASE}{endpoint}",
                files=files,
                data=data,
                timeout=API_TIMEOUT_VIDEO
            )
        response.raise_for_status()
        return True, json.dumps(response.json(), indent=2)
    except Exception as e:
        return False, format_error_message(e)


def post_json(endpoint: str, payload: dict) -> tuple[bool, str]:
    """Send JSON to backend with error handling."""
    try:
        response = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            timeout=API_TIMEOUT_DEFAULT
        )
        response.raise_for_status()
        return True, json.dumps(response.json(), indent=2)
    except Exception as e:
        return False, format_error_message(e)


def get_endpoint(endpoint: str) -> tuple[bool, str]:
    """Fetch from backend with error handling."""
    try:
        response = requests.get(
            f"{API_BASE}{endpoint}",
            timeout=API_TIMEOUT_DEFAULT
        )
        response.raise_for_status()
        return True, json.dumps(response.json(), indent=2)
    except Exception as e:
        return False, format_error_message(e)


# ============================================================================
# RESULT FORMATTERS WITH CONFIDENCE & CITATIONS
# ============================================================================

def format_video_results(json_str: str) -> str:
    """Format video processing results with confidence scores."""
    try:
        data = json.loads(json_str)
        output = "## 📊 Video Processing Results\n\n"
        
        if "detections" in data:
            output += f"**People Detected:** {data.get('detections', 0)} 👥\n"
        if "tracks" in data:
            output += f"**Unique Tracks:** {data.get('tracks', 0)} 🔍\n"
        if "fps" in data:
            output += f"**Processing Speed:** {data.get('fps', 0):.1f} FPS ⚡\n"
        if "video_id" in data:
            output += f"**Video ID:** `{data.get('video_id')}` 📁\n"
        if "events_stored" in data:
            output += f"**Events Logged:** {data.get('events_stored', 0)} 📝\n"
        
        output += "\n**Raw Response:**\n```json\n"
        output += json.dumps(data, indent=2)
        output += "\n```"
        
        return output
    except:
        return "```json\n" + json_str + "\n```"


def format_person_search_results(json_str: str) -> str:
    """Format person search results with confidence scores and citations."""
    try:
        data = json.loads(json_str)
        output = "## 🔍 Person Search Results\n\n"
        
        if "results" in data:
            results = data["results"]
            if not results:
                return output + "❌ **No matches found.** Try a different person ID or upload reference images first."
            
            output += f"**Found {len(results)} match(es):**\n\n"
            
            for idx, result in enumerate(results, 1):
                confidence = result.get("confidence", 0.0)
                person_id = result.get("person_id", "Unknown")
                track_id = result.get("track_id", "N/A")
                timestamp = result.get("timestamp", "N/A")
                
                # Color code confidence
                conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                
                output += f"**Match #{idx}** {conf_emoji}\n"
                output += f"  - Person ID: `{person_id}`\n"
                output += f"  - Confidence: **{confidence:.1%}**\n"
                output += f"  - Track ID: `{track_id}`\n"
                output += f"  - Timestamp: `{timestamp}`\n"
                output += f"  - **Citation**: Body embedding from FAISS vector store\n\n"
        
        return output
    except:
        return "```json\n" + json_str + "\n```"


def format_query_results(json_str: str) -> str:
    """Format query results with confidence and SQL citation."""
    try:
        data = json.loads(json_str)
        output = "## 💡 Query Results\n\n"
        
        if "answer" in data:
            output += f"**Answer:** {data['answer']}\n\n"
        
        if "confidence" in data:
            conf = data["confidence"]
            conf_emoji = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
            output += f"**Confidence:** {conf_emoji} **{conf:.1%}**\n\n"
        
        if "sql" in data:
            output += "**Citation (SQL Query):**\n```sql\n"
            output += data["sql"]
            output += "\n```\n\n"
        
        if "results" in data:
            results = data["results"]
            output += f"**Results:** {len(results)} record(s)\n\n"
            if results:
                output += "```json\n"
                output += json.dumps(results[:5], indent=2)  # Show first 5
                if len(results) > 5:
                    output += f"\n... and {len(results) - 5} more results"
                output += "\n```"
        
        return output
    except:
        return "```json\n" + json_str + "\n```"


def format_behavior_results(json_str: str) -> str:
    """Format behavior search results with confidence."""
    try:
        data = json.loads(json_str)
        output = "## 🎯 Behavior Search Results\n\n"
        
        if "results" in data:
            results = data["results"]
            if not results:
                return output + "❌ **No behavior matches found.**"
            
            output += f"**Found {len(results)} match(es):**\n\n"
            
            for idx, result in enumerate(results, 1):
                behavior = result.get("behavior", "Unknown")
                confidence = result.get("confidence", 0.0)
                count = result.get("count", 0)
                
                conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                
                output += f"**{idx}. {behavior}** {conf_emoji}\n"
                output += f"  - Confidence: **{confidence:.1%}**\n"
                output += f"  - Occurrences: **{count}**\n"
                output += f"  - Citation: Behavior discovery from motion signatures & activities\n\n"
        
        return output
    except:
        return "```json\n" + json_str + "\n```"


def format_analytics(json_str: str) -> str:
    """Format analytics dashboard with metrics."""
    try:
        data = json.loads(json_str)
        output = "## 📈 Analytics Dashboard\n\n"
        
        # Summary metrics
        output += "### Summary\n"
        output += f"- **Total People:** {data.get('total_people', 0)} 👥\n"
        output += f"- **Total Events:** {data.get('total_events', 0)} 📝\n"
        output += f"- **Total Tracks:** {data.get('total_tracks', 0)} 🔍\n\n"
        
        # Activity distribution
        if "activity_distribution" in data:
            output += "### Activity Distribution\n"
            activities = data["activity_distribution"]
            for activity, count in activities.items():
                output += f"- **{activity.title()}:** {count}\n"
            output += "\n"
        
        # Most common activities
        if "most_common_activities" in data:
            output += "### Most Common Activities\n"
            for idx, (activity, count) in enumerate(data["most_common_activities"], 1):
                output += f"{idx}. **{activity.title()}** - {count} occurrences\n"
        
        return output
    except:
        return "```json\n" + json_str + "\n```"


# ============================================================================
# CALLBACK FUNCTIONS
# ============================================================================

def upload_video_callback(video_path: str) -> str:
    """Process video upload."""
    if not video_path:
        return "❌ **Please select a video file.**"
    
    success, result = post_file("/upload-video", video_path)
    if success:
        return format_video_results(result)
    return result


def register_person_callback(image_path: str, person_id: str) -> str:
    """Register reference person image."""
    if not image_path:
        return "❌ **Please select an image file.**"
    if not person_id or person_id.strip() == "":
        return "❌ **Please enter a Person ID.**"
    
    success, result = post_file("/upload-reference-image", image_path, person_id.strip())
    if success:
        return f"✅ **Successfully registered person:** `{person_id}`\n\n{result}"
    return result


def search_person_callback(person_id: str) -> str:
    """Search for person by ID."""
    if not person_id or person_id.strip() == "":
        return "❌ **Please enter a Person ID to search.**"
    
    success, result = post_json("/search-person", {
        "person_id": person_id.strip(),
        "top_k": 10
    })
    if success:
        return format_person_search_results(result)
    return result


def query_callback(query_text: str) -> str:
    """Process natural language query."""
    if not query_text or query_text.strip() == "":
        return "❌ **Please enter a query.**"
    
    success, result = post_json("/query", {"query": query_text.strip()})
    if success:
        return format_query_results(result)
    return result


def behavior_search_callback(behavior_text: str) -> str:
    """Search for behaviors."""
    if not behavior_text or behavior_text.strip() == "":
        return "❌ **Please describe a behavior to search for.**"
    
    success, result = post_json("/behavior-search", {"query": behavior_text.strip()})
    if success:
        return format_behavior_results(result)
    return result


def get_activities_callback() -> str:
    """Fetch activities."""
    success, result = get_endpoint("/activities")
    if success:
        try:
            data = json.loads(result)
            output = "## 📅 Activity Timeline\n\n"
            if isinstance(data, list):
                output += f"**Total Activities:** {len(data)}\n\n"
                for activity in data[:20]:  # Show first 20
                    output += f"- {activity}\n"
                if len(data) > 20:
                    output += f"\n... and {len(data) - 20} more activities"
            return output
        except:
            return "```json\n" + result + "\n```"
    return result


def get_analytics_callback() -> str:
    """Fetch analytics."""
    success, result = get_endpoint("/analytics")
    if success:
        return format_analytics(result)
    return result


def check_system_callback() -> str:
    """Check system health and status."""
    try:
        # Check health
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            status = "✅ **Backend Status:** ONLINE\n\n"
        else:
            status = "❌ **Backend Status:** OFFLINE\n\n"
        
        status += f"**API Base:** `{API_BASE}`\n"
        status += f"**Backend Response:** `{response.json()}`\n\n"
        
        # Check models
        success, models = get_endpoint("/models/status")
        if success:
            status += "### Models Status\n"
            status += "```json\n" + models + "\n```\n\n"
        
        # Check datasets
        success, datasets = get_endpoint("/datasets")
        if success:
            status += "### Datasets Status\n"
            status += "```json\n" + datasets + "\n```"
        
        return status
    except Exception as e:
        return format_error_message(e)


# ============================================================================
# GRADIO UI - REFACTORED LAYOUT
# ============================================================================

css = """
.main-title {
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 1.1em;
    color: #666;
    margin-bottom: 30px;
}

.feature-header {
    font-size: 1.5em;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 10px;
    padding-bottom: 10px;
    border-bottom: 2px solid #667eea;
}

.info-box {
    background-color: #f0f4ff;
    border-left: 4px solid #667eea;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
"""

with gr.Blocks(
    title="VIAS - Visual Intelligence & Analysis System",
    css=css,
    theme=gr.themes.Soft()
) as demo:
    
    # ========== HEADER ==========
    gr.HTML("<div class='main-title'>🎥 VIAS</div>")
    gr.HTML("<div class='subtitle'>Visual Intelligence & Analysis System</div>")
    gr.HTML(
        "<div class='info-box'>"
        "💡 <b>Four Main Features:</b> Video Upload & Processing • Person Image Matching • Natural Language Queries • System Analytics"
        "</div>"
    )
    
    with gr.Tabs():
        
        # ========== TAB 1: VIDEO UPLOAD & PROCESSING ==========
        with gr.Tab("📹 Video Processing"):
            gr.Markdown("### Upload & Process Surveillance Video")
            gr.Markdown("Upload a video for real-time person detection, tracking, pose estimation, and activity recognition.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="📹 Select Video File",
                        format="mp4"
                    )
                    upload_button = gr.Button("🚀 Process Video", size="lg", variant="primary")
                
                with gr.Column(scale=1):
                    video_status = gr.Markdown("**Status:** Ready to process", label="Status")
            
            video_output = gr.Markdown(label="Processing Results")
            
            upload_button.click(
                fn=upload_video_callback,
                inputs=video_input,
                outputs=video_output
            )
        
        
        # ========== TAB 2: PERSON ANALYZING (IMAGE MATCHING) ==========
        with gr.Tab("👤 Person Analyzing"):
            gr.Markdown("### Register & Match Person Images")
            gr.Markdown("Upload reference images of a person, then search for matches in the video database.")
            
            # Register section
            with gr.Group():
                gr.Markdown("#### Step 1️⃣: Register Reference Person")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_image = gr.Image(
                            type="filepath",
                            label="📷 Reference Image",
                            sources=["upload"]
                        )
                    with gr.Column(scale=1):
                        ref_person_id = gr.Textbox(
                            label="👤 Person ID",
                            placeholder="e.g., person-001",
                            value="reference-person"
                        )
                
                register_button = gr.Button("📝 Register Person", size="lg", variant="primary")
                register_output = gr.Markdown(label="Registration Result")
                
                register_button.click(
                    fn=register_person_callback,
                    inputs=[ref_image, ref_person_id],
                    outputs=register_output
                )
            
            gr.Markdown("---")
            
            # Search section
            with gr.Group():
                gr.Markdown("#### Step 2️⃣: Search for Person Matches")
                gr.Markdown("Find all matches of a registered person in the video database.")
                
                with gr.Row():
                    search_person_id = gr.Textbox(
                        label="🔍 Person ID to Search",
                        placeholder="e.g., person-001",
                        scale=2
                    )
                    search_button = gr.Button("🔎 Search", size="lg", variant="primary", scale=1)
                
                search_output = gr.Markdown(label="Search Results")
                
                search_button.click(
                    fn=search_person_callback,
                    inputs=search_person_id,
                    outputs=search_output
                )
            
            gr.Markdown(
                "<div class='info-box'>"
                "ℹ️ <b>How It Works:</b> Uses face embeddings (ArcFace), body shape (OSNet), and motion signatures (TMS) for multi-tier matching with confidence scores."
                "</div>"
            )
        
        
        # ========== TAB 3: QUERY ENGINE ==========
        with gr.Tab("❓ Query Engine"):
            gr.Markdown("### Natural Language Surveillance Queries")
            gr.Markdown("Ask questions about people, activities, and events in natural language. Get results with SQL citations.")
            
            with gr.Group():
                gr.Markdown("#### Examples of Valid Queries:")
                gr.Markdown(
                    "- *How many people waved?*\n"
                    "- *Find all standing activities in the last hour*\n"
                    "- *What activities happened most frequently?*\n"
                    "- *Show events with high confidence*"
                )
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="❓ Ask a Question",
                    placeholder="e.g., How many people were detected?",
                    lines=3,
                    scale=3
                )
                query_button = gr.Button("💬 Query", size="lg", variant="primary", scale=1)
            
            query_output = gr.Markdown(label="Query Results")
            
            query_button.click(
                fn=query_callback,
                inputs=query_input,
                outputs=query_output
            )
            
            # Behavior search
            gr.Markdown("---")
            gr.Markdown("#### Advanced: Semantic Behavior Search")
            
            with gr.Row():
                behavior_input = gr.Textbox(
                    label="🎯 Describe a Behavior",
                    placeholder="e.g., person walking and then standing",
                    lines=2,
                    scale=3
                )
                behavior_button = gr.Button("🔍 Search", size="lg", variant="primary", scale=1)
            
            behavior_output = gr.Markdown(label="Behavior Matches")
            
            behavior_button.click(
                fn=behavior_search_callback,
                inputs=behavior_input,
                outputs=behavior_output
            )
        
        
        # ========== TAB 4: ANALYTICS & STATUS ==========
        with gr.Tab("📊 Analytics & Status"):
            gr.Markdown("### System Analytics & Diagnostics")
            
            with gr.Row():
                analytics_button = gr.Button("📈 Refresh Analytics", size="lg", variant="primary", scale=1)
                status_button = gr.Button("🔧 Check System Status", size="lg", variant="secondary", scale=1)
                activities_button = gr.Button("📅 Activity Timeline", size="lg", variant="secondary", scale=1)
            
            with gr.Group():
                gr.Markdown("### Analytics Dashboard")
                analytics_output = gr.Markdown(label="Analytics Summary")
                
                analytics_button.click(
                    fn=get_analytics_callback,
                    outputs=analytics_output
                )
            
            with gr.Group():
                gr.Markdown("### Activity Timeline")
                activities_output = gr.Markdown(label="Activities")
                
                activities_button.click(
                    fn=get_activities_callback,
                    outputs=activities_output
                )
            
            with gr.Group():
                gr.Markdown("### System Status")
                status_output = gr.Markdown(label="System Info")
                
                status_button.click(
                    fn=check_system_callback,
                    outputs=status_output
                )


if __name__ == "__main__":
    print(f"🚀 Starting VIAS Frontend...")
    print(f"📡 Backend API: {API_BASE}")
    print(f"🌐 Gradio UI will be available at: http://127.0.0.1:7860")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
