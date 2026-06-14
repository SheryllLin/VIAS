from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import requests


API_BASE = "http://127.0.0.1:8000"


def _post_file(endpoint: str, file_path: str, person_id: str | None = None) -> str:
    with open(file_path, "rb") as handle:
        files = {"file": (Path(file_path).name, handle)}
        data = {"person_id": person_id} if person_id else None
        response = requests.post(f"{API_BASE}{endpoint}", files=files, data=data, timeout=1200)
    response.raise_for_status()
    return json.dumps(response.json(), indent=2)


def _post_json(endpoint: str, payload: dict) -> str:
    response = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=300)
    response.raise_for_status()
    return json.dumps(response.json(), indent=2)


def _get(endpoint: str) -> str:
    response = requests.get(f"{API_BASE}{endpoint}", timeout=300)
    response.raise_for_status()
    return json.dumps(response.json(), indent=2)


with gr.Blocks(title="VIAS") as demo:
    gr.Markdown("# VIAS — Visual Intelligence & Analysis System")
    with gr.Tabs():
        with gr.Tab("Upload Video"):
            video_input = gr.Video(label="Input Video")
            upload_output = gr.Textbox(label="Processing Result", lines=16)
            upload_button = gr.Button("Process Video")
            upload_button.click(fn=lambda x: _post_file("/upload-video", x), inputs=video_input, outputs=upload_output)
        with gr.Tab("Upload Reference Image"):
            image_input = gr.Image(type="filepath", label="Reference Image")
            person_id_input = gr.Textbox(label="Person ID", value="reference-person")
            ref_output = gr.Textbox(label="Registration Result", lines=10)
            ref_button = gr.Button("Register Reference")
            ref_button.click(fn=lambda image_path, person_id: _post_file("/upload-reference-image", image_path, person_id), inputs=[image_input, person_id_input], outputs=ref_output)
        with gr.Tab("Search Person"):
            search_input = gr.Textbox(label="Person ID")
            search_output = gr.Textbox(label="Search Results", lines=12)
            search_button = gr.Button("Search")
            search_button.click(fn=lambda person_id: _post_json("/search-person", {"person_id": person_id, "top_k": 5}), inputs=search_input, outputs=search_output)
        with gr.Tab("Activity Timeline"):
            activity_output = gr.Textbox(label="Activities", lines=18)
            gr.Button("Refresh Activities").click(fn=lambda: _get("/activities"), outputs=activity_output)
        with gr.Tab("Event Explorer"):
            event_output = gr.Textbox(label="Events", lines=18)
            gr.Button("Refresh Events").click(fn=lambda: _get("/events"), outputs=event_output)
        with gr.Tab("Natural Language Query"):
            nlq_input = gr.Textbox(label="Ask a Query", lines=3)
            nlq_output = gr.Textbox(label="Answer", lines=16)
            gr.Button("Run Query").click(fn=lambda query: _post_json("/query", {"query": query}), inputs=nlq_input, outputs=nlq_output)
        with gr.Tab("Behavior Search"):
            behavior_input = gr.Textbox(label="Behavior Description", lines=3)
            behavior_output = gr.Textbox(label="Behavior Matches", lines=16)
            gr.Button("Search Behavior").click(fn=lambda query: _post_json("/behavior-search", {"query": query}), inputs=behavior_input, outputs=behavior_output)
        with gr.Tab("Analytics Dashboard"):
            dashboard_output = gr.Textbox(label="Analytics Summary", lines=16)
            gr.Button("Refresh Dashboard").click(fn=lambda: _get("/analytics"), outputs=dashboard_output)
        with gr.Tab("System Status"):
            datasets_output = gr.Textbox(label="Datasets", lines=12)
            models_output = gr.Textbox(label="Models", lines=12)
            gr.Button("Check Datasets").click(fn=lambda: _get("/datasets"), outputs=datasets_output)
            gr.Button("Check Models").click(fn=lambda: _get("/models/status"), outputs=models_output)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
