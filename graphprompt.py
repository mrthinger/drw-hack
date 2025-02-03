from typing import TypedDict
import json
import os
import urllib.request


class Statement(TypedDict):
    what: str
    citations: list[str]
    how: str


class AnalysisResponse(TypedDict):
    month: int
    day: int
    year: int
    observations: list[Statement]


class VicariousAmaranthTickInputs(TypedDict):
    document: str


def gprompt_analyze(inputs: VicariousAmaranthTickInputs) -> AnalysisResponse:
    api_key = os.environ.get("GRAPHPROMPT_API_KEY")
    if not api_key:
        raise ValueError("GRAPHPROMPT_API_KEY environment variable not set")

    url = "https://api.graphprompt.ai/vicarious-amaranth-tick"
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }
    data = json.dumps(inputs).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        if response.status != 200:
            raise Exception(
                f"GraphPrompt API request failed with status {response.status}"
            )

        response_data = response.read().decode("utf-8")
        parsed_response_data = json.loads(response_data)
        res = parsed_response_data["result"]
        return json.loads(res)


class ReasonInputs(TypedDict):
    chunk: str


def gprompt_reason(inputs: ReasonInputs) -> str:
    api_key = os.environ.get("GRAPHPROMPT_API_KEY")
    if not api_key:
        raise ValueError("GRAPHPROMPT_API_KEY environment variable not set")

    url = "https://api.graphprompt.ai/reason"
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }
    data = json.dumps(inputs).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        if response.status != 200:
            raise Exception(
                f"GraphPrompt API request failed with status {response.status}"
            )

        response_data = response.read().decode("utf-8")
        parsed_response_data = json.loads(response_data)
        return parsed_response_data["result"]


class ReasonTopInputs(TypedDict):
    analysis: str


def gprompt_reason_top(inputs: ReasonTopInputs) -> str:
    api_key = os.environ.get("GRAPHPROMPT_API_KEY")
    if not api_key:
        raise ValueError("GRAPHPROMPT_API_KEY environment variable not set")

    url = "https://api.graphprompt.ai/reason-top"
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key,
    }
    data = json.dumps(inputs).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        if response.status != 200:
            raise Exception(
                f"GraphPrompt API request failed with status {response.status}"
            )

        response_data = response.read().decode("utf-8")
        parsed_response_data = json.loads(response_data)
        return parsed_response_data["result"]
