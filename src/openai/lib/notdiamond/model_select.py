import copy
import json

import httpx
import ppdeep


def model_select(messages: list, llm_providers: list, api_key: str):
    url = "https://not-diamond-server.onrender.com/v2/optimizer/modelSelect"

    payload = {
        "messages": transform_messages(copy.deepcopy(messages)),
        "hash_digest": hash_digest(messages),
        "llm_providers": transformed_providers(llm_providers),
        "metric": "accuracy",
        "max_model_depth": len(llm_providers),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = httpx.post(url, data=json.dumps(payload), headers=headers)
    except Exception as e:
        print(f"ND API error for modelSelect: {e}")
        return None

    if response.status_code == 200:
        response_json = response.json()

        result_providers = response_json["providers"]
        top_provider = result_providers[0]

        return top_provider["model"]
    else:
        if response.status_code == 401:
            print(f"ND API error: {response.status_code}. Make sure you have a valid NotDiamond API Key.")
        else:
            print(f"ND API error: {response.status_code}")
        return None


def transform_messages(messages: list):
    hashed_messages = []
    for role, content in messages:
        hashed_messages.append({"role": role, "content": nd_hash(content)})

    return hashed_messages


def hash_digest(messages):
    message_to_hash = ""
    if messages[0][0] == "system":
        message_to_hash += messages[0][1] + "\n"

    if messages[-1][0] == "user":
        message_to_hash += messages[-1][1]

    return nd_hash(message_to_hash)


def transformed_providers(llm_providers: list):
    result = []
    for v in llm_providers:
        splits = v.split("/")
        result.append({"provider": splits[0], "model": splits[1]})
    return result


def nd_hash(s: str) -> str:
    """
    Source of library from: https://github.com/elceef/ppdeep
    """
    return ppdeep.hash(s)
