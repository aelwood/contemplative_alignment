## NOTE - just a copy and paste of chatGPT output, needs testing and running

import os
import subprocess
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from pydantic import BaseModel
import openai

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Define the OpenAI SUT Request and Response structures
class OpenAIRequest(BaseModel):
    text: str

class OpenAIResponse(BaseModel):
    text: str

# --- Define Different SUT Implementations ---
@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class OpenAIBaselineSUT(PromptResponseSUT[OpenAIRequest, OpenAIResponse]):
    """Baseline SUT that sends a prompt to OpenAI API without modifications."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIRequest:
        return OpenAIRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIRequest:
        return OpenAIRequest(text=format_chat(prompt))

    def evaluate(self, request: OpenAIRequest) -> OpenAIResponse:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request.text}],
            temperature=0.7,
            max_tokens=300
        )
        return OpenAIResponse(text=response.choices[0].message.content)

    def translate_response(self, request: OpenAIRequest, response: OpenAIResponse) -> SUTResponse:
        return SUTResponse(text=response.text)

SUTS.register(OpenAIBaselineSUT, "openai_baseline")

@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class OpenAIReflectiveSUT(PromptResponseSUT[OpenAIRequest, OpenAIResponse]):
    """SUT with Self-Reflective Prior Relaxation Prompting."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIRequest:
        return OpenAIRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIRequest:
        return OpenAIRequest(text=format_chat(prompt))

    def evaluate(self, request: OpenAIRequest) -> OpenAIResponse:
        # Step 1: Reflect on the request
        reflection_prompt = f"Think carefully: What are potential biases or safety concerns in the following prompt? {request.text}"
        reflection_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": reflection_prompt}],
            temperature=0.2,
            max_tokens=100
        )
        reflection = reflection_response.choices[0].message.content

        # Step 2: Answer the question with self-reflection included
        final_prompt = f"Consider this reflection: {reflection}\nNow answer thoughtfully: {request.text}"
        final_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return OpenAIResponse(text=final_response.choices[0].message.content)

    def translate_response(self, request: OpenAIRequest, response: OpenAIResponse) -> SUTResponse:
        return SUTResponse(text=response.text)

SUTS.register(OpenAIReflectiveSUT, "openai_reflective")

@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class OpenAITemperatureSUT(PromptResponseSUT[OpenAIRequest, OpenAIResponse]):
    """SUT that modifies the temperature setting to test diversity vs. determinism."""

    def __init__(self, temperature=0.3):
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def translate_text_prompt(self, prompt: TextPrompt) -> OpenAIRequest:
        return OpenAIRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> OpenAIRequest:
        return OpenAIRequest(text=format_chat(prompt))

    def evaluate(self, request: OpenAIRequest) -> OpenAIResponse:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request.text}],
            temperature=self.temperature,  # Uses specified temperature
            max_tokens=300
        )
        return OpenAIResponse(text=response.choices[0].message.content)

    def translate_response(self, request: OpenAIRequest, response: OpenAIResponse) -> SUTResponse:
        return SUTResponse(text=response.text)

SUTS.register(OpenAITemperatureSUT(0.3), "openai_temp_low")
SUTS.register(OpenAITemperatureSUT(1.0), "openai_temp_high")

# --- Save the Script ---
script_path = os.path.expanduser("~/modelbench_plugins/openai_suts.py")
with open(script_path, "w") as file:
    file.write(__file__)

# --- Run ModelBench Benchmark ---
print("\n🚀 Running ModelBench with Custom SUTs...\n")
subprocess.run([
    "modelbench", "benchmark", "-m", "10",
    "--plugin-dir", os.path.expanduser("~/modelbench_plugins/"),
    "--sut", "openai_baseline",
    "--sut", "openai_reflective",
    "--sut", "openai_temp_low",
    "--sut", "openai_temp_high"
])

print("\n✅ ModelBench test completed. Open 'web/index.html' to view results.")
