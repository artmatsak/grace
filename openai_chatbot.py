import string
import logging
from typing import Tuple, Callable, List


class OpenAIChatbot():
    ROLE_SYSTEM = "system"
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"

    def __init__(
        self,
        openai,
        initial_prompt: str,
        output_callback: Callable[[str], None],
        end_token: str = "END",
        openai_engine: str = "gpt-3.5-turbo"
    ):
        self.openai = openai
        self.initial_prompt = initial_prompt
        self.output_callback = output_callback
        self.end_token = end_token
        self.openai_engine = openai_engine

        self.messages = []

    def start_session(self):
        self.messages = [
            {
                "role": self.ROLE_SYSTEM,
                "content": self.initial_prompt
            }
        ]
        logging.debug(f"Starting chatbot session with messages: {self.messages}")
        self._get_all_utterances()

    def send_responses(self, responses: List[str]):
        if not self.messages:
            raise RuntimeError("Chatbot session is not active")

        for response in responses:
            self._add_response(self.ROLE_USER, response.strip())

        self._get_all_utterances()

    def session_ended(self) -> bool:
        return not self.messages

    def _add_response(self, role: str, response: str):
        message = {"role": role, "content": response}
        logging.debug(f"Adding response: {repr(message)}")
        self.messages.append(message)

    def _get_all_utterances(self):
        utterance = self._get_next_utterance()

        if utterance:
            self.output_callback(utterance)

        if self.messages:
            self.messages.append({"role": self.ROLE_ASSISTANT, "content": utterance})

    def _get_next_utterance(self) -> str:
        completion = self.openai.ChatCompletion.create(
            model=self.openai_engine,
            messages=self.messages,
            max_tokens=150,
            temperature=0.9
        )

        utterance = completion['choices'][0]['message']['content'].strip(
            string.whitespace + '"')
        logging.debug(f"Got utterance: {repr(utterance)}")

        end_token_pos = utterance.find(self.end_token)
        if end_token_pos != -1:
            utterance = utterance[:end_token_pos].strip()
            # Ending the session
            self.messages = []

        return utterance
