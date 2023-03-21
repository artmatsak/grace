import re
import json
import logging
from openai_chatbot import OpenAIChatbot
from router import Router
from knowledge_base import KnowledgeBase
from typing import Dict, Callable


class GRACEChatbot(OpenAIChatbot):
    INITIAL_PROMPT_TEMPLATE = """You are an AI assistant for {business_name}, {business_description}. You process customers' requests as follows:

1. Greet the customer and ask how you can be of help.
2. Identify the customer's request and the backend command to process it.
3. Ensure you have the values for all of the parameters required by the backend command. Collect from the customer any values you don't have. Do not collect information that is not required. No values are available to you except for those provided by the customer. If the customer cannot provide you with a value, you refuse to process their request.
4. Ask the customer to hold on and then process their request by sending a command JSON to the backend in the following format:

All right, let me look into this for you. [json]{command_example_json}[/json]
Backend response: {command_example_result}

5. Communicate the execution result back to the customer and ask if there's anything else you can do for them.
6. If there's nothing else, say goodbye and output "END".

Only the following commands are available to you. If the customer's request is not among the provided commands, you refuse to process it:

{commands_string}

You can use the look_up command to look up answers to questions related to {business_name}. For example:

Do you have parking on site?
[json]{{"command": "look_up", "params": {{"question": "Do you have parking on site?"}}}}[/json]
Backend response: On-site parking is available
Yes, we do offer on-site parking. Is there anything else I can help you with?

To execute a backend command, you use the provided [json][/json] syntax. You never use computer code in your responses.

You use all dates exactly as provided by the customer, without rephrasing or converting them. {extra_instructions}

You start the conversation by outputting your first utterance."""

    def __init__(
        self,
        openai,
        backend: Router,
        domain: Dict[str, str],
        output_callback: Callable[[str], None],
        openai_engine: str = "gpt-3.5-turbo"
    ):
        self.knowledge_base = KnowledgeBase(domain["answers"])

        @backend.command(desc="look up a question", example_params=("What are your opening hours?",))
        def look_up(question: str) -> str:
            answer, score = self.knowledge_base.look_up(question)
            logging.debug(f"Knowledge base lookup score: {score}")
            return answer if score > 0.4 else "Cannot answer the question"

        command_example = domain["command_example"]
        command_example_json = json.dumps({
            "command": command_example["command"],
            "params": command_example["params"]
        })

        commands_string = "\n".join([f'- {c["python_sig"]} - {c["desc"]}. Example JSON: [json]{c["example_json"]}[/json]'
                                     for c in backend.registry.values()])

        initial_prompt = self.INITIAL_PROMPT_TEMPLATE.format(
            **domain,
            command_example_json=command_example_json,
            command_example_result=command_example["result"],
            commands_string=commands_string
        )

        super().__init__(openai=openai,
                         initial_prompt=initial_prompt,
                         output_callback=output_callback,
                         openai_engine=openai_engine)

        self.backend = backend
        self.domain = domain

    def _get_all_utterances(self):
        utterance = self._get_next_utterance()

        m = re.match(r"(.*?)($|\[json\](.*?)\[/json\])",
                     utterance, re.IGNORECASE | re.DOTALL)
        utterance = m[1].strip()
        command_json = m[3]

        if utterance:
            self.output_callback(utterance)

        if self.messages:
            self.messages.append({"role": self.ROLE_ASSISTANT, "content": m[0]})

        if command_json:
            logging.debug(f"Invoking backend command: {repr(command_json)}")

            try:
                result = self.backend.invoke(command_json)
                logging.debug(f"Got backend response: {repr(result)}")
            except Exception as e:
                result = str(e)
                logging.error(e)

            if self.messages:
                self._add_response(self.ROLE_SYSTEM, f"Backend response: {result}")
                self._get_all_utterances()
