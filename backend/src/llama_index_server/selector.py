from typing import Any, Dict, List, Sequence, cast

from llama_index.core.base.base_selector import (
    BaseSelector,
    MultiSelection,
    SingleSelection,
)
from llama_index.core.output_parsers.base import StructuredOutput
from llama_index.core.output_parsers.selection import Answer
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata


class CustomMultiSelector(BaseSelector):

    def __init__(self, prompt: str = None):
        self._prompt = prompt

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "prompt" in prompts:
            self._prompt = prompts["prompt"]

    def _select(self, choices: Sequence[ToolMetadata],
                query: QueryBundle) -> MultiSelection:
        question = query.query_str

        answers = []
        for i, choice in enumerate(choices):
            description = choice.description
            name = choice.name

            answers.append(
                Answer(
                    choice=i + 1,
                    reason=
                    f'The {question} is similar to the description of {name}: {description}'
                ))

        selections = [
            SingleSelection(index=answer.choice - 1, reason=answer.reason)
            for answer in answers
        ]

        return MultiSelection(selections=selections)

        # return 一定要有ind、inds、reason、reasons
        # 每個selections都是一個SingleSelection
        # 這邊可以自己寫一些規則，來判斷是否要選擇這個choice
        # 例如description跟question的相似度

        # 原本的作法
        '''
        # prepare input
        context_list = _build_choices_text(choices)
        max_outputs = self._max_outputs or len(choices)

        prediction = self._llm.predict(
            prompt=self._prompt,
            num_choices=len(choices),
            max_outputs=max_outputs,
            context_list=context_list,
            query_str=query.query_str,
        )

        assert self._prompt.output_parser is not None
        parsed = self._prompt.output_parser.parse(prediction)

        return _structured_output_to_selector_result(parsed)
        '''

    async def _aselect(self, choices: Sequence[ToolMetadata],
                       query: QueryBundle) -> MultiSelection:
        results = await self._select(choices, query)
        return results


def _structured_output_to_selector_result(output: Any) -> MultiSelection:
    """Convert structured output to selector result."""
    structured_output = cast(StructuredOutput, output)
    answers = cast(List[Answer], structured_output.parsed_output)

    # adjust for zero indexing
    selections = [
        SingleSelection(index=answer.choice - 1, reason=answer.reason)
        for answer in answers
    ]
    return MultiSelection(selections=selections)
