"""Hermes MQTT server for Rhasspy fuzzywuzzy"""
import asyncio
import json
import logging
import typing
from pathlib import Path

import networkx as nx
import rhasspyfuzzywuzzy
import rhasspynlu
from rhasspyfuzzywuzzy.const import ExamplesType
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluIntentParsed,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from rhasspynlu.jsgf import Sentence

_LOGGER = logging.getLogger("rhasspyfuzzywuzzy_hermes")

# -----------------------------------------------------------------------------


class NluHermesMqtt(HermesClient):
    """Hermes MQTT server for Rhasspy fuzzywuzzy."""

    def __init__(
        self,
        client,
        intent_graph: typing.Optional[nx.DiGraph] = None,
        intent_graph_path: typing.Optional[Path] = None,
        examples: typing.Optional[ExamplesType] = None,
        examples_path: typing.Optional[Path] = None,
        sentences: typing.Optional[typing.List[Path]] = None,
        default_entities: typing.Dict[str, typing.Iterable[Sentence]] = None,
        word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        replace_numbers: bool = False,
        language: typing.Optional[str] = None,
        confidence_threshold: float = 0.0,
        siteIds: typing.Optional[typing.List[str]] = None,
        loop=None,
    ):
        super().__init__("rhasspyfuzzywuzzy_hermes", client, siteIds=siteIds, loop=loop)

        self.subscribe(NluQuery, NluTrain)

        # Intent graph
        self.intent_graph = intent_graph
        self.intent_graph_path = intent_graph_path

        # Examples
        self.examples = examples
        self.examples_path = examples_path

        self.sentences = sentences or []
        self.default_entities = default_entities or {}
        self.word_transform = word_transform

        self.replace_numbers = replace_numbers
        self.language = language

        # Minimum confidence before not recognized
        self.confidence_threshold = confidence_threshold

        # Event loop
        self.loop = loop or asyncio.get_event_loop()

    # -------------------------------------------------------------------------

    async def handle_query(
        self, query: NluQuery
    ) -> typing.AsyncIterable[
        typing.Union[
            NluIntentParsed,
            typing.Tuple[NluIntent, TopicArgs],
            NluIntentNotRecognized,
            NluError,
        ]
    ]:
        """Do intent recognition."""
        # Check intent graph
        if (
            not self.intent_graph
            and self.intent_graph_path
            and self.intent_graph_path.is_file()
        ):
            _LOGGER.debug("Loading %s", self.intent_graph_path)
            with open(self.intent_graph_path, mode="rb") as graph_file:
                self.intent_graph = rhasspynlu.gzip_pickle_to_graph(graph_file)

        # Check examples
        if not self.examples and self.examples_path and self.examples_path.is_file():
            # Load examples from file
            _LOGGER.debug("Loading examples from %s", str(self.examples_path))
            with open(self.examples_path, "r") as examples_file:
                self.examples = json.load(examples_file)

        if self.intent_graph and self.examples:

            def intent_filter(intent_name: str) -> bool:
                """Filter out intents."""
                if query.intentFilter:
                    return intent_name in query.intentFilter
                return True

            original_text = query.input

            # Replace digits with words
            if self.replace_numbers:
                # Have to assume whitespace tokenization
                words = rhasspynlu.replace_numbers(query.input.split(), self.language)
                query.input = " ".join(words)

            input_text = query.input

            # Fix casing
            if self.word_transform:
                input_text = self.word_transform(input_text)

            recognitions: typing.List[rhasspynlu.Recognition] = []

            if input_text:
                recognitions = rhasspyfuzzywuzzy.recognize(
                    input_text,
                    self.intent_graph,
                    self.examples,
                    intent_filter=intent_filter,
                )
        else:
            _LOGGER.error("No intent graph or examples loaded")
            recognitions = []

        # Use first recognition only if above threshold
        if (
            recognitions
            and recognitions[0]
            and recognitions[0].intent
            and (recognitions[0].intent.confidence >= self.confidence_threshold)
        ):
            recognition = recognitions[0]
            assert recognition.intent
            intent = Intent(
                intentName=recognition.intent.name,
                confidenceScore=recognition.intent.confidence,
            )
            slots = [
                Slot(
                    entity=e.entity,
                    slotName=e.entity,
                    confidence=1,
                    value=e.value,
                    raw_value=e.raw_value,
                    range=SlotRange(
                        start=e.start,
                        end=e.end,
                        raw_start=e.raw_start,
                        raw_end=e.raw_end,
                    ),
                )
                for e in recognition.entities
            ]

            # intentParsed
            yield NluIntentParsed(
                input=input_text,
                id=query.id,
                siteId=query.siteId,
                sessionId=query.sessionId,
                intent=intent,
                slots=slots,
            )

            # intent
            yield (
                NluIntent(
                    input=input_text,
                    id=query.id,
                    siteId=query.siteId,
                    sessionId=query.sessionId,
                    intent=intent,
                    slots=slots,
                    asrTokens=input_text.split(),
                    rawAsrTokens=original_text.split(),
                    wakewordId=query.wakewordId,
                ),
                {"intentName": recognition.intent.name},
            )
        else:
            # Not recognized
            yield NluIntentNotRecognized(
                input=query.input,
                id=query.id,
                siteId=query.siteId,
                sessionId=query.sessionId,
            )

    # -------------------------------------------------------------------------

    async def handle_train(
        self, train: NluTrain, siteId: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[NluTrainSuccess, TopicArgs], NluError]
    ]:
        """Transform sentences to intent examples"""
        try:
            _LOGGER.debug("Loading %s", train.graph_path)
            with open(train.graph_path, mode="rb") as graph_file:
                self.intent_graph = rhasspynlu.gzip_pickle_to_graph(graph_file)

            self.examples = rhasspyfuzzywuzzy.train(self.intent_graph)

            if self.examples_path:
                # Write examples to JSON file
                with open(self.examples_path, "w") as examples_file:
                    json.dump(self.examples, examples_file)

                _LOGGER.debug("Wrote %s", str(self.examples_path))

            yield (NluTrainSuccess(id=train.id), {"siteId": siteId})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield NluError(
                siteId=siteId, sessionId=train.id, error=str(e), context=train.id
            )

    # -------------------------------------------------------------------------

    async def on_message(
        self,
        message: Message,
        siteId: typing.Optional[str] = None,
        sessionId: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        if isinstance(message, NluQuery):
            async for query_result in self.handle_query(message):
                yield query_result
        elif isinstance(message, NluTrain):
            assert siteId, "Missing siteId"
            async for train_result in self.handle_train(message, siteId=siteId):
                yield train_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)
