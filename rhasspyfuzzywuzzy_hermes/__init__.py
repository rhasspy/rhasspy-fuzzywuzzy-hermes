"""Hermes MQTT server for Rhasspy fuzzywuzzy"""
import json
import logging
import sqlite3
import typing
from pathlib import Path

import networkx as nx
import rhasspynlu
from rhasspynlu.jsgf import Sentence

import rhasspyfuzzywuzzy
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

_LOGGER = logging.getLogger("rhasspyfuzzywuzzy_hermes")

# -----------------------------------------------------------------------------


class NluHermesMqtt(HermesClient):
    """Hermes MQTT server for Rhasspy fuzzywuzzy."""

    def __init__(
        self,
        client,
        intent_graph: typing.Optional[nx.DiGraph] = None,
        intent_graph_path: typing.Optional[Path] = None,
        examples_path: typing.Optional[Path] = None,
        sentences: typing.Optional[typing.List[Path]] = None,
        default_entities: typing.Dict[str, typing.Iterable[Sentence]] = None,
        word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        replace_numbers: bool = False,
        language: typing.Optional[str] = None,
        confidence_threshold: float = 0.0,
        site_ids: typing.Optional[typing.List[str]] = None,
    ):
        super().__init__("rhasspyfuzzywuzzy_hermes", client, site_ids=site_ids)

        self.subscribe(NluQuery, NluTrain)

        # Intent graph
        self.intent_graph = intent_graph
        self.intent_graph_path = intent_graph_path

        # Examples
        self.examples_path = examples_path

        self.sentences = sentences or []
        self.default_entities = default_entities or {}
        self.word_transform = word_transform

        self.replace_numbers = replace_numbers
        self.language = language

        # Minimum confidence before not recognized
        self.confidence_threshold = confidence_threshold

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
        try:
            if (
                not self.intent_graph
                and self.intent_graph_path
                and self.intent_graph_path.is_file()
            ):
                _LOGGER.debug("Loading %s", self.intent_graph_path)
                with open(self.intent_graph_path, mode="rb") as graph_file:
                    self.intent_graph = rhasspynlu.gzip_pickle_to_graph(graph_file)

            # Check examples
            if (
                self.intent_graph
                and self.examples_path
                and self.examples_path.is_file()
            ):

                def intent_filter(intent_name: str) -> bool:
                    """Filter out intents."""
                    if query.intent_filter:
                        return intent_name in query.intent_filter
                    return True

                original_text = query.input

                # Replace digits with words
                if self.replace_numbers:
                    # Have to assume whitespace tokenization
                    words = rhasspynlu.replace_numbers(
                        query.input.split(), self.language
                    )
                    query.input = " ".join(words)

                input_text = query.input

                # Fix casing
                if self.word_transform:
                    input_text = self.word_transform(input_text)

                recognitions: typing.List[rhasspynlu.intent.Recognition] = []

                if input_text:
                    recognitions = rhasspyfuzzywuzzy.recognize(
                        input_text,
                        self.intent_graph,
                        str(self.examples_path),
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
                    intent_name=recognition.intent.name,
                    confidence_score=recognition.intent.confidence,
                )
                slots = [
                    Slot(
                        entity=(e.source or e.entity),
                        slot_name=e.entity,
                        confidence=1.0,
                        value=e.value_dict,
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
                    input=recognition.text,
                    id=query.id,
                    site_id=query.site_id,
                    session_id=query.session_id,
                    intent=intent,
                    slots=slots,
                )

                # intent
                yield (
                    NluIntent(
                        input=recognition.text,
                        id=query.id,
                        site_id=query.site_id,
                        session_id=query.session_id,
                        intent=intent,
                        slots=slots,
                        asr_tokens=[NluIntent.make_asr_tokens(recognition.tokens)],
                        raw_input=original_text,
                        wakeword_id=query.wakeword_id,
                        lang=query.lang,
                    ),
                    {"intent_name": recognition.intent.name},
                )
            else:
                # Not recognized
                yield NluIntentNotRecognized(
                    input=query.input,
                    id=query.id,
                    site_id=query.site_id,
                    session_id=query.session_id,
                )
        except Exception as e:
            _LOGGER.exception("handle_query")
            yield NluError(
                site_id=query.site_id,
                session_id=query.session_id,
                error=str(e),
                context=original_text,
            )

    # -------------------------------------------------------------------------

    async def handle_train(
        self, train: NluTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[NluTrainSuccess, TopicArgs], NluError]
    ]:
        """Transform sentences to intent examples"""
        try:
            _LOGGER.debug("Loading %s", train.graph_path)
            with open(train.graph_path, mode="rb") as graph_file:
                self.intent_graph = rhasspynlu.gzip_pickle_to_graph(graph_file)

            examples = rhasspyfuzzywuzzy.train(self.intent_graph)

            if self.examples_path:
                if self.examples_path.is_file():
                    # Delete existing file
                    self.examples_path.unlink()

                # Write examples to SQLite database
                conn = sqlite3.connect(str(self.examples_path))
                c = conn.cursor()
                c.execute("""DROP TABLE IF EXISTS intents""")
                c.execute("""CREATE TABLE intents (sentence text, path text)""")

                for _, sentences in examples.items():
                    for sentence, path in sentences.items():
                        c.execute(
                            "INSERT INTO intents VALUES (?, ?)",
                            (sentence, json.dumps(path, ensure_ascii=False)),
                        )

                conn.commit()
                conn.close()

                _LOGGER.debug("Wrote %s", str(self.examples_path))
            yield (NluTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield NluError(
                site_id=site_id, session_id=train.id, error=str(e), context=train.id
            )

    # -------------------------------------------------------------------------

    async def on_message(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        if isinstance(message, NluQuery):
            async for query_result in self.handle_query(message):
                yield query_result
        elif isinstance(message, NluTrain):
            assert site_id, "Missing site_id"
            async for train_result in self.handle_train(message, site_id=site_id):
                yield train_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)
