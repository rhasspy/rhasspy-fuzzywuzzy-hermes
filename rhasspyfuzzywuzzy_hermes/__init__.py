"""Hermes MQTT server for Rhasspy fuzzywuzzy"""
import json
import logging
import typing
from pathlib import Path

import attr
import networkx as nx
import rhasspyfuzzywuzzy
import rhasspynlu
from rhasspyfuzzywuzzy.const import ExamplesType
from rhasspyhermes.base import Message
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from rhasspynlu.jsgf import Sentence

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class NluHermesMqtt:
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
        language: str = "en",
        siteIds: typing.Optional[typing.List[str]] = None,
    ):
        self.client = client

        # Intent graph
        self.intent_graph = intent_graph
        self.intent_graph_path = intent_graph_path

        # Examples
        self.examples = examples
        self.examples_path = examples_path

        self.sentences = sentences or []
        self.default_entities = default_entities or {}
        self.word_transform = word_transform

        self.siteIds = siteIds or []
        self.language = language

    # -------------------------------------------------------------------------

    def handle_query(self, query: NluQuery):
        """Do intent recognition."""
        # Check intent graph
        if (
            not self.intent_graph
            and self.intent_graph_path
            and self.intent_graph_path.is_file()
        ):
            # Load intent graph from file
            _LOGGER.debug("Loading graph from %s", str(self.intent_graph_path))
            with open(self.intent_graph_path, "r") as intent_graph_file:
                graph_dict = json.load(intent_graph_file)
                self.intent_graph = rhasspynlu.json_to_graph(graph_dict)

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

            input_text = query.input

            # Fix casing
            if self.word_transform:
                input_text = self.word_transform(input_text)

            recognitions = rhasspyfuzzywuzzy.recognize(
                input_text,
                self.intent_graph,
                self.examples,
                intent_filter=intent_filter,
                language=self.language,
            )
        else:
            _LOGGER.error("No intent graph or examples loaded")
            recognitions = []

        if recognitions:
            # Use first recognition only.
            recognition = recognitions[0]
            assert recognition is not None
            assert recognition.intent is not None

            self.publish(
                NluIntent(
                    input=query.input,
                    id=query.id,
                    siteId=query.siteId,
                    sessionId=query.sessionId,
                    intent=Intent(
                        intentName=recognition.intent.name,
                        confidenceScore=recognition.intent.confidence,
                    ),
                    slots=[
                        Slot(
                            entity=e.entity,
                            slotName=e.entity,
                            confidence=1,
                            value=e.value,
                            raw_value=e.raw_value,
                            range=SlotRange(start=e.raw_start, end=e.raw_end),
                        )
                        for e in recognition.entities
                    ],
                ),
                intentName=recognition.intent.name,
            )
        else:
            # Not recognized
            self.publish(
                NluIntentNotRecognized(
                    input=query.input,
                    id=query.id,
                    siteId=query.siteId,
                    sessionId=query.sessionId,
                )
            )

    # -------------------------------------------------------------------------

    def handle_train(
        self, message: NluTrain, siteId: str = "default"
    ) -> typing.Union[NluTrainSuccess, NluError]:
        """Transform sentences to intent examples"""
        _LOGGER.debug("<- %s(%s)", message.__class__.__name__, message.id)

        try:
            self.examples = rhasspyfuzzywuzzy.train(message.graph_dict)

            if self.examples_path:
                # Write examples to JSON file
                with open(self.examples_path, "w") as examples_file:
                    json.dump(self.examples, examples_file)

                _LOGGER.debug("Wrote %s", str(self.examples_path))

            return NluTrainSuccess(id=message.id)
        except Exception as e:
            _LOGGER.exception("handle_train")
            return NluError(siteId=siteId, error=str(e), context=message.id)

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [NluQuery.topic()]

            if self.siteIds:
                # Specific siteIds
                topics.extend(
                    [NluTrain.topic(siteId=siteId) for siteId in self.siteIds]
                )
            else:
                # All siteIds
                topics.append(NluTrain.topic(siteId="+"))

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)
            if msg.topic == NluQuery.topic():
                json_payload = json.loads(msg.payload)

                # Check siteId
                if not self._check_siteId(json_payload):
                    return

                try:
                    query = NluQuery(**json_payload)
                    _LOGGER.debug("<- %s", query)
                    self.handle_query(query)
                except Exception as e:
                    _LOGGER.exception("nlu query")
                    self.publish(
                        NluError(
                            siteId=query.siteId,
                            sessionId=json_payload.get("sessionId", ""),
                            error=str(e),
                            context="",
                        )
                    )
            elif NluTrain.is_topic(msg.topic):
                siteId = NluTrain.get_siteId(msg.topic)
                if self.siteIds and (siteId not in self.siteIds):
                    return

                json_payload = json.loads(msg.payload)
                train = NluTrain(**json_payload)
                result = self.handle_train(train)
                self.publish(result)
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            _LOGGER.debug("-> %s", message)
            topic = message.topic(**topic_args)
            payload = json.dumps(attr.asdict(message))
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True
