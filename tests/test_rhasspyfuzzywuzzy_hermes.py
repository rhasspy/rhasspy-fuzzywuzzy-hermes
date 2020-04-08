"""Unit tests for rhasspyfuzzwuzzy_hermes"""
import asyncio
import logging
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import rhasspyfuzzywuzzy
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
from rhasspynlu import intents_to_graph, parse_ini

from rhasspyfuzzywuzzy_hermes import NluHermesMqtt

_LOGGER = logging.getLogger(__name__)
_LOOP = asyncio.get_event_loop()


class RhasspyFuzzywuzzyHermesTestCase(unittest.TestCase):
    """Tests for rhasspyfuzzywuzzy_hermes"""

    def setUp(self):
        self.site_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())

        ini_text = """
        [SetLightColor]
        set the (bedroom | living room){name} light to (red | green | blue){color}

        [GetTime]
        what time is it
        """

        self.graph = intents_to_graph(parse_ini(ini_text))
        self.examples = rhasspyfuzzywuzzy.train(self.graph)
        self.client = MagicMock()
        self.hermes = NluHermesMqtt(
            self.client,
            self.graph,
            examples=self.examples,
            confidence_threshold=1.0,
            site_ids=[self.site_id],
        )

    # -------------------------------------------------------------------------

    async def async_test_handle_query(self):
        """Verify valid input leads to a query message."""
        query_id = str(uuid.uuid4())
        text = "set the bedroom light to red"

        query = NluQuery(
            input=text, id=query_id, site_id=self.site_id, session_id=self.session_id
        )

        results = []
        async for result in self.hermes.on_message(query):
            results.append(result)

        # Check results
        intent = Intent(intent_name="SetLightColor", confidence_score=1.0)
        slots = [
            Slot(
                entity="name",
                slot_name="name",
                value={"kind": "Unknown", "value": "bedroom"},
                raw_value="bedroom",
                confidence=1.0,
                range=SlotRange(start=8, end=15, raw_start=8, raw_end=15),
            ),
            Slot(
                entity="color",
                slot_name="color",
                value={"kind": "Unknown", "value": "red"},
                raw_value="red",
                confidence=1.0,
                range=SlotRange(start=25, end=28, raw_start=25, raw_end=28),
            ),
        ]

        self.assertEqual(
            results,
            [
                NluIntentParsed(
                    input=text,
                    id=query_id,
                    site_id=self.site_id,
                    session_id=self.session_id,
                    intent=intent,
                    slots=slots,
                ),
                (
                    NluIntent(
                        input=text,
                        id=query_id,
                        site_id=self.site_id,
                        session_id=self.session_id,
                        intent=intent,
                        slots=slots,
                        asr_tokens=[NluIntent.make_asr_tokens(text.split())],
                        raw_input=text,
                    ),
                    {"intent_name": intent.intent_name},
                ),
            ],
        )

    def test_handle_query(self):
        """Call async_test_handle_query."""
        _LOOP.run_until_complete(self.async_test_handle_query())

    # -------------------------------------------------------------------------

    async def async_test_intent_filter(self):
        """Verify intent filter works."""
        query_id = str(uuid.uuid4())
        text = "what time is it"

        query = NluQuery(
            input=text, id=query_id, site_id=self.site_id, session_id=self.session_id
        )

        # Query should succeed
        results = []
        async for result in self.hermes.on_message(query):
            results.append(result)

        # Check results
        self.assertEqual(len(results), 2)

        # Ignore intentParsed
        nlu_intent = results[1][0]
        self.assertIsInstance(nlu_intent, NluIntent)
        self.assertEqual(nlu_intent.intent.intent_name, "GetTime")

        # Add intent filter
        query_id = str(uuid.uuid4())
        query = NluQuery(
            input=text,
            id=query_id,
            intent_filter=["SetLightColor"],
            site_id=self.site_id,
            session_id=self.session_id,
        )

        # Query should fail
        results = []
        async for result in self.hermes.on_message(query):
            results.append(result)

        # Check results
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], NluIntentNotRecognized)

    def test_intent_filter(self):
        """Call async_test_intent_filter."""
        _LOOP.run_until_complete(self.async_test_intent_filter())

    # -------------------------------------------------------------------------

    async def async_test_not_recognized(self):
        """Verify invalid input leads to recognition failure."""
        query_id = str(uuid.uuid4())
        text = "not a valid sentence at all"

        query = NluQuery(
            input=text, id=query_id, site_id=self.site_id, session_id=self.session_id
        )

        results = []
        async for result in self.hermes.on_message(query):
            results.append(result)

        # Check results
        self.assertEqual(
            results,
            [
                NluIntentNotRecognized(
                    input=text,
                    id=query_id,
                    site_id=self.site_id,
                    session_id=self.session_id,
                )
            ],
        )

    def test_not_recognized(self):
        """Call async_test_not_recognized."""
        _LOOP.run_until_complete(self.async_test_not_recognized())

    # -------------------------------------------------------------------------

    async def async_test_train_success(self):
        """Verify successful training."""
        train_id = str(uuid.uuid4())

        def fake_read_graph(*args, **kwargs):
            return MagicMock()

        def fake_train(*args, **kwargs):
            return MagicMock()

        # Create temporary file for "open"
        with tempfile.NamedTemporaryFile(mode="wb+", suffix=".gz") as graph_file:
            train = NluTrain(id=train_id, graph_path=graph_file.name)

            # Ensure fake graph "loads" and training goes through
            with patch("rhasspynlu.gzip_pickle_to_graph", new=fake_read_graph):
                with patch("rhasspyfuzzywuzzy.train", new=fake_train):
                    results = []
                    async for result in self.hermes.on_message(
                        train, site_id=self.site_id
                    ):
                        results.append(result)

            self.assertEqual(
                results, [(NluTrainSuccess(id=train_id), {"site_id": self.site_id})]
            )

    def test_train_success(self):
        """Call async_test_train_success."""
        _LOOP.run_until_complete(self.async_test_train_success())

    # -------------------------------------------------------------------------

    async def async_test_train_error(self):
        """Verify training error."""
        train = NluTrain(id=self.session_id, graph_path=Path("fake-graph.pickle.gz"))

        # Allow failed attempt to access missing graph
        results = []
        async for result in self.hermes.on_message(train, site_id=self.site_id):
            results.append(result)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, NluError)
        self.assertEqual(result.site_id, self.site_id)
        self.assertEqual(result.session_id, self.session_id)

    def test_train_error(self):
        """Call async_test_train_error."""
        _LOOP.run_until_complete(self.async_test_train_error())
