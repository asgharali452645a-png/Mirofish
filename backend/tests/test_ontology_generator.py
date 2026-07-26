import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.ontology_generator import OntologyGenerator
from app.services.graph_builder import GraphBuilderService


class DummyLLMClient:
    def chat_json(self, *args, **kwargs):
        raise RuntimeError("simulated LLM failure")


class OntologyGeneratorFallbackTest(unittest.TestCase):
    def test_generate_returns_default_ontology_when_llm_fails(self):
        generator = OntologyGenerator(llm_client=DummyLLMClient())
        ontology = generator.generate(
            document_texts=["Sample document"],
            simulation_requirement="Generate a simulation ontology",
            additional_context=None,
        )

        self.assertIn("entity_types", ontology)
        self.assertIn("edge_types", ontology)
        self.assertIn("analysis_summary", ontology)
        self.assertGreaterEqual(len(ontology["entity_types"]), 10)
        self.assertGreaterEqual(len(ontology["edge_types"]), 6)

    def test_graph_builder_falls_back_when_zep_key_is_placeholder(self):
        service = GraphBuilderService(api_key="dummy")
        graph_id = service.create_graph("test-graph")

        self.assertTrue(graph_id.startswith("local_"))
        graph_data = service.get_graph_data(graph_id)
        self.assertGreaterEqual(graph_data["node_count"], 1)
        self.assertGreaterEqual(graph_data["edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
