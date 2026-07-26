import os
import sys
sys.path.insert(0, os.getcwd())
from backend.app.services.ontology_generator import OntologyGenerator

class Dummy:
    def chat_json(self, *args, **kwargs):
        raise RuntimeError('simulated failure')

og = OntologyGenerator(llm_client=Dummy())
result = og.generate(['sample text'], 'simulate a public event')
print(result['analysis_summary'])
print(len(result['entity_types']), len(result['edge_types']))
