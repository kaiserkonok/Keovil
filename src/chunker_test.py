from src import intelligent_rag_chunker
import json

sample = open('/home/kaiserkonok/computer_programming/K_RAG/test_data/test.txt', 'r').read()

ch = intelligent_rag_chunker.IntelligentChunker()
res = ch.chunk_document(sample)
print(json.dumps([{'id': c.id, 'text_preview': c.text, 'meta': c.meta} for c in res], indent=2))
