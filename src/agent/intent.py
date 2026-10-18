"""Conservative write-intent refusal before caches, SQL generation or execution.

This recognises common requests, not every possible paraphrase. SQLite read-only
permissions remain the enforcement boundary for generated SQL.
"""
import re
import unicodedata

REFUSAL = ('This assistant only supports read-only analysis. I cannot delete, insert, '
           'update or otherwise modify database records or its structure. No changes '
           'were made. You can ask me to count, filter, compare or chart the data instead.')


def requests_data_change(question):
    text = ''.join(c for c in unicodedata.normalize('NFKD', question.casefold().replace('đ', 'd')) if not unicodedata.combining(c))
    # Ignore fenced formatting and normalise punctuation, retaining SQL words.
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)
    # An explicit SQL write statement must not be translated into a SELECT.
    sql = r'\b(?:delete\s+from|insert\s+into|replace\s+into|update\s+[a-z0-9_]+\s+set|(?:drop|alter|create|truncate)\s+(?:table|database|index|view|trigger)|attach\s+database|grant\s+|revoke\s+)'
    if re.search(sql, text):
        return True
    # Deduplicating the returned output is analysis, not a database mutation.
    text = re.sub(r'\bremove\s+duplicates\s+from\s+(?:the\s+)?(?:result|output)\b', 'deduplicate output', text)
    if re.search(r'\b(?:delete|erase|purge|wipe)\s+(?:everything|it|them)\b', text) or re.fullmatch(r'\s*(?:please\s+)?(?:delete|erase|purge|wipe)\s+all\s*', text):
        return True
    actions = r'(?:delete|remove|erase|purge|wipe|clear|update|modify|edit|insert|add|overwrite|change|set|mark|rename|drop|truncate|xoa|sua|cap\s+nhat|chen|them|ghi\s+de|doi)'
    entities = r'(?:data|database|dataset|records?|rows?|tables?|columns?|orders?|customers?|products?|payments?|reviews?|statuses?|prices?|amounts?|du\s+lieu|co\s+so\s+du\s+lieu|bang|don\s+hang|khach\s+hang|dong|cot|trang\s+thai)'
    pattern = rf'\b{actions}\b(?:\s+[a-z0-9_]+){{0,8}}\s+\b{entities}\b'
    return bool(re.search(pattern, text))
