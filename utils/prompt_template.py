def get_prompt(context, question):
    return f"""
You are a helpful teaching assistant. Answer the question using the context below.
If the answer is not in the context, try to reason and give a best guess or say 'Not enough information'.

Context:
{context}

Question:
{question}
"""
