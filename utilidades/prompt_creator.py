from langchain.prompts import ChatPromptTemplate

def crear_prompt(texto):
    system_prompt = (
        texto +
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    return prompt
