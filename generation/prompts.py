def build_prompt(text, direction):
    if direction == "n2s":
        return f"""
You are rewriting neutral news headlines into sarcastic Onion-style headlines.

Requirements:
- Keep the same topic and entities.
- Keep all names, places, organisations and numbers unchanged.
- Make the headline obviously sarcastic.
- Use exaggeration, irony, absurd confidence, or over-the-top praise.
- Do NOT simply paraphrase the original headline.
- Produce exactly one short headline.

Neutral: Government announces new economic policy
Sarcastic: Government proudly unveils yet another flawless economic policy guaranteed to solve everything forever

Neutral: City opens new parking garage
Sarcastic: City finally ends all traffic problems with one more parking garage

Neutral: Scientists discover water on Mars
Sarcastic: Scientists shocked to learn Mars may contain the one thing humans keep searching for

Neutral: Singapore Airlines opens Terminal 5
Sarcastic:
""".strip()

    elif direction == "s2n":
        return f"""
You are rewriting sarcastic headlines into neutral factual news headlines.

Requirements:
- Remove exaggeration and irony.
- Keep the same meaning and entities.
- Produce exactly one short factual headline.

Sarcastic: Government proudly unveils yet another flawless economic policy guaranteed to solve everything forever
Neutral: Government announces new economic policy

Sarcastic: City finally ends all traffic problems with one more parking garage
Neutral: City opens new parking garage

Sarcastic: {text}
Neutral:
""".strip()