def build_prompt(text, direction):
    if direction == "n2s":
        return f"""
You rewrite neutral news headlines into sarcastic Onion-style headlines.

Rules:
- Input will always be a news headline.
- Keep the same topic and preserve all names, organisations, places, dates and numbers.
- Make the headline obviously sarcastic.
- Use irony, exaggerated praise, absurd confidence, or ridiculous overstatement.
- Do not simply paraphrase the original headline.
- Output exactly one short headline and nothing else.

Examples:

Neutral: Government announces new economic policy
Sarcastic: Government proudly unveils economic policy guaranteed to solve literally every problem forever

Neutral: City opens new parking garage
Sarcastic: City finally ends all traffic forever with one more parking garage

Neutral: Scientists discover water on Mars
Sarcastic: Scientists shocked to discover Mars contains the one thing humans have been searching for

Neutral: Changi Airport opens Terminal 5
Sarcastic: Changi Airport finally solves air travel forever by opening Terminal 5

Neutral: {text}
Sarcastic:
""".strip()

    elif direction == "s2n":
        return f"""
You rewrite sarcastic Onion-style headlines into neutral factual news headlines.

Rules:
- Input will always be a sarcastic news headline.
- Remove sarcasm, irony and exaggeration.
- Preserve all names, organisations, places, dates and numbers.
- Keep the same topic.
- Output exactly one short factual headline and nothing else.

Examples:

Sarcastic: Government proudly unveils economic policy guaranteed to solve literally every problem forever
Neutral: Government announces new economic policy

Sarcastic: City finally ends all traffic forever with one more parking garage
Neutral: City opens new parking garage

Sarcastic: Changi Airport finally solves air travel forever by opening Terminal 5
Neutral: Changi Airport opens Terminal 5

Sarcastic: {text}
Neutral:
""".strip()

    else:
        raise ValueError(f"Unknown direction: {direction}")