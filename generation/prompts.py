def build_prompt(text, direction):
    if direction == "n2s":
        return f"""
Convert the neutral news headline into a sarcastic headline.

Example:
Neutral: city opens new parking garage
Sarcastic: city finally solves all transportation problems with another parking garage

Example:
Neutral: government announces new economic policy
Sarcastic: government proudly unveils another flawless economic policy guaranteed to fix everything

Neutral: {text}
Sarcastic:
""".strip()

    elif direction == "s2n":
        return f"""
Convert the sarcastic headline into a neutral factual headline.

Example:
Sarcastic: government proudly unveils another flawless economic policy guaranteed to fix everything
Neutral: government announces new economic policy

Example:
Sarcastic: city finally solves all transportation problems with another parking garage
Neutral: city opens new parking garage

Sarcastic: {text}
Neutral:
""".strip()