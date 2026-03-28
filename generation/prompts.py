def build_prompt(text, direction):
    if direction == "n2s":
        return f"""
Rewrite the neutral headline as a sarcastic news headline.

Use one or more of:
- exaggerated praise
- obvious irony
- "surely", "of course", "finally", "yet another"
- imply the policy/action will magically solve everything

Keep all names, places, numbers and dates unchanged.

Neutral: government announces new economic policy
Sarcastic: government proudly unveils yet another flawless economic policy sure to fix absolutely everything

Neutral: city opens new parking garage
Sarcastic: city finally solves all transportation problems with another parking garage

Neutral: scientists discover water on mars
Sarcastic: scientists shocked to learn mars may contain the one thing humans keep looking for

Neutral: {text}
Sarcastic:
""".strip()