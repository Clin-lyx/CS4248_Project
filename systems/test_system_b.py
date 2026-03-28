from systems.system_b_encdec import generate_candidates

headline = "government announces new economic policy"

outputs = generate_candidates(
    headline,
    direction="n2s",
    k=3
)

for i, output in enumerate(outputs, 1):
    print(f"{i}. {output}")