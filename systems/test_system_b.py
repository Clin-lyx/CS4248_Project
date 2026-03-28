from systems.system_b_encdec import generate_candidates


def main():
    headline = "Changi Airport just opened Terminal 5"

    outputs = generate_candidates(
        input_text=headline,
        direction="n2s",
        k=5,
    )

    print(f"\nInput: {headline}\n")

    for i, output in enumerate(outputs, 1):
        print(f"{i}. {output}")


if __name__ == "__main__":
    main()