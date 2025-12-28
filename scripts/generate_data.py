#!/usr/bin/env python3
"""
Generate synthetic product review data for sentiment classification.

This script creates realistic Amazon-style product reviews with positive,
negative, and neutral labels. It includes edge cases like negation,
mixed sentiment, typos, and varying lengths.

Usage:
    python generate_data.py [--seed SEED] [--output PATH] [--rows N]
"""

import argparse
import csv
import random
from pathlib import Path


# === POSITIVE REVIEW TEMPLATES AND COMPONENTS ===

POSITIVE_OPENINGS = [
    "Absolutely love this product!",
    "Best purchase I've made in a long time.",
    "Exceeded my expectations in every way.",
    "Can't believe how good this is.",
    "So happy with this purchase!",
    "This is exactly what I was looking for.",
    "Wow, just wow.",
    "Five stars all the way!",
    "Highly recommend this to everyone.",
    "This has changed my life.",
    "Outstanding quality!",
    "Perfect in every way.",
    "I'm genuinely impressed.",
    "Worth every single penny.",
    "This is a game changer.",
]

POSITIVE_DETAILS = [
    "The quality is exceptional and you can tell it's built to last.",
    "Setup was a breeze and it works perfectly right out of the box.",
    "Customer service was incredibly helpful when I had questions.",
    "It arrived earlier than expected and was packaged beautifully.",
    "The attention to detail is remarkable.",
    "Performs even better than advertised.",
    "My whole family loves it.",
    "I've already recommended it to all my friends.",
    "The design is sleek and modern.",
    "It's so intuitive and easy to use.",
    "The materials feel premium and durable.",
    "It's been running flawlessly for months now.",
    "Way better than the competitor products I've tried.",
    "The value for money is unbeatable.",
    "Every feature works exactly as described.",
]

POSITIVE_CLOSINGS = [
    "Will definitely buy again!",
    "Already ordered another one as a gift.",
    "Can't recommend this enough.",
    "You won't regret this purchase.",
    "10/10 would recommend.",
    "A must-have for sure.",
    "Don't hesitate, just buy it!",
    "Thank you for making such a great product.",
    "Best money I've ever spent.",
    "I'm a customer for life now.",
]

# === NEGATIVE REVIEW TEMPLATES AND COMPONENTS ===

NEGATIVE_OPENINGS = [
    "Extremely disappointed with this purchase.",
    "Complete waste of money.",
    "I want my money back.",
    "Do NOT buy this product.",
    "Terrible experience from start to finish.",
    "I can't believe how bad this is.",
    "Worst purchase I've ever made.",
    "Save your money and look elsewhere.",
    "This is absolute garbage.",
    "I regret buying this so much.",
    "What a letdown.",
    "Totally not worth it.",
    "I'm so frustrated with this product.",
    "This should be recalled.",
    "How is this even legal to sell?",
]

NEGATIVE_DETAILS = [
    "Broke within the first week of use.",
    "The quality is incredibly cheap and flimsy.",
    "Doesn't work as advertised at all.",
    "Customer service was useless and rude.",
    "Arrived damaged and took forever to ship.",
    "The instructions are impossible to follow.",
    "It smells like chemicals and gave me a headache.",
    "Stopped working after just a few days.",
    "The sizing is completely off from the description.",
    "It looks nothing like the pictures online.",
    "Made awful noises right from the start.",
    "The battery dies after only an hour.",
    "It overheats and becomes dangerous to use.",
    "Parts were missing from the package.",
    "The build quality is embarrassingly poor.",
]

NEGATIVE_CLOSINGS = [
    "Returning this immediately.",
    "I wish I could give zero stars.",
    "Never buying from this brand again.",
    "Learn from my mistake and avoid this.",
    "Complete scam in my opinion.",
    "I feel ripped off.",
    "Don't make the same mistake I did.",
    "Going back to my old product.",
    "Contacted my credit card for a chargeback.",
    "Threw it in the trash where it belongs.",
]

# === NEUTRAL REVIEW TEMPLATES AND COMPONENTS ===

NEUTRAL_OPENINGS = [
    "It's okay, I guess.",
    "Decent product for the price.",
    "Not bad, but not great either.",
    "It does what it's supposed to do.",
    "Average product overall.",
    "Mixed feelings about this one.",
    "It's fine for basic use.",
    "Nothing special, but it works.",
    "Serviceable but unremarkable.",
    "Does the job adequately.",
    "It's exactly what you'd expect for this price.",
    "Neither impressed nor disappointed.",
    "Just an ordinary product.",
    "It has its pros and cons.",
    "Middle of the road quality.",
]

NEUTRAL_DETAILS = [
    "Some features work well, others not so much.",
    "The quality is acceptable but nothing premium.",
    "It took a while to set up but works now.",
    "Customer service was neither helpful nor unhelpful.",
    "Shipping was on time but packaging was basic.",
    "It does the basics but lacks advanced features.",
    "Works fine for occasional use.",
    "I've seen better, but I've also seen worse.",
    "The design is functional but plain.",
    "It meets the minimum requirements.",
    "Some minor issues but nothing major.",
    "It's adequate for the price point.",
    "Could use some improvements but acceptable.",
    "It's a standard product with no surprises.",
    "Works as expected, no more no less.",
]

NEUTRAL_CLOSINGS = [
    "Might look for alternatives next time.",
    "It's fine if you just need something basic.",
    "Not sure if I'd buy again, maybe.",
    "It's okay for what it is.",
    "Would give it 3 out of 5 stars.",
    "Take it or leave it.",
    "It serves its purpose.",
    "Nothing to complain about, nothing to praise.",
    "It'll do for now.",
    "Average purchase, average experience.",
]

# === EDGE CASES ===

# Short reviews (single phrase)
SHORT_POSITIVE = [
    "Great!",
    "Love it!",
    "Perfect!",
    "Amazing!",
    "Excellent!",
    "Fantastic!",
    "Wonderful!",
    "Outstanding!",
    "Superb!",
    "Brilliant!",
    "A++",
    "Best ever!",
    "So good!",
    "Obsessed!",
    "Must buy!",
]

SHORT_NEGATIVE = [
    "Terrible!",
    "Awful!",
    "Garbage!",
    "Horrible!",
    "Waste!",
    "Junk!",
    "Trash!",
    "Broken!",
    "Useless!",
    "Disgusting!",
    "Worst ever!",
    "Total scam!",
    "Avoid!",
    "Returned it!",
    "Never again!",
]

SHORT_NEUTRAL = [
    "It's okay.",
    "Meh.",
    "Average.",
    "Decent.",
    "Acceptable.",
    "Fair.",
    "Alright.",
    "So-so.",
    "Fine.",
    "Passable.",
    "Nothing special.",
    "Basic.",
    "Standard.",
    "Regular.",
    "Ordinary.",
]

# Negation patterns (tricky for simple models)
NEGATION_POSITIVE = [
    "I thought it wouldn't work but I was wrong - it's excellent!",
    "Not a single complaint from me. Absolutely perfect!",
    "Can't say anything bad about this product. Love it!",
    "Never thought I'd find something this good.",
    "This isn't your typical cheap product - it's actually quality.",
    "I was skeptical but now I can't imagine life without it.",
    "No regrets whatsoever about this purchase!",
    "Don't believe the negative reviews, this thing is amazing.",
    "It's not bad at all - in fact, it's the best I've owned!",
    "Never been happier with a purchase.",
]

NEGATION_NEGATIVE = [
    "Not good at all.",
    "This is not what I expected and I'm disappointed.",
    "I can't recommend this to anyone.",
    "Doesn't work, don't buy.",
    "Not worth the money in any way.",
    "It's not terrible, but it's definitely not good either. Actually, it's pretty bad.",
    "I wouldn't buy this again if you paid me.",
    "Nothing about this product works correctly.",
    "I've never been so disappointed with a purchase.",
    "Can't return it fast enough.",
]

NEGATION_NEUTRAL = [
    "Not bad, not great.",
    "It's not the worst thing I've bought, but not the best either.",
    "I can't complain, but I can't praise it either.",
    "Wasn't expecting much and that's exactly what I got.",
    "It's not disappointing, but it's not impressive.",
    "Can't say I love it, can't say I hate it.",
    "Wouldn't say it's bad, just mediocre.",
    "Not unhappy with it, just underwhelmed.",
    "It didn't exceed expectations but didn't fall short either.",
    "Can't really form a strong opinion either way.",
]

# Mixed sentiment (tricky classifications)
MIXED_MOSTLY_POSITIVE = [
    "Great product but shipping took forever. Still worth it though!",
    "Love the quality, hate the price, but I'd buy again.",
    "Minor scratches on arrival but works perfectly otherwise.",
    "Instructions were confusing but the product itself is amazing.",
    "Took a while to figure out but now I love it!",
    "Packaging was damaged but product was fine. Very happy overall.",
    "A bit overpriced but the quality justifies it completely.",
    "Small learning curve but totally worth the effort.",
    "Customer service was slow but the product makes up for it.",
    "Wish it came in more colors, but the one I got is great!",
]

MIXED_MOSTLY_NEGATIVE = [
    "Nice design but falls apart after a week. Not worth it.",
    "Looks great in photos but terrible in real life. Disappointed.",
    "Fast shipping at least, but the product itself is garbage.",
    "Good concept, awful execution. Save your money.",
    "Love the idea, hate the implementation. Returning it.",
    "Cheap price for a reason - you get what you pay for.",
    "It works... barely. Too many issues to recommend.",
    "Pretty to look at but completely non-functional.",
    "Started great but deteriorated quickly. Regret buying.",
    "Some good features buried under tons of problems.",
]

MIXED_TRULY_NEUTRAL = [
    "Good build quality but lacks important features. Hard to rate.",
    "Pros and cons balance out evenly. It's just average.",
    "Fast shipping, slow product. Average experience overall.",
    "Love some aspects, hate others. Can't decide how I feel.",
    "Great for some things, terrible for others. Depends on your needs.",
    "Quality is good but value for money is questionable.",
    "Works well sometimes, doesn't work other times. Frustrating but okay.",
    "The good and bad cancel each other out.",
    "Some days I like it, some days I don't. Mixed bag overall.",
    "Exceeded expectations in some ways, disappointed in others.",
]

# Reviews with typos and informal language
INFORMAL_POSITIVE = [
    "omg this thing is AMAZING!!!!! soooo happy rn",
    "legit the best thing ive ever bought lol",
    "my friend recommended this and she was SO right",
    "cant belive how good this is tbh",
    "this thing slaps no cap fr fr",
    "yoooo this is actually fire ngl",
    "ok but like... this is actually so good???",
    "def worth it 100%",
    "bruh this changed my life no joke",
    "lowkey obsessed w this product haha",
]

INFORMAL_NEGATIVE = [
    "ugh this thing is sooo bad dont waste ur money",
    "literally the worst thing ive ever bought smh",
    "nope nope nope returning this asap",
    "wtf is this garbage lmao",
    "bruh moment... this thing sucks",
    "im so mad rn this is trash",
    "should of read the reviews first smh",
    "complete waist of money tbh",
    "this aint it chief",
    "yikes... just yikes",
]

INFORMAL_NEUTRAL = [
    "idk its alright i guess",
    "meh its okay nothing special tbh",
    "its fine whatever",
    "like its not bad but its not good either ya know",
    "kinda mid ngl",
    "eh could be worse could be better",
    "its aight for the price i suppose",
    "not great not terrible just ok",
    "its giving average energy lol",
    "idk how to feel about this one honestly",
]


def generate_positive_review(rng: random.Random) -> str:
    """Generate a positive review with varying structure."""
    structure = rng.choice(["short", "medium", "long", "edge"])

    if structure == "short":
        return rng.choice(SHORT_POSITIVE)
    elif structure == "medium":
        return f"{rng.choice(POSITIVE_OPENINGS)} {rng.choice(POSITIVE_DETAILS)}"
    elif structure == "long":
        return f"{rng.choice(POSITIVE_OPENINGS)} {rng.choice(POSITIVE_DETAILS)} {rng.choice(POSITIVE_CLOSINGS)}"
    else:  # edge cases
        edge_type = rng.choice(["negation", "mixed", "informal"])
        if edge_type == "negation":
            return rng.choice(NEGATION_POSITIVE)
        elif edge_type == "mixed":
            return rng.choice(MIXED_MOSTLY_POSITIVE)
        else:
            return rng.choice(INFORMAL_POSITIVE)


def generate_negative_review(rng: random.Random) -> str:
    """Generate a negative review with varying structure."""
    structure = rng.choice(["short", "medium", "long", "edge"])

    if structure == "short":
        return rng.choice(SHORT_NEGATIVE)
    elif structure == "medium":
        return f"{rng.choice(NEGATIVE_OPENINGS)} {rng.choice(NEGATIVE_DETAILS)}"
    elif structure == "long":
        return f"{rng.choice(NEGATIVE_OPENINGS)} {rng.choice(NEGATIVE_DETAILS)} {rng.choice(NEGATIVE_CLOSINGS)}"
    else:  # edge cases
        edge_type = rng.choice(["negation", "mixed", "informal"])
        if edge_type == "negation":
            return rng.choice(NEGATION_NEGATIVE)
        elif edge_type == "mixed":
            return rng.choice(MIXED_MOSTLY_NEGATIVE)
        else:
            return rng.choice(INFORMAL_NEGATIVE)


def generate_neutral_review(rng: random.Random) -> str:
    """Generate a neutral review with varying structure."""
    structure = rng.choice(["short", "medium", "long", "edge"])

    if structure == "short":
        return rng.choice(SHORT_NEUTRAL)
    elif structure == "medium":
        return f"{rng.choice(NEUTRAL_OPENINGS)} {rng.choice(NEUTRAL_DETAILS)}"
    elif structure == "long":
        return f"{rng.choice(NEUTRAL_OPENINGS)} {rng.choice(NEUTRAL_DETAILS)} {rng.choice(NEUTRAL_CLOSINGS)}"
    else:  # edge cases
        edge_type = rng.choice(["negation", "mixed", "informal"])
        if edge_type == "negation":
            return rng.choice(NEGATION_NEUTRAL)
        elif edge_type == "mixed":
            return rng.choice(MIXED_TRULY_NEUTRAL)
        else:
            return rng.choice(INFORMAL_NEUTRAL)


def generate_dataset(
    num_per_class: int = 100,
    seed: int = 42,
    shuffle: bool = True,
) -> list[tuple[str, str]]:
    """
    Generate a balanced dataset of product reviews.

    Args:
        num_per_class: Number of reviews per sentiment class
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the final dataset

    Returns:
        List of (text, label) tuples
    """
    rng = random.Random(seed)
    data = []

    # Generate reviews for each class
    for _ in range(num_per_class):
        data.append((generate_positive_review(rng), "positive"))
        data.append((generate_negative_review(rng), "negative"))
        data.append((generate_neutral_review(rng), "neutral"))

    if shuffle:
        rng.shuffle(data)

    return data


def write_csv(data: list[tuple[str, str]], output_path: Path) -> None:
    """Write the dataset to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text", "label"])
        writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic product review data for sentiment classification"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "project" / "data" / "reviews.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=300,
        help="Total number of rows (will be rounded to nearest multiple of 3 for balance)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle the dataset (keep classes grouped)",
    )

    args = parser.parse_args()

    # Calculate per-class count (ensure balanced)
    num_per_class = args.rows // 3
    if num_per_class < 1:
        num_per_class = 1

    total_rows = num_per_class * 3

    print(f"Generating {total_rows} reviews ({num_per_class} per class)...")
    print(f"Random seed: {args.seed}")
    print(f"Output path: {args.output}")

    data = generate_dataset(
        num_per_class=num_per_class,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    write_csv(data, args.output)

    print(f"Successfully wrote {len(data)} reviews to {args.output}")

    # Print distribution summary
    label_counts = {}
    for _, label in data:
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count / len(data) * 100:.1f}%)")


if __name__ == "__main__":
    main()
