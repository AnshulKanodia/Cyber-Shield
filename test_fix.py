import requests, json

API = "http://localhost:5000/api"

tests = [
    # FALSE POSITIVES from user screenshots (should be SAFE)
    "Dost ke phone se photo leta to vo delete na karta apane se",
    "Can the voicemail be like u r my female bestie so that i can tease my real bestie",
    "Iff possible say some works like cutie, motu patlu which somehow make it more real",
    "Great job on your presentation today!",
    "The weather is beautiful lets go for a walk",
    "I dont want to see that movie again it was boring",
    "Happy birthday! Hope you have an amazing day!",
    "Thanks for helping me with my homework",
    # TRUE BULLYING (should be BULLYING)
    "just jump from the cliff i dont want to see u again",
    "nobody wants you here just leave and never come back",
    "the world would be a better place without you",
    "you are such a loser nobody likes you go kill yourself",
    "ur so dumb stfu nobody asked",
    "you are a waste of space and oxygen",
    "just go die no one will miss you",
]

print("=" * 85)
print(f"{'TEXT':<58} {'RESULT':<13} {'CONF':>5}")
print("=" * 85)

for text in tests:
    r = requests.post(f"{API}/predict", json={"text": text})
    d = r.json()
    label = d.get("label", "?")
    conf = d.get("confidence", 0)
    bp = d.get("bullying_probability", 0)
    override = " [P]" if d.get("pattern_override") else ""
    
    short = text[:55] + "..." if len(text) > 58 else text
    print(f"{short:<58} {label + override:<13} {conf*100:5.1f}%  (bully_p={bp:.2f})")

print("=" * 85)
