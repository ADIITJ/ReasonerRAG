import pickle

# Load the summaries file
with open("data/community_summaries.pkl", "rb") as f:
    summaries = pickle.load(f)

# Print all summaries with their keys
print("\n✅ Loaded Summaries:\n")
for key, summary in summaries.items():
    print(f"--- Summary Key: {key} ---")
    print(summary)
    print("\n" + "="*80 + "\n")
