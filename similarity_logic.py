# Import numpy for math and sklearn to calculate similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. DEFINE OUR IDEAS AS VECTORS: 
# Imagine [Food-ness, Tech-ness]
# 'Pizza' is high on Food, low on Tech
pizza_vector = np.array([[0.9, 0.1]]) 

# 'Burger' is also high on Food, low on Tech
burger_vector = np.array([[0.8, 0.2]])

# 'Smartphone' is low on Food, high on Tech
phone_vector = np.array([[0.1, 0.9]])

# 2. THE SEARCH: How similar is a 'Burger' to 'Pizza'?
# 1.0 means identical, 0.0 means completely different
food_score = cosine_similarity(pizza_vector, burger_vector)

# 3. THE SEARCH: How similar is a 'Smartphone' to 'Pizza'?
tech_score = cosine_similarity(pizza_vector, phone_vector)

print("--- SIMILARITY AUDIT ---")
print(f"Similarity Score (Pizza vs Burger): {food_score[0][0]:.4f}")
print(f"Similarity Score (Pizza vs Smartphone): {tech_score[0][0]:.4f}")

# 4. LOGIC CHECK
if food_score > tech_score:
    print("\nResult: The computer knows Pizza is more like a Burger than a Phone!")

