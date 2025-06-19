import numpy as np
from scipy.spatial.distance import cosine

class RecommendationSystem:
    def __init__(self):
        self.user_ratings = {}
        self.item_ratings = {}
        
    def add_rating(self, user_id, item_id, rating):
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        self.user_ratings[user_id][item_id] = rating
        
        if item_id not in self.item_ratings:
            self.item_ratings[item_id] = {}
        self.item_ratings[item_id][user_id] = rating
    
    def get_user_similarity(self, user1, user2):
        items = set(self.user_ratings[user1].keys()) & set(self.user_ratings[user2].keys())
        if not items:
            return 0
        
        user1_ratings = [self.user_ratings[user1][item] for item in items]
        user2_ratings = [self.user_ratings[user2][item] for item in items]
        
        return 1 - cosine(user1_ratings, user2_ratings)
    
    def get_item_similarity(self, item1, item2):
        users = set(self.item_ratings[item1].keys()) & set(self.item_ratings[item2].keys())
        if not users:
            return 0
        
        item1_ratings = [self.item_ratings[item1][user] for user in users]
        item2_ratings = [self.item_ratings[item2][user] for user in users]
        
        return 1 - cosine(item1_ratings, item2_ratings)
    
    def get_user_based_recommendations(self, user_id, n_recommendations=5):
        if user_id not in self.user_ratings:
            return []
        
        user_sim_scores = {}
        for other_user in self.user_ratings:
            if other_user != user_id:
                sim = self.get_user_similarity(user_id, other_user)
                user_sim_scores[other_user] = sim
        
        recommendations = {}
        for other_user, sim in user_sim_scores.items():
            if sim <= 0:
                continue
            
            for item in self.user_ratings[other_user]:
                if item not in self.user_ratings[user_id]:
                    if item not in recommendations:
                        recommendations[item] = 0
                    recommendations[item] += sim * self.user_ratings[other_user][item]
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]
    
    def get_item_based_recommendations(self, user_id, n_recommendations=5):
        if user_id not in self.user_ratings:
            return []
        
        user_items = set(self.user_ratings[user_id].keys())
        recommendations = {}
        
        for item in self.item_ratings:
            if item not in user_items:
                item_sim_sum = 0
                weighted_sum = 0
                
                for user_item in user_items:
                    sim = self.get_item_similarity(item, user_item)
                    if sim > 0:
                        item_sim_sum += sim
                        weighted_sum += sim * self.user_ratings[user_id][user_item]
                
                if item_sim_sum > 0:
                    recommendations[item] = weighted_sum / item_sim_sum
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:n_recommendations]

def main():
    rs = RecommendationSystem()
    
    sample_ratings = [
        (1, 'movie1', 5),
        (1, 'movie2', 3),
        (1, 'movie3', 4),
        (2, 'movie1', 3),
        (2, 'movie2', 4),
        (2, 'movie4', 5),
        (3, 'movie1', 4),
        (3, 'movie3', 5),
        (3, 'movie4', 2),
    ]
    
    for user_id, movie_id, rating in sample_ratings:
        rs.add_rating(user_id, movie_id, rating)
    
    print("User-based recommendations for user 1:")
    print(rs.get_user_based_recommendations(1))
    
    print("\nItem-based recommendations for user 1:")
    print(rs.get_item_based_recommendations(1))

if __name__ == '__main__':
    main()