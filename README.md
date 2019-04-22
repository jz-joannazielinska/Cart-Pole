# Cart-Pole

Implementation of Q-learning and SARSA algorithms for Cart-Pole problem. 


Values of observation were discretized and assigned to chosen numbers of buckets per variable. 

Learning was tested for several parameters:
    --> learning_rate_values = [0.1, 0.3, 0.5]
    --> epsilon_values = [0.9, 0.85, 0.8]
    --> buckets = [(2, 2, 8, 3), (2, 2, 6, 6), (1, 1, 6, 3)]
    --> discount_values = [0.9, 0.98, 1.0]
    
For paramteres that performed the best, learning was performed multiple times to check if learning is stable. 
