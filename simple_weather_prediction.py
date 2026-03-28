# Dataset
data = [
    ["Sunny","Hot","High","Weak","No"],
    ["Sunny","Hot","High","Strong","No"],
    ["Cloudy","Hot","High","Weak","Yes"],
    ["Rainy","Mild","High","Weak","Yes"],
    ["Rainy","Cool","Normal","Weak","Yes"],
    ["Rainy","Cool","Normal","Strong","No"],
    ["Cloudy","Cool","Normal","Strong","Yes"],
    ["Sunny","Mild","High","Weak","No"],
    ["Sunny","Cool","Normal","Weak","Yes"],
    ["Rainy","Mild","Normal","Weak","Yes"]
]
features = ["Outlook", "Temp", "Humidity", "Wind"]

# Separate X and y
X = [row[:4] for row in data]
y = [row[4] for row in data]

# Unique classes
classes = list(set(y))

# Prior probabilities P(Class)
class_counts = {}
for c in y:
    class_counts[c] = class_counts.get(c, 0) + 1

total_samples = len(y)
priors = {c: class_counts[c] / total_samples for c in classes}

# Likelihoods P(feature=value | class)
# Structure: likelihoods[class][feature_index][value] = probability

likelihoods = {}

for c in classes:
    likelihoods[c] = {}
    # filter rows of this class
    X_c = [X[i] for i in range(len(X)) if y[i] == c]

    for j in range(len(features)):
        likelihoods[c][j] = {}
        
        # possible values for this feature
        values = set([row[j] for row in X])
        
        for v in values:
            # count occurrences
            count = sum(1 for row in X_c if row[j] == v)
            
            # Laplace smoothing
            likelihoods[c][j][v] = (count + 1) / (len(X_c) + len(values))

# Prediction function
def predict(sample):
    probs = {}
    
    for c in classes:
        prob = priors[c]
        
        for j in range(len(sample)):
            value = sample[j]
            prob *= likelihoods[c][j].get(value, 1e-6)
        
        probs[c] = prob
    
    # return class with max probability
    return max(probs, key=probs.get), probs

# Test sample
test = ["Sunny","Cool","High","Strong"]

prediction, probabilities = predict(test)
print("Probabilities:", probabilities)
print("Prediction:", prediction)