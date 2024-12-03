import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Custom decision tree implementation
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, n_features, replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = np.bincount(y)
        most_common = np.argmax(counter)
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Load data
data_hewan = pd.read_csv('DATABASE KB TERBARU - Sheet1.csv')
df_hewan = pd.DataFrame(data_hewan)

# Features and labels
X = df_hewan.drop(columns=['Nama hewan']).values
y = df_hewan['Nama hewan'].astype('category').cat.codes.values

# Train the model
tree = DecisionTree()
tree.fit(X, y)

app = Flask(__name__, template_folder="templates/")

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/home')
def back_toHome():
    return render_template('index2.html')

@app.route('/process_result', methods=['POST'])
def process_result():
    data = request.get_json()
    result = data['result']
    prediction = tree.predict([result])
    animal_name = df_hewan['Nama hewan'].astype('category').cat.categories[prediction[0]]
    return jsonify({'animal': animal_name})

@app.route('/add_newAnimal', methods=['POST'])
def add_animal_csv():
    global df_hewan, tree
    
    try:
        data = request.get_json()
        new_animal = data['result']
        animal_name = new_animal[0]
        animal_features = new_animal[1:]
        
        # Buat DataFrame baru dengan data baru
        new_data = pd.DataFrame([[animal_name] + animal_features], columns=['Nama hewan'] + list(df_hewan.columns[1:]))
        
        # Tambahkan data baru ke DataFrame asli
        df_hewan = pd.concat([df_hewan, new_data], ignore_index=True)
        
        # Simpan DataFrame ke CSV
        df_hewan.to_csv('DATABASE KB TERBARU - Sheet1.csv', index=False)
        
        # Latih ulang model dengan data yang diperbarui
        X = df_hewan.drop(columns=['Nama hewan']).values
        y = df_hewan['Nama hewan'].astype('category').cat.codes.values
        tree = DecisionTree()
        tree.fit(X, y)
        
        return jsonify({'status': 'success', 'message': 'Animal added successfully'})
    except Exception as e:
        logging.error("Error occurred: %s", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
