import pickle

test_fp = "benchmark_simple_majority_a2c.pkl"
res = pickle.load(open(test_fp, "rb"))

print(res)