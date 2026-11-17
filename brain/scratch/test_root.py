import os

def find_root():
    curr = os.path.abspath(os.getcwd())
    while curr and not os.path.exists(os.path.join(curr, "backend")):
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return curr

print("From current dir:", find_root())
os.chdir("backend/notebooks")
print("From backend/notebooks:", find_root())
os.chdir(find_root())
print("Back to root:", os.getcwd())
assert os.path.exists("backend/checkpoints/improved_rnn_encoder.pt")
print("Checkpoints exist here: TRUE")
