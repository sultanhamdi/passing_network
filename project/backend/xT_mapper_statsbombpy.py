import numpy as np
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
x_bins, y_bins = 16, 12
pitch_length, pitch_width = 120, 80

def get_cell(x, y):
    x_idx = min(int(x / pitch_length * x_bins), x_bins - 1)
    y_idx = min(int(y / pitch_width * y_bins), y_bins - 1)
    return x_idx, y_idx

# Load all La Liga 2020/21 matches (Only Barca matches anyways)
matches = sb.matches(competition_id=11, season_id=90)
events_df = pd.concat([sb.events(match_id=mid) for mid in tqdm(matches['match_id'], desc="Loading events")], ignore_index=True)

# Initialize counts
shot_counts = np.zeros((x_bins, y_bins))
goal_counts = np.zeros((x_bins, y_bins))
pass_counts = np.zeros((x_bins, y_bins))
transition_counts = np.zeros((x_bins, y_bins, x_bins, y_bins))

# Count passes and shots
for _, row in events_df.iterrows():
    if not isinstance(row.get('location'), list):
        continue
    x0, y0 = row['location']
    x_idx, y_idx = get_cell(x0, y0)

    if row['type'] == 'Shot':
        shot_counts[x_idx][y_idx] += 1
        if row.get('shot_outcome') == 'Goal':
            goal_counts[x_idx][y_idx] += 1
    elif row['type'] == 'Pass' and isinstance(row.get('pass_end_location'), list):
        x1, y1 = row['pass_end_location']
        x1_idx, y1_idx = get_cell(x1, y1)
        pass_counts[x_idx][y_idx] += 1
        transition_counts[x_idx][y_idx][x1_idx][y1_idx] += 1

# Calculate probabilities
total_actions = shot_counts + pass_counts
P_shot = np.divide(shot_counts, total_actions, out=np.zeros_like(shot_counts), where=total_actions != 0)
P_goal = np.divide(goal_counts, shot_counts, out=np.zeros_like(goal_counts), where=shot_counts != 0)
P_move = 1 - P_shot

# Normalize transition probabilities
P_transition = np.zeros_like(transition_counts)
for x in range(x_bins):
    for y in range(y_bins):
        total = np.sum(transition_counts[x][y])
        if total > 0:
            P_transition[x][y] = transition_counts[x][y] / total

# Compute xT grid iteratively
xT = np.zeros((x_bins, y_bins))
for _ in range(20):  # 20 iterations
    new_xT = np.copy(xT)
    for x in range(x_bins):
        for y in range(y_bins):
            shot_val = P_shot[x][y] * P_goal[x][y]
            move_val = np.sum(P_transition[x][y] * xT)
            new_xT[x][y] = shot_val + P_move[x][y] * move_val
    xT = new_xT

# Visualize xT
plt.figure(figsize=(10, 6))
sns.heatmap(xT.T, cmap='inferno', square=True, cbar_kws={"label": "xT Value"})
plt.title("Expected Threat (xT) Grid")
plt.xlabel("Pitch X Bins")
plt.ylabel("Pitch Y Bins")
plt.tight_layout()
plt.show()

np.save("barcelona_xT", xT)