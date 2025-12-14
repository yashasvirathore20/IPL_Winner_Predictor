import os
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter, defaultdict

# ---------------- CONFIG ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
MATCH_INFO_PATH = os.path.join(BASE, "Match_Info.csv")
SCHEDULE_PATH = os.path.join(BASE, "schedule_2025.csv")
SIM_ITERS = 1000  # for probability runs (not used in single-run league output)
RNG_SEED = 42

# Current IPL teams (only these participate in full-season sims)
CURRENT_TEAMS = {
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "RCB": "Royal Challengers Bangalore",
    "KKR": "Kolkata Knight Riders",
    "SRH": "Sunrisers Hyderabad",
    "RR": "Rajasthan Royals",
    "DC": "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants"
}

PLACEHOLDER_TOKENS = ["QUAL", "WINNER", "ELIM", "FINAL", "T1", "T2", "T3", "T4"]


def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

df_hist = safe_read_csv("Match_Info.csv")
df_sched = safe_read_csv("schedule_2025.csv")


df_hist = df_hist.dropna(subset=["team1", "team2", "winner", "toss_winner"])
df_hist = df_hist[
    (df_hist["team1"].astype(str) != "") &
    (df_hist["team2"].astype(str) != "") &
    (df_hist["winner"].astype(str) != "") &
    (df_hist["toss_winner"].astype(str) != "")
].copy()

# normalize toss decision
df_hist["toss_decision"] = df_hist["toss_decision"].astype(str).str.lower().apply(lambda x: "bat" if x == "bat" else "field")

# ---------------- MAP SCHEDULE CODES ----------------
# first build a best-effort abbreviation map (extend if needed)
abbr_map = {k: v for k, v in CURRENT_TEAMS.items()}

# function to detect placeholders
def is_placeholder(code):
    s = str(code).upper()
    return any(tok in s for tok in PLACEHOLDER_TOKENS)

# Build mapping from schedule codes -> names (prefer CURRENT_TEAMS then history)
history_names = pd.unique(pd.concat([df_hist["team1"], df_hist["team2"], df_hist["winner"], df_hist["toss_winner"]]))
schedule_codes = sorted(pd.unique(pd.concat([df_sched["team1"], df_sched["team2"]])))

schedule_to_name = {}
for code in schedule_codes:
    if is_placeholder(code):
        schedule_to_name[code] = None
        continue
    # prefer current teams abbreviation
    if code in abbr_map:
        schedule_to_name[code] = abbr_map[code]
        continue
    # try direct match with history
    matched = None
    for cand in history_names:
        if str(code).lower() == str(cand).lower() or str(code).lower() in str(cand).lower():
            matched = cand
            break
    schedule_to_name[code] = matched if matched is not None else code

# ---------------- BUILD FIXTURE LIST (league matches only) ----------------
league_fixtures = []
for _, r in df_sched.iterrows():
    a_code = r["team1"]
    b_code = r["team2"]
    venue = r.get("venue", None)
    # skip placeholders
    if is_placeholder(a_code) or is_placeholder(b_code):
        continue
    a = schedule_to_name.get(a_code, None)
    b = schedule_to_name.get(b_code, None)
    if a is None or b is None:  # skip if mapping failed
        continue
    # ensure both are current teams for full-season sim
    if a not in CURRENT_TEAMS.values() or b not in CURRENT_TEAMS.values():
        # This row may contain legacy teams — skip in league simulation
        continue
    league_fixtures.append((a, b, venue))

# Use the list as-is (no adding reverse fixtures) — keeps the exact schedule fixture count
print(f"Loaded {len(league_fixtures)} league fixtures from schedule (placeholders ignored).")

# ---------------- ENCODERS (based on historical teams + ensure current teams included) ----------------
hist_team_list = sorted(list(set(list(history_names) + list(CURRENT_TEAMS.values()))))
team_encoding = {team: idx for idx, team in enumerate(hist_team_list)}
reverse_team_encoding = {idx: team for team, idx in team_encoding.items()}

# venues
hist_venues = sorted(df_hist["venue"].dropna().unique().tolist())
sched_venues = sorted(pd.unique(df_sched["venue"].dropna().tolist()))
for v in sched_venues:
    if v not in hist_venues:
        hist_venues.append(v)
venue_encoding = {v: i for i, v in enumerate(sorted(hist_venues))}
reverse_venue_encoding = {i: v for v, i in venue_encoding.items()}

# ---------------- BUILD TRAINING DATA ----------------
df = df_hist.copy()
df["team1_enc"] = df["team1"].map(team_encoding)
df["team2_enc"] = df["team2"].map(team_encoding)
df["toss_winner_enc"] = df["toss_winner"].map(team_encoding)
df["winner_enc"] = df["winner"].map(team_encoding)
df["venue_enc"] = df["venue"].map(lambda x: venue_encoding.get(x, 0))
df["toss_decision_bin"] = df["toss_decision"].apply(lambda x: 1 if x == "bat" else 0)

train_df = df[df["winner_enc"].notna()].copy()
X = pd.get_dummies(train_df[["team1_enc", "team2_enc", "toss_winner_enc", "toss_decision_bin", "venue_enc"]].astype(int),
                   columns=["team1_enc", "team2_enc", "toss_winner_enc", "venue_enc"],
                   prefix=["team1", "team2", "toss_winner", "venue"])
y = train_df["winner_enc"].astype(int)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
feature_columns = X.columns.tolist()

print(f"\nModel trained. Training accuracy: {(model.predict(X)==y).mean()*100:.2f}%\n")

# ---------------- TOSS DECISION PROBS ----------------
toss_df = df_hist.copy()
toss_df["toss_decision_bin"] = toss_df["toss_decision"].astype(str).str.lower().apply(lambda x: 1 if x == "bat" else 0)
p_bat_by_venue = toss_df.groupby("venue")["toss_decision_bin"].mean().to_dict()
p_bat_global = toss_df["toss_decision_bin"].mean() if not toss_df["toss_decision_bin"].empty else 0.5

# ---------------- FEATURE VECTOR helper (return DataFrame to satisfy sklearn) ----------------
feature_index = {c: i for i, c in enumerate(feature_columns)}
n_cols = len(feature_columns)

def make_feature_df(t1, t2, toss_winner, toss_decision, venue):
    vec = np.zeros(n_cols, dtype=int)
    t1i = team_encoding.get(t1, None)
    t2i = team_encoding.get(t2, None)
    twi = team_encoding.get(toss_winner, None)
    venu_i = venue_encoding.get(venue, 0)
    # toss_decision column (find column that starts with 'toss_decision' if exists)
    toss_col_candidates = [c for c in feature_columns if c.startswith("toss_decision")]
    if toss_col_candidates:
        vec[feature_index[toss_col_candidates[0]]] = 1 if toss_decision == "bat" else 0
    # set one-hot team/venue fields
    if t1i is not None:
        cname = f"team1_{t1i}"
        if cname in feature_index: vec[feature_index[cname]] = 1
    if t2i is not None:
        cname = f"team2_{t2i}"
        if cname in feature_index: vec[feature_index[cname]] = 1
    if twi is not None:
        cname = f"toss_winner_{twi}"
        if cname in feature_index: vec[feature_index[cname]] = 1
    cname = f"venue_{venu_i}"
    if cname in feature_index: vec[feature_index[cname]] = 1
    df_vec = pd.DataFrame([vec], columns=feature_columns)
    return df_vec

# ---------------- Single-match prediction ----------------
def predict_single_match(team1, team2, venue):
    toss_winner = random.choice([team1, team2])
    p_bat = p_bat_by_venue.get(venue, p_bat_global)
    toss_decision = "bat" if random.random() < p_bat else "field"
    feat = make_feature_df(team1, team2, toss_winner, toss_decision, venue)
    proba = model.predict_proba(feat)[0]
    # map class probs to our two teams
    a_idx = team_encoding.get(team1)
    b_idx = team_encoding.get(team2)
    # get probability mass for a and b from model classes
    class_idx = list(model.classes_)
    pa = proba[class_idx.index(a_idx)] if a_idx in class_idx else 0.5
    pb = proba[class_idx.index(b_idx)] if b_idx in class_idx else 0.5
    winner = team1 if random.random() < pa/(pa+pb) else team2
    return winner, {"p_win_team1": pa/(pa+pb), "toss_winner": toss_winner, "toss_decision": toss_decision}

# ---------------- Run one full league + playoffs simulation (deterministic single run) ----------------
def run_one_tournament():
    rng = random.Random(RNG_SEED)
    # initialize stats
    teams = sorted(list(CURRENT_TEAMS.values()))
    points = {t: 0 for t in teams}
    matches_played = {t: 0 for t in teams}
    runs_for = {t: 0 for t in teams}
    runs_against = {t: 0 for t in teams}
    head2head = defaultdict(int)  # (winner, loser) -> wins

    # league matches: use exactly league_fixtures
    for a, b, venue in league_fixtures:
        toss_winner = rng.choice([a, b])
        p_bat = p_bat_by_venue.get(venue, p_bat_global)
        toss_decision = "bat" if rng.random() < p_bat else "field"
        feat = make_feature_df(a, b, toss_winner, toss_decision, venue)
        proba = model.predict_proba(feat)[0]
        class_idx = list(model.classes_)
        a_idx = team_encoding.get(a)
        b_idx = team_encoding.get(b)
        pa = proba[class_idx.index(a_idx)] if a_idx in class_idx else 0.5
        pb = proba[class_idx.index(b_idx)] if b_idx in class_idx else 0.5
        winner = a if rng.random() < pa/(pa+pb) else b
        loser = b if winner == a else a
        points[winner] += 2
        matches_played[a] += 1; matches_played[b] += 1
        # approximate runs: winner ~150+margin, loser ~150 (T20)
        margin = max(1, int(8 + abs(pa - pb) * 40))
        runs_for[winner] += 150 + margin; runs_against[winner] += 150
        runs_for[loser] += 150; runs_against[loser] += 150 + margin
        head2head[(winner, loser)] += 1

    # compute NRR
    nrr = {}
    for t in teams:
        if matches_played[t] == 0:
            nrr[t] = -999.0
        else:
            nrr[t] = (runs_for[t]/matches_played[t]) - (runs_against[t]/matches_played[t])

    # sort standings: pts desc, nrr desc, head-to-head, deterministic rng shuffle for final tie-break
    # initial sort by pts & nrr
    standings = sorted(teams, key=lambda t: (points[t], nrr[t]), reverse=True)

    # handle exact ties by checking head-to-head within tie groups
    final_sorted = []
    i = 0
    while i < len(standings):
        group = [standings[i]]
        j = i+1
        # collect tied group (same pts and nearly same nrr)
        while j < len(standings) and points[standings[j]] == points[standings[i]] and abs(nrr[standings[j]] - nrr[standings[i]]) < 1e-6:
            group.append(standings[j]); j += 1
        if len(group) == 1:
            final_sorted.extend(group)
        else:
            # use head-to-head wins within group
            hh_scores = []
            for t in group:
                wins = sum(head2head.get((t, o), 0) for o in group if o != t)
                hh_scores.append((t, wins))
            hh_sorted = sorted(hh_scores, key=lambda x: x[1], reverse=True)
            # if still tie, deterministic order by name
            final_sorted.extend([t for t,_ in hh_sorted])
        i = j

    # print points table
    print("\n-- Points Table (sorted) --")
    for rank, team in enumerate(final_sorted, start=1):
        print(f"{rank}. {team} - {points[team]} pts  (NRR: {nrr[team]:.2f})")

    top4 = final_sorted[:4]
    print("\nTop 4:", top4)

    # Playoffs: Q1: 1 vs 2 ; Eliminator: 3 vs 4 ; Q2: Loser Q1 vs Winner Eliminator ; Final: Winner Q1 vs Winner Q2
    A, B, C, D = top4
    venues_for_playoffs = [v for v in venue_encoding.keys()] if venue_encoding else [league_fixtures[0][2] if league_fixtures else None]
    final_venue = random.choice(venues_for_playoffs) if venues_for_playoffs else None

    def playoff(a, b, venue_for):
        toss_winner = random.choice([a, b])
        p_bat = p_bat_by_venue.get(venue_for, p_bat_global)
        toss_decision = "bat" if random.random() < p_bat else "field"
        feat = make_feature_df(a, b, toss_winner, toss_decision, venue_for)
        proba = model.predict_proba(feat)[0]
        class_idx = list(model.classes_)
        a_idx = team_encoding.get(a); b_idx = team_encoding.get(b)
        pa = proba[class_idx.index(a_idx)] if a_idx in class_idx else 0.5
        pb = proba[class_idx.index(b_idx)] if b_idx in class_idx else 0.5
        return a if random.random() < pa/(pa+pb) else b

    q1_winner = playoff(A, B, final_venue)
    q1_loser = B if q1_winner == A else A
    elim_winner = playoff(C, D, final_venue)
    q2_winner = playoff(q1_loser, elim_winner, final_venue)

    finalists = (q1_winner, q2_winner)
    champion = playoff(q1_winner, q2_winner, final_venue)

    print("\n-- Playoffs --")
    print("Qualifier1 Winner:", q1_winner)
    print("Qualifier1 Loser:", q1_loser)
    print("Eliminator Winner:", elim_winner)
    print("Qualifier2 Winner:", q2_winner)
    print("Finalists:", finalists)
    print("Tournament Winner:", champion)
    print("Final Venue:", final_venue)

# ---------------- INTERACTIVE MENU ----------------
def menu():
    print("Choose an option:")
    print("1 - Predict a single match (custom teams allowed, including legacy teams)")
    print("2 - Run one full tournament simulation (league + playoffs) [uses schedule league fixtures]")
    print("3 - Exit")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        # For prediction, allow any team names present in historical data (including legacy)
        all_known = sorted(set(list(history_names) + list(CURRENT_TEAMS.values())))
        for i, t in enumerate(all_known):
            print(f"{i}: {t}")
        t1_idx = int(input("Team 1 index: ").strip())
        t2_idx = int(input("Team 2 index: ").strip())
        team1 = all_known[t1_idx]; team2 = all_known[t2_idx]
        # choose a venue from known venues
        venues = sorted(list(venue_encoding.keys()))
        for i, v in enumerate(venues):
            print(f"{i}: {v}")
        v_idx = int(input("Venue index: ").strip())
        venue = venues[v_idx]
        winner, meta = predict_single_match(team1, team2, venue)
        print(f"Predicted winner: {winner}  (P_team1_win: {meta['p_win_team1']:.2f})")
        print("Toss winner:", meta["toss_winner"], "| Toss decision:", meta["toss_decision"])
    elif choice == "2":
        run_one_tournament()
    else:
        print("Exiting.")

if __name__ == "__main__":
    menu()