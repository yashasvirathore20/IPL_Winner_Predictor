from flask import Flask, render_template, request
import core  # ML script

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/single", methods=["GET", "POST"])
def single():
    teams = sorted(core.CURRENT_TEAMS.values())

    venues = sorted(core.venue_encoding.keys())

    result = None
    if request.method == "POST":
        t1 = request.form.get("team1")
        t2 = request.form.get("team2")
        v = request.form.get("venue")
        winner, meta = core.predict_single_match(t1, t2, v)
        result = {
            "winner": winner,
            "prob": round(meta["p_win_team1"], 2),
            "toss": meta["toss_winner"],
            "decision": meta["toss_decision"]
        }

    return render_template("single.html", teams=teams, venues=venues, result=result)

@app.route("/tournament", methods=["GET", "POST"])
def tournament():
    output = None
    if request.method == "POST":
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            core.run_one_tournament()
        output = buffer.getvalue()

    return render_template("tournament.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)
