"""LegacyCore: a deliberately hostile mock core-banking web app.

Design constraints (on purpose):
- frameset layout: a "menu" frame and a "work" frame. A top-document-only
  perceiver sees nothing useful.
- server-rendered, table layout, <font> tags, no ids, no data-* attributes,
  no ARIA, cryptic form field names, generic class names.
- buttons are <input type=submit>, some links are javascript: links.
- the word "Search" appears as a button in BOTH frames (ambiguity).
- runtime conditions are injected per member number (see data.CONDITIONS).

Run: hands serve-mock   (binds 127.0.0.1:5050, no reloader)
"""
from __future__ import annotations

import time

from flask import Flask, abort, redirect, render_template, request, session, url_for

from . import data

app = Flask(__name__)
app.secret_key = "legacycore-mock-session-key-not-a-secret"


def _logged_in() -> bool:
    return session.get("uid") is not None


def _require_login():
    if not _logged_in():
        session["notice"] = "Please sign on."
        return redirect(url_for("main"))
    return None


@app.route("/")
def frameset():
    return render_template("frameset.html")


@app.route("/nav")
def nav():
    return render_template("nav.html")


@app.route("/main", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        uid = request.form.get("uid", "").strip()
        pwd = request.form.get("pwd", "")
        if data.USERS.get(uid) == pwd:
            session.clear()
            session["uid"] = uid
            return redirect(url_for("lookup"))
        return render_template("login.html", error="Invalid user ID or password.")
    if _logged_in():
        return redirect(url_for("lookup"))
    return render_template("login.html", notice=session.pop("notice", None))


@app.route("/signoff")
def signoff():
    session.clear()
    session["notice"] = "You have been signed off."
    return redirect(url_for("main"))


@app.route("/lookup")
def lookup():
    r = _require_login()
    if r:
        return r
    return render_template("lookup.html")


@app.route("/member")
def member():
    r = _require_login()
    if r:
        return r
    mno = request.args.get("mno", "").strip()
    if not mno:
        return render_template("lookup.html", error="Member number is required.")
    cond = data.CONDITIONS.get(mno)
    if cond == "app_error":
        abort(500)
    if cond == "session_expired":
        session.clear()
        session["notice"] = "Your session has expired. Please sign on again."
        return redirect(url_for("main"))
    if cond == "restricted":
        return render_template("restricted.html", mno=mno)
    if cond == "not_found" or mno not in data.MEMBERS:
        return render_template("notfound.html", mno=mno)
    if cond == "busy_once":
        key = f"busy_seen_{mno}"
        if not session.get(key):
            session[key] = True
            return render_template("busy.html", mno=mno)
    if cond == "slow":
        time.sleep(4)
    return render_template("member.html", mno=mno, m=data.MEMBERS[mno])


@app.route("/subaccount/new")
def sub_new():
    r = _require_login()
    if r:
        return r
    mno = request.args.get("mno", "").strip()
    if mno not in data.MEMBERS:
        return render_template("notfound.html", mno=mno)
    return render_template(
        "sub_new.html", mno=mno, m=data.MEMBERS[mno], products=data.PRODUCTS,
        confirm_dialog=(data.CONDITIONS.get(mno) == "confirm_dialog"),
    )


@app.route("/subaccount/review", methods=["POST"])
def sub_review():
    r = _require_login()
    if r:
        return r
    mno = request.form.get("mno", "").strip()
    acct_typ = request.form.get("acct_typ", "SAV")
    nick = request.form.get("nick", "").strip()
    if mno not in data.MEMBERS:
        return render_template("notfound.html", mno=mno)
    if not nick:
        return render_template(
            "sub_new.html", mno=mno, m=data.MEMBERS[mno], products=data.PRODUCTS,
            confirm_dialog=(data.CONDITIONS.get(mno) == "confirm_dialog"),
            error="Nickname is required.", acct_typ=acct_typ,
        )
    return render_template(
        "sub_review.html", mno=mno, m=data.MEMBERS[mno], acct_typ=acct_typ, nick=nick,
        product_name=dict(data.PRODUCTS).get(acct_typ, acct_typ),
        needs_override=(data.CONDITIONS.get(mno) == "supervisor_override"),
    )


@app.route("/subaccount/confirm", methods=["POST"])
def sub_confirm():
    r = _require_login()
    if r:
        return r
    mno = request.form.get("mno", "").strip()
    acct_typ = request.form.get("acct_typ", "SAV")
    nick = request.form.get("nick", "").strip()
    if mno not in data.MEMBERS:
        return render_template("notfound.html", mno=mno)
    needs_override = data.CONDITIONS.get(mno) == "supervisor_override"
    if needs_override and request.form.get("sup", "") != data.SUPERVISOR_OVERRIDE_CODE:
        return render_template(
            "sub_review.html", mno=mno, m=data.MEMBERS[mno], acct_typ=acct_typ, nick=nick,
            product_name=dict(data.PRODUCTS).get(acct_typ, acct_typ),
            needs_override=True,
            error="Supervisor override code required to open this sub-account.",
        )
    n = session.get("sa_counter", 0) + 1
    session["sa_counter"] = n
    conf = f"SA-{mno}-{n:04d}"
    return render_template(
        "sub_done.html", mno=mno, conf=conf, nick=nick,
        product_name=dict(data.PRODUCTS).get(acct_typ, acct_typ),
    )


@app.errorhandler(500)
def internal_error(_e):
    return render_template("error500.html"), 500


def run(host: str = "127.0.0.1", port: int = 5050) -> None:
    # No reloader: it re-execs the CLI process. No debug: the debugger page
    # would replace our legacy 500 page.
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    run()
