"""Seed data for LegacyCore, the mock core-banking target application.

Everything in this file is fake. Member numbers double as switches for the
runtime conditions the automation has to cope with (see CONDITIONS); that is
how the evidence runs inject "not found", "system busy", a 500, and a screen
that needs a human, deterministically.
"""

# Teller credentials for the mock. Documented in README; passed to runs as a
# *sensitive* input so they never land in artifacts or logs.
USERS = {"teller1": "Passw0rd!"}

# Only a supervisor knows this. It is deliberately NOT an input of any
# capability; member 31337 exists to force a human handoff.
SUPERVISOR_OVERRIDE_CODE = "4242"

# (acct_no, type, current balance, available balance, status)
MEMBERS = {
    "12345": {
        "name": "Jane Q. Member", "status": "Active", "since": "03/14/2011",
        "accounts": [
            ("0012345-S1", "SAV", "2,340.55", "2,340.55", "Open"),
            ("0012345-C1", "CHK", "812.10", "762.10", "Open"),
            ("0012345-L1", "LN", "-4,500.00", "0.00", "Open"),
        ],
    },
    "23456": {
        "name": "Marcus T. Ortega", "status": "Active", "since": "11/02/2018",
        "accounts": [
            ("0023456-C1", "CHK", "1,105.00", "1,105.00", "Open"),
            ("0023456-S1", "SAV", "15,020.31", "15,020.31", "Open"),
        ],
    },
    "40404": {
        "name": "Priya N. Raman", "status": "Active", "since": "06/30/2015",
        "accounts": [
            ("0040404-S1", "SAV", "980.00", "980.00", "Open"),
            ("0040404-C1", "CHK", "42.17", "42.17", "Open"),
        ],
    },
    "30303": {
        "name": "Dmitri K. Volkov", "status": "Active", "since": "01/09/2020",
        "accounts": [
            ("0030303-S1", "SAV", "5,555.55", "5,555.55", "Open"),
        ],
    },
    "20202": {
        "name": "Aisha B. Nwosu", "status": "Active", "since": "09/21/2009",
        "accounts": [
            ("0020202-C1", "CHK", "300.00", "300.00", "Open"),
            ("0020202-S1", "SAV", "7,000.00", "6,500.00", "Open"),
        ],
    },
    "31337": {
        "name": "Leo P. Castellanos", "status": "Active", "since": "04/01/2013",
        "accounts": [
            ("0031337-S1", "SAV", "12.00", "12.00", "Open"),
        ],
    },
    "50500": {
        "name": "Hannah S. Yoo", "status": "Active", "since": "07/07/2017",
        "accounts": [
            ("0050500-S1", "SAV", "640.00", "640.00", "Open"),
        ],
    },
}

# member number -> injected runtime condition
CONDITIONS = {
    "99999": "not_found",            # legitimate business outcome
    "77777": "restricted",           # permission denial (business outcome)
    "40404": "busy_once",            # interstitial, clears on retry
    "30303": "slow",                 # 4s delay before the page renders
    "20202": "confirm_dialog",       # window.confirm() on sub-account submit
    "50500": "session_expired",      # session dropped -> back to sign-on
    "31337": "supervisor_override",  # needs a human with the override code
    "50000": "app_error",            # HTTP 500
}

PRODUCTS = [("SAV", "Savings"), ("CLB", "Club Savings"), ("MMA", "Money Market")]
