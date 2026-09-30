SUPPORT_CHOICES = [
    {"name": "billing", "description": "Payments, charges, invoices, refunds, and subscription billing."},
    {"name": "technical_support", "description": "Crashes, bugs, integrations, and product errors."},
    {"name": "account_access", "description": "Sign-in, passwords, locked accounts, and authentication."},
    {"name": "other", "description": "None of the above; route for manual triage."},
]

TICKETS = [
    ("Double charge", "I was charged twice for the same subscription this month. Please refund the duplicate payment.", "billing"),
    ("App crash", "The desktop app crashes every time I try to export a PDF. I can sign in normally.", "technical_support"),
    ("Password reset", "I forgot my password and the reset email never arrives. I cannot sign in.", "account_access"),
    ("Partnership enquiry", "We would like to sponsor your upcoming developer conference. Who handles partnerships?", "other"),
    ("Wrong invoice", "My invoice lists the enterprise price, but I have the basic subscription. Please fix the bill.", "billing"),
    ("Broken integration", "Our integration returns error 500 on every export. The API credentials are valid.", "technical_support"),
    ("Lost authenticator", "I replaced my phone and lost the authenticator app. My password works but I cannot pass two-factor verification.", "account_access"),
    ("Press enquiry", "I am a journalist writing about your company and would like to speak with your press team.", "other"),
    ("Refund request", "I cancelled before renewal but you charged my card again. I need that payment returned.", "billing"),
    ("Unresponsive dashboard", "The dashboard freezes when I change the date filter. Refreshing does not help.", "technical_support"),
    ("Locked out", "My account was locked after too many password attempts. How do I regain access?", "account_access"),
    ("Workshop invitation", "Would your team like to give a talk at our local programming meetup next month?", "other"),
]

EXAMPLES = [
    {"id": f"support-{i+1}", "title": title, "state": state,
     "question": "Which team should handle this support ticket?",
     "choices": SUPPORT_CHOICES, "expected": expected}
    for i, (title, state, expected) in enumerate(TICKETS)
]
