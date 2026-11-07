"""
Deposit Recovery AI Lawyer (India) - Kuku AI Generalist Assignment
Domain: Residential Landlord Security Deposit Recovery in India ONLY.

Run locally:   streamlit run deposit_ai_lawyer.py
Secrets:       ANTHROPIC_API_KEY or OPENAI_API_KEY (env var, .streamlit/secrets.toml, or sidebar)
"""
import os
from datetime import date, timedelta

import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Deposit Recovery AI Lawyer (India)", page_icon="⚖️", layout="centered")

DISCLAIMER = (
    "**⚠️ Important:** This tool provides legal *awareness and self-advocacy assistance* only. "
    "It is **not** a law firm, does **not** provide formal legal representation, and no "
    "advocate-client relationship is created. Laws vary by state and facts matter. "
    "If the dispute escalates (no reply to notice, court/consumer filing, threats), "
    "please consult a qualified human advocate."
)

# Facts the model is ALLOWED to rely on. Everything else must be hedged.
SAFE_LAW_CONTEXT = """
ALLOWED LEGAL BASIS (use only this, in plain language):
- General contract principles under the Indian Contract Act, 1872: a security deposit is held under the
  rental agreement; the landlord may deduct only for genuine, provable dues (unpaid rent, utility bills,
  damage beyond normal wear and tear) and must return the balance when possession is handed back.
  Wrongfully keeping money owed can be treated as breach of contract; the tenant can claim the amount
  and may ask for reasonable interest/compensation.
- The Model Tenancy Act, 2021 is a CENTRAL MODEL LAW. It only binds a state if that state has adopted or
  enacted it. Its general ideas: a residential deposit is meant to be capped at about two months' rent,
  and it should be refunded when vacant possession is handed over after lawful deductions.
  Always tell the user to check whether their state has adopted it and what the state rent law says.
- Typical practical path: written demand notice -> negotiation/mediation -> civil suit for recovery of money
  (or the state's rent authority/tribunal where one exists). Limitation for money claims is generally about
  three years; tell the user to verify with an advocate.
- Normal wear and tear (minor paint fading, small nail holes, ageing) is generally not a valid deduction.
HARD RULES:
- NEVER cite specific section numbers, case names, judgments, citations, or municipal/local bylaws.
  If asked, say you cannot verify them and recommend a human advocate or the official state rent law text.
- If unsure, say "I am not certain; please confirm with an advocate."
- Do not guarantee outcomes. Use words like "may", "generally", "likely".
- Plain simple English, minimal jargon; explain any legal word you must use.
- Stay strictly within residential security deposit recovery in India. Politely decline other legal topics.
"""

EVAL_PROMPT = f"""You are a careful legal-awareness assistant for tenants in India (NOT a lawyer, NOT legal representation).
Explain the user's standing for their security deposit dispute in simple language.
{SAFE_LAW_CONTEXT}
Output format (use short headings):
1. Where you stand (strong / moderate / weak, with honest reasoning)
2. What the landlord can and cannot deduct, applied to THEIR stated reason
3. Your evidence: what helps, what is missing and how to get it
4. Likely next steps and rough timeline (no guarantees)
5. When to hire a human advocate
End with one line reminding that this is awareness, not legal advice."""


def roleplay_prompt(c: dict) -> str:
    return f"""You are role-playing a STUBBORN but realistic Indian residential landlord so a tenant can practise negotiating.
Case facts: tenant {c['tenant_name']} lived at the property from {c['move_in']} to {c['move_out']}.
Total deposit: Rs {c['deposit']:,.0f}. Withheld: Rs {c['withheld']:,.0f}. Your stated reason: "{c['reason'] or 'none given, you just stall'}".
Behaviour: be dismissive, delay, exaggerate damage, say "market practice", and counter with partial refunds.
Concede ONLY if the tenant makes calm, specific, evidence-based points (agreement clause, photos, receipts, written notice).
Rules: short replies (2-4 sentences), natural Indian English, never abusive, no threats or illegal acts, never invent
laws, sections or cases. If the tenant cites a law you cannot verify, respond sceptically ("show me where it says that").
Stay in character; do not give coaching while in character."""


COACH_PROMPT = f"""You are a negotiation coach for a tenant in India. Review the transcript of the practice negotiation
(landlord = stubborn roleplay). Give: 3 things the tenant did well, 3 things to improve, and 2 stronger sample replies.
Praise evidence-based, calm arguments. Flag any legal claim that sounded unverified or overconfident.
{SAFE_LAW_CONTEXT}"""

# ----------------------------------------------------------------------------
# LLM LAYER (Anthropic or OpenAI)
# ----------------------------------------------------------------------------
def get_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:
        return os.getenv(name, "")


def call_llm(system: str, messages: list, max_tokens: int = 1200, temperature: float = 0.4) -> str:
    provider = st.session_state.get("provider", "Gemini (free tier)")
    key = st.session_state.get("api_key") or {
        "Anthropic": lambda: get_secret("ANTHROPIC_API_KEY"),
        "OpenAI": lambda: get_secret("OPENAI_API_KEY"),
        "Gemini (free tier)": lambda: get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY"),
    }[provider]()
    if not key:
        return "⚠️ No API key found. Add one in the sidebar or in Streamlit secrets."
    try:
        if provider == "Anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            kwargs = dict(
                model=st.session_state.get("model", "claude-sonnet-4-6"),
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            try:
                r = client.messages.create(temperature=temperature, **kwargs)
            except Exception as e:
                # Some SDK versions/models reject `temperature`; retry without it.
                if "temperature" in str(e).lower():
                    r = client.messages.create(**kwargs)
                else:
                    raise
            return "".join(b.text for b in r.content if b.type == "text")
        from openai import OpenAI

        if provider == "Gemini (free tier)":
            client = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            max_tokens *= 3  # Gemini 2.5 "thinking" tokens count toward the limit; leave headroom
        else:
            client = OpenAI(api_key=key)
        kwargs = dict(
            model=st.session_state.get("model", "gpt-4o"),
            messages=[{"role": "system", "content": system}] + messages,
        )
        try:
            r = client.chat.completions.create(
    max_tokens=max_tokens,
    temperature=temperature,
    extra_body={
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinking_budget": 0
                }
            }
        }
    },
    **kwargs
)
            # r = client.chat.completions.create(max_tokens=max_tokens, temperature=temperature, **kwargs)
            


        except Exception as e:
            # Newer OpenAI models reject max_tokens/temperature; retry with compatible params.
            if "temperature" in str(e).lower() or "max_tokens" in str(e).lower():
                r = client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
            else:
                raise
        return r.choices[0].message.content
    except Exception as e:  # surface errors in UI instead of crashing
        return f"⚠️ LLM error: {e}"


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("step", 1)
ss.setdefault("case", None)
ss.setdefault("evaluation", None)
ss.setdefault("chat", [])
ss.setdefault("coach", None)
ss.setdefault("ack", False)

STEPS = ["1. Case Intake", "2. Legal Evaluation", "3. Demand Notice", "4. Negotiation Practice"]


def fmt_date(d: date) -> str:
    return d.strftime("%d %B %Y")


def case_summary(c: dict) -> str:
    return (
        f"Tenant: {c['tenant_name']} | Landlord: {c['landlord_name']} | Property: {c['address']}\n"
        f"Monthly rent: Rs {c['rent']:,.0f} | Stay: {fmt_date(c['move_in'])} to {fmt_date(c['move_out'])} "
        f"({c['days']} days, ~{c['days'] // 30} months)\n"
        f"Deposit paid: Rs {c['deposit']:,.0f} | Withheld: Rs {c['withheld']:,.0f} "
        f"| Returned already: Rs {c['deposit'] - c['withheld']:,.0f}\n"
        f"Days since move-out: {c['since_moveout']}\n"
        f"Landlord's reason: {c['reason'] or 'None given'}\n"
        f"Proof available: {', '.join(c['proof']) or 'None listed'}\n"
        f"Written agreement says about deposit/deductions: {c['clause'] or 'Not provided'}"
    )


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    ss.provider = st.selectbox("LLM provider", ["Gemini (free tier)", "Anthropic", "OpenAI"])
    DEFAULT_MODELS = {"Gemini (free tier)": "gemini-2.5-flash", "Anthropic": "claude-sonnet-4-6", "OpenAI": "gpt-4o"}
    ss.model = st.text_input("Model", DEFAULT_MODELS[ss.provider], key=f"model_{ss.provider}")
    ss.api_key = st.text_input("API key (optional if in secrets)", type="password")
    st.divider()
    st.subheader("Progress")
    for i, s in enumerate(STEPS, 1):
        st.write(("✅ " if i < ss.step else "👉 " if i == ss.step else "⬜ ") + s)
    if st.button("🔄 Start over"):
        for k in ["step", "case", "evaluation", "chat", "coach"]:
            ss.pop(k, None)
        st.rerun()

st.title("⚖️ Security Deposit Recovery Assistant")
st.caption("Residential landlord security deposits in India — awareness & self-advocacy tool")
st.warning(DISCLAIMER)

if not ss.ack:
    ss.ack = st.checkbox("I understand this is not legal representation or formal legal advice.")
    if not ss.ack:
        st.stop()

# ----------------------------------------------------------------------------
# STEP 1: INTAKE
# ----------------------------------------------------------------------------
if ss.step == 1:
    st.subheader("Step 1: Tell me about your case")
    with st.form("intake"):
        c1, c2 = st.columns(2)
        tenant_name = c1.text_input("Your full name")
        landlord_name = c2.text_input("Landlord's full name")
        address = st.text_input("Rented property address")
        c3, c4 = st.columns(2)
        move_in = c3.date_input("Move-in date", value=date.today() - timedelta(days=400))
        move_out = c4.date_input("Move-out date", value=date.today() - timedelta(days=30))
        c5, c6, c7 = st.columns(3)
        rent = c5.number_input("Monthly rent (Rs)", min_value=0.0, step=500.0)
        deposit = c6.number_input("Total deposit paid (Rs)", min_value=0.0, step=1000.0)
        withheld = c7.number_input("Amount unfairly withheld (Rs)", min_value=0.0, step=1000.0)
        reason = st.text_area("Landlord's reason for withholding (leave blank if none given)")
        clause = st.text_area("What does your agreement say about deposit refund/deductions? (optional)")
        proof = st.multiselect(
            "Proof you have",
            ["Rental agreement", "Chat/WhatsApp screenshots", "Payment receipts / bank statements",
             "Move-in/move-out photos or video", "Utility bill receipts", "Email/written notice of vacating",
             "None of these"],
        )
        ok = st.form_submit_button("Save & continue →")

    if ok:
        errs = []
        if not tenant_name or not landlord_name or not address:
            errs.append("Please fill your name, landlord's name and the property address.")
        if move_out < move_in:
            errs.append("Move-out date cannot be before move-in date.")
        if deposit <= 0 or withheld <= 0 or withheld > deposit:
            errs.append("Deposit and withheld amounts must be positive, and withheld cannot exceed the deposit.")
        for e in errs:
            st.error(e)
        if not errs:
            ss.case = dict(
                tenant_name=tenant_name, landlord_name=landlord_name, address=address,
                move_in=move_in, move_out=move_out, rent=rent, deposit=deposit, withheld=withheld,
                reason=reason.strip(), clause=clause.strip(), proof=proof,
                days=(move_out - move_in).days, since_moveout=(date.today() - move_out).days,
            )
            ss.step = 2
            st.rerun()

# ----------------------------------------------------------------------------
# STEP 2: EVALUATION
# ----------------------------------------------------------------------------
elif ss.step == 2:
    c = ss.case
    st.subheader("Step 2: Where you stand")
    st.code(case_summary(c), language=None)

    # Deterministic red/green flags (no LLM needed, no hallucination risk)
    if c["rent"] and c["deposit"] > 2 * c["rent"]:
        st.info("Your deposit is more than 2 months' rent. The Model Tenancy Act, 2021 suggests a cap of about "
                "2 months for residential homes, but it only applies if your state has adopted it.")
    if c["since_moveout"] > 30:
        st.info(f"It has been {c['since_moveout']} days since you moved out — a delay this long strengthens your "
                "case if the landlord has no valid reason.")
    if c["since_moveout"] > 365 * 2:
        st.warning("Money claims generally have a time limit (usually around 3 years). Act soon and verify with an advocate.")
    if not c["proof"] or c["proof"] == ["None of these"]:
        st.warning("You listed no proof. Try recovering chats, bank statements and any photos before sending a notice.")

    if ss.evaluation is None:
        with st.spinner("Analysing your case..."):
            result = call_llm(EVAL_PROMPT, [{"role": "user", "content": case_summary(c)}], max_tokens=800)
        if result.startswith("⚠️"):
            st.error(result)  # show the error but don't cache it, so a fixed key/model can retry
            if st.button("🔁 Retry"):
                st.rerun()
        else:
            ss.evaluation = result
    if ss.evaluation:
        st.markdown(ss.evaluation)

    a, b = st.columns(2)
    if a.button("← Edit details"):
        ss.step, ss.evaluation = 1, None
        st.rerun()
    if b.button("Generate demand notice →"):
        ss.step = 3
        st.rerun()

# ----------------------------------------------------------------------------
# STEP 3: NOTICE (deterministic template, zero invented law)
# ----------------------------------------------------------------------------
elif ss.step == 3:
    c = ss.case
    st.subheader("Step 3: Your demand notice")
    days_to_pay = st.slider("Days you will give the landlord to pay", 7, 30, 15)
    interest = st.checkbox("Reserve the right to claim reasonable interest/compensation", value=True)

    returned = c["deposit"] - c["withheld"]
    reason_line = (
        f'You have stated that the amount was withheld because: "{c["reason"]}". '
        "No itemised statement or supporting proof of such deductions has been provided to me."
        if c["reason"] else
        "No reason or itemised statement of deductions has been given to me for retaining this amount."
    )
    interest_line = (
        "Failing this, I reserve my right to claim reasonable interest and compensation along with the amount due, "
        "and to recover the costs of proceedings. "
        if interest else ""
    )
    notice = f"""Date: {fmt_date(date.today())}

To,
{c['landlord_name']}
(Landlord of {c['address']})

From,
{c['tenant_name']}

Subject: Demand Notice for Return of Security Deposit of Rs {c['withheld']:,.0f}

Dear {c['landlord_name']},

1. I was a tenant at {c['address']} from {fmt_date(c['move_in'])} to {fmt_date(c['move_out'])}, under the rental arrangement between us{f", at a monthly rent of Rs {c['rent']:,.0f}" if c['rent'] else ""}.

2. At the start of the tenancy, I paid a refundable security deposit of Rs {c['deposit']:,.0f}. I vacated and handed back possession of the premises on {fmt_date(c['move_out'])}.

3. Of this deposit, Rs {returned:,.0f} has been returned, and Rs {c['withheld']:,.0f} remains unpaid to date.

4. {reason_line}

5. A security deposit is held only to cover genuine, proven dues such as unpaid rent, utility bills or damage beyond normal wear and tear. I request that any deduction you claim be supported by an itemised statement and proof.

6. I therefore request you to pay Rs {c['withheld']:,.0f} to me within {days_to_pay} days of receiving this notice, by bank transfer to the account details I will share on request.

7. {interest_line}If the amount is not paid within this period, I may be left with no option but to pursue further remedies available under law, including approaching the appropriate legal forum.

I would prefer to resolve this amicably and request your cooperation.

Yours sincerely,
{c['tenant_name']}
Date: {fmt_date(date.today())}
Place: ____________
Contact: ____________  (phone / email)

Enclosures (attach copies): {', '.join(p for p in c['proof'] if p != 'None of these') or 'Rental agreement, payment proof, relevant chats'}
"""
    st.text_area("Copy your notice", notice, height=520)
    st.download_button("⬇️ Download as .txt", notice, file_name="deposit_demand_notice.txt")
    st.info("**Before sending:** send by Speed Post/Registered Post AD **and** email/WhatsApp, and keep the postal "
            "receipt. Do not threaten or use abusive language. If there is no reply, speak to a human advocate "
            "about next steps.")

    a, b = st.columns(2)
    if a.button("← Back to evaluation"):
        ss.step = 2
        st.rerun()
    if b.button("Practise negotiating →"):
        ss.step = 4
        st.rerun()

# ----------------------------------------------------------------------------
# STEP 4: ROLEPLAY
# ----------------------------------------------------------------------------
elif ss.step == 4:
    c = ss.case
    st.subheader("Step 4: Practise against a stubborn landlord")
    st.caption("The AI plays your landlord. Try staying calm, specific and evidence-based. This is only practice.")

    BEGIN = {"role": "user", "content": "Begin: open the conversation as the landlord responding to the tenant's demand notice."}
    if not ss.chat:
        with st.spinner("Landlord is typing..."):
            opener = call_llm(roleplay_prompt(c), [BEGIN], max_tokens=300, temperature=0.9)
        ss.chat.append({"role": "assistant", "content": opener})

    for m in ss.chat:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user",
                             avatar="🏠" if m["role"] == "assistant" else "🧑"):
            st.write(m["content"])

    if msg := st.chat_input("Your reply to the landlord..."):
        ss.chat.append({"role": "user", "content": msg})
        with st.spinner("Landlord is typing..."):
            reply = call_llm(roleplay_prompt(c), [BEGIN] + ss.chat, max_tokens=300, temperature=0.9)
        ss.chat.append({"role": "assistant", "content": reply})
        st.rerun()

    a, b, d = st.columns(3)
    if a.button("🎓 Get coaching feedback") and len(ss.chat) > 2:
        transcript = "\n".join(f"{'LANDLORD' if m['role']=='assistant' else 'TENANT'}: {m['content']}" for m in ss.chat)
        with st.spinner("Reviewing your negotiation..."):
            ss.coach = call_llm(COACH_PROMPT, [{"role": "user", "content": case_summary(c) + "\n\nTRANSCRIPT:\n" + transcript}], max_tokens=1200)
    if b.button("🔁 Restart roleplay"):
        ss.chat, ss.coach = [], None
        st.rerun()
    if d.button("← Back to notice"):
        ss.step = 3
        st.rerun()
    if ss.coach:
        st.markdown("### Coach's feedback")
        st.markdown(ss.coach)

st.divider()
st.caption("Legal awareness tool only. Not legal advice or representation. Consult a human advocate if the dispute escalates.")
