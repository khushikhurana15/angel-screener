import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_rule_based_explanation(trade):
    reasons = []

    if trade["smma_gap_pct"] > 0.5:
        reasons.append("SMMA gap strong hai, momentum clear dikh raha hai")
    else:
        reasons.append("SMMA gap chhota hai, signal weak ho sakta hai")

    if trade["ltq_ratio_2v5"] > 1.2:
        reasons.append("recent LTQ mein spike dikha, activity badhi hai")
    else:
        reasons.append("LTQ mein koi bada spike nahi dikha")

    verdict = "Accept karne layak" if trade["ml_prediction"] == "Profitable" else "Avoid karna behtar"
    return f"{verdict} — {', '.join(reasons)}."


def generate_ai_explanation(trade):
    prompt = f"""You are a trading analyst. Explain in exactly 1-2 short sentences whether 
this SMMA crossover trade signal should be accepted or avoided, based on this data. 

Write ONLY in clear, simple English. Do not use Hindi or Hinglish. 
Do not add any parenthetical translations. Just one clean English explanation.

Symbol: {trade['symbol']}
Signal: {trade['signal']}
SMMA Gap %: {trade['smma_gap_pct']}
LTQ Ratio (2min vs 5min): {trade['ltq_ratio_2v5']}
Volatility: {trade['volatility']}
ML Prediction: {trade['ml_prediction']}
Confidence: {trade['confidence']}%

Be direct and specific - mention the actual reason (SMMA gap strength, LTQ spike, etc), 
not generic statements."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            timeout=8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Groq failed, using fallback: {e}")
        return generate_rule_based_explanation(trade)


sample_trade = {
    "symbol": "MOMENTUM-EQ",
    "signal": "BUY",
    "smma_gap_pct": 0.85,
    "ltq_ratio_2v5": 1.45,
    "volatility": 0.32,
    "ml_prediction": "Profitable",
    "confidence": 78.5,
}

print("🤖 AI Explanation:")
print(generate_ai_explanation(sample_trade))

print("\n📏 Rule-based Fallback (for comparison):")
print(generate_rule_based_explanation(sample_trade))