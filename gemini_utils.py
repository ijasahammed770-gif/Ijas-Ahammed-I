"""AI + recommendation logic for PocketSmart AI."""
import json
import os 
import time
from urllib.parse import quote_plus

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key or key.startswith("PASTE"):
            raise RuntimeError("GEMINI_API_KEY is missing in your .env file")
        _client = genai.Client(api_key=key)
    return _client


# ---------- product links (no scraping, just search links) ----------
def product_link(platform: str, query: str) -> str:
    q = quote_plus(query)
    p = platform.lower()
    if "amazon" in p:
        return f"https://www.amazon.in/s?k={q}"
    if "flipkart" in p:
        return f"https://www.flipkart.com/search?q={q}"
    if "ikea" in p:
        return f"https://www.ikea.com/in/en/search/?q={q}"
    return f"https://www.google.com/search?q={quote_plus(query + ' ' + platform)}"


# ---------- Gemini call ----------
SCHEMA_HINT = """
Return ONLY valid JSON in exactly this shape:
{
  "summary": "one or two sentences",
  "budget_breakdown": [{"category": "string", "amount": number}],
  "recommendations": [
    {"name": "string", "category": "string", "platform": "string",
     "price_inr": number, "reason": "short reason"}
  ]
}
Rules: prices in Indian Rupees (INR); total of price_inr must not exceed the budget;
give 5 to 8 recommendations; platform must be a real one such as Amazon, Flipkart,
IKEA, Swiggy, Zomato or OYO.
"""


def ask_gemini(prompt: str, image_bytes: bytes = None, mime_type: str = None) -> dict:
    contents = [prompt + "\n" + SCHEMA_HINT]
    if image_bytes:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg"))

    response = get_client().models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )
    text = (response.text or "").replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def finish(data: dict) -> dict:
    """Add shopping links to every recommendation."""
    for item in data.get("recommendations", []):
        item["link"] = product_link(item.get("platform", ""), item.get("name", ""))
    return data


# ---------- fallback if the AI fails or returns nothing ----------
def fallback(budget: float, parts: dict, platform: str, note: str) -> dict:
    breakdown, recs = [], []
    for category, share in parts.items():
        amount = round(budget * share)
        breakdown.append({"category": category, "amount": amount})
        recs.append({
            "name": f"Best {category} option",
            "category": category,
            "platform": platform,
            "price_inr": amount,
            "reason": "Default suggestion (AI unavailable).",
        })
    return finish({"summary": note, "budget_breakdown": breakdown, "recommendations": recs})


def safe_run(prompt, budget, parts, platform, image_bytes=None, mime_type=None):
    try:
        data = ask_gemini(prompt, image_bytes, mime_type)
        if not data.get("recommendations"):
            raise ValueError("Empty recommendations")
        return finish(data)
    except Exception as e:  # keep the app working, but show the reason in the terminal
        print("GEMINI ERROR:", repr(e))
        return fallback(budget, parts, platform, f"Showing default plan. Reason: {e}")


# ---------- the three planners ----------
def home_plan(budget, room, items):
    prompt = (f"You are a home interior budget planner in India. Budget: INR {budget}. "
              f"Room: {room}. Items needed (with quantities): {items}. "
              "Recommend cost-effective products from IKEA, Amazon and Flipkart, "
              "balanced across function, style and price.")
    parts = {"Furniture": 0.5, "Lighting": 0.2, "Decor": 0.3}
    return safe_run(prompt, budget, parts, "Amazon")


def party_plan(budget, guests, event, venue):
    prompt = (f"You are a party budget planner in India. Budget: INR {budget}. Guests: {guests}. "
              f"Event type: {event}. Venue details: {venue}. Split the budget across catering, "
              "decoration and entertainment (and stay/venue if needed). Use Swiggy, Zomato, "
              "OYO, Amazon and Flipkart as sources.")
    parts = {"Catering": 0.5, "Decoration": 0.25, "Entertainment": 0.15, "Venue": 0.10}
    return safe_run(prompt, budget, parts, "Zomato")


def jewelry_plan(budget, occasion, style, image_bytes=None, mime_type=None):
    prompt = (f"You are a jewelry stylist in India. Budget: INR {budget}. Occasion: {occasion}. "
              f"Style preference: {style}. "
              + ("Match colours and style to the attached outfit image. " if image_bytes else "")
              + "Recommend jewelry from Amazon and Flipkart.")
    parts = {"Necklace": 0.4, "Earrings": 0.3, "Bracelet/Ring": 0.3}
    return safe_run(prompt, budget, parts, "Amazon", image_bytes, mime_type)