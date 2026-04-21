import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from app.config import get_settings
from app.services import session_store

_IMAGES_DIR = Path(__file__).resolve().parent.parent / "static" / "images"


def _image_data_uri(filename: str) -> str:
    """Return a base64 data URI for an image file, or empty string if missing."""
    path = _IMAGES_DIR / filename
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lstrip(".")
    return f"data:image/{ext};base64,{data}"


def _render_note_list(title: str, notes: list[str]) -> str:
    if not notes:
        return ""
    items = "".join(f"<li>{note}</li>" for note in notes)
    return f"""
      <div class="note-group">
        <h3>{title}</h3>
        <ul>{items}</ul>
      </div>"""


def _build_html(formula: dict, inline_images: bool = False) -> str:
    """Build the mail HTML. Use inline_images=True for PDF and email (embeds image as base64)."""
    profile = formula.get("profile", "")
    description = formula.get("description", "")
    top_notes = formula.get("top_notes", [])
    heart_notes = formula.get("heart_notes", [])
    base_notes = formula.get("base_notes", [])

    notes_html = (
        _render_note_list("Notes de tête", top_notes)
        + _render_note_list("Notes de cœur", heart_notes)
        + _render_note_list("Notes de fond", base_notes)
    )

    if inline_images:
        img_src = _image_data_uri("pyramide.png")
    else:
        img_src = "/static/images/pyramide.png"

    img_tag = f'<img src="{img_src}" alt="Pyramide olfactive" />' if img_src else ""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Votre formule — {profile}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      padding: 40px;
      max-width: 820px;
      margin: 0 auto;
      color: #333;
    }}
    .intro {{
      margin-bottom: 24px;
    }}
    .intro p {{
      margin: 4px 0;
      font-size: 1rem;
    }}
    .profile-name {{
      font-size: 1.3rem;
      font-weight: bold;
      margin: 0 0 4px;
    }}
    .description {{
      color: #666;
      font-style: italic;
      font-size: 0.95rem;
    }}
    .recap {{
      display: flex;
      align-items: flex-start;
      gap: 40px;
    }}
    .pyramid {{
      flex: 0 0 auto;
    }}
    .pyramid img {{
      width: 260px;
      height: auto;
      display: block;
    }}
    .notes {{
      flex: 1;
    }}
    .note-group {{
      margin-bottom: 20px;
    }}
    .note-group h3 {{
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #555;
      margin: 0 0 6px;
      border-bottom: 1px solid #eee;
      padding-bottom: 4px;
    }}
    .note-group ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .note-group li {{
      font-size: 0.95rem;
      line-height: 1.6;
    }}
    @media print {{
      body {{ padding: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="intro">
    <p>Bonjour,</p>
    <p>Voici un récapitulatif de votre formule.</p>
  </div>

  <p class="profile-name">{profile}</p>
  <p class="description">{description}</p>

  <div class="recap">
    <div class="pyramid">{img_tag}</div>
    <div class="notes">{notes_html}</div>
  </div>
</body>
</html>"""


def generate_mail_html(session_id: str, formula: dict) -> str:
    """Return the HTML for in-browser display (image served via URL)."""
    return _build_html(formula, inline_images=False)


def generate_mail_pdf(session_id: str, formula: dict) -> bytes:
    """Return a PDF binary of the formula."""
    from app.services.pdf_service import generate_formula_pdf
    return generate_formula_pdf(formula)


# ── Données statiques pour le mail de test ────────────────────────────

_MOCK_FORMULA_TEST = {
    "profile": "Cosy",
    "description": "Un parfum chaleureux et enveloppant, évoquant la douceur du foyer et les soirées apaisantes.",
    "sizes": {
        "30ml": {
            "top_notes": [
                {"name": "Bergamote", "ml": 4.5},
                {"name": "Rose", "ml": 4.0},
                {"name": "Néroli", "ml": 3.5},
            ],
            "heart_notes": [
                {"name": "Jasmin", "ml": 2.5},
                {"name": "Ylang-Ylang", "ml": 2.0},
                {"name": "Géranium", "ml": 2.0},
            ],
            "base_notes": [
                {"name": "Santal blanc", "ml": 5.5},
                {"name": "Cèdre de l'Atlas", "ml": 4.0},
            ],
        }
    },
}

_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "subject": "Votre formule de parfum personnalisée",
        "greeting": "Bonjour,",
        "subtext": "Voici votre formule de parfum personnalisée.",
        "top": "Notes de tête",
        "heart": "Notes de cœur",
        "base": "Notes de fond",
        "goodbye": "Merci pour votre visite. Nous espérons vous retrouver très bientôt pour créer votre prochaine fragrance. À bientôt !",
    },
    "en": {
        "subject": "Your personalized fragrance formula",
        "greeting": "Hello,",
        "subtext": "Here is your personalized fragrance formula.",
        "top": "Top notes",
        "heart": "Heart notes",
        "base": "Base notes",
        "goodbye": "Thank you for your visit. We hope to see you again very soon to create your next fragrance. See you soon!",
    },
}


def _top3_by_ml(notes: list[dict]) -> list[dict]:
    return sorted(notes, key=lambda n: n.get("ml", 0), reverse=True)[:3]


def _render_note_section(title: str, notes: list[dict]) -> str:
    if not notes:
        return ""
    rows = "".join(
        f"<tr>"
        f'<td style="font-size:0.92rem;padding:3px 0;line-height:1.6;">{n["name"]}</td>'
        f'<td style="font-size:0.82rem;color:#bbb;text-align:right;white-space:nowrap;padding-left:16px;">{n["ml"]} ml</td>'
        f"</tr>"
        for n in notes
    )
    return (
        f'<div style="margin-bottom:18px;">'
        f'<p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.09em;color:#aaa;font-weight:bold;'
        f'margin:0 0 7px;padding-bottom:5px;border-bottom:1px solid #efefef;">{title}</p>'
        f'<table style="border-collapse:collapse;width:100%;"><tbody>{rows}</tbody></table>'
        f"</div>"
    )


def _build_formula_html(
    profile: str,
    description: str,
    notes_30ml: dict,
    language: str = "fr",
    image_base_url: str = "",
) -> str:
    """Build the formula email HTML. Image served via URL (no base64 embedding)."""
    labels = _LABELS.get(language, _LABELS["fr"])

    top = _top3_by_ml(notes_30ml.get("top_notes", []))
    heart = _top3_by_ml(notes_30ml.get("heart_notes", []))
    base = _top3_by_ml(notes_30ml.get("base_notes", []))

    notes_html = (
        _render_note_section(labels["top"], top)
        + _render_note_section(labels["heart"], heart)
        + _render_note_section(labels["base"], base)
    )

    content_html = notes_html

    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{profile}</title>
</head>
<body style="font-family:Arial,sans-serif;background:#faf9f7;margin:0;padding:0;color:#333;">
  <div style="max-width:600px;margin:0 auto;background:#ffffff;padding:40px 36px;">
    <p style="font-size:1rem;margin:0 0 4px;">{labels["greeting"]}</p>
    <p style="font-size:1rem;color:#666;margin:0 0 28px;">{labels["subtext"]}</p>
    <p style="font-size:1.4rem;font-weight:bold;letter-spacing:0.04em;margin:0 0 6px;">{profile}</p>
    <p style="font-style:italic;color:#888;font-size:0.92rem;margin:0 0 28px;">{description}</p>
    {content_html}
    <p style="margin-top:40px;padding-top:24px;border-top:1px solid #efefef;font-size:0.88rem;color:#888;font-style:italic;line-height:1.6;">{labels["goodbye"]}</p>
  </div>
</body>
</html>"""


def send_test_mail(to_email: str) -> None:
    """Send a simple test email to verify SMTP connectivity."""
    settings = get_settings()
    if not settings.smtp_user or not settings.smtp_password:
        raise RuntimeError("SMTP is not configured")

    notes_30ml = _MOCK_FORMULA_TEST["sizes"]["30ml"]
    html = _build_formula_html(
        profile=_MOCK_FORMULA_TEST["profile"],
        description=_MOCK_FORMULA_TEST["description"],
        notes_30ml=notes_30ml,
        language="fr",
    )

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = "[Lylo] Test — Votre formule de parfum"
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = to_email

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(msg["From"], to_email, msg.as_string())


def _build_internal_html(formula: dict) -> str:
    """Build a complete internal recap email with all notes, booster and all sizes."""
    profile = formula.get("profile", "")
    description = formula.get("description", "")
    formula_type = formula.get("formula_type", "")
    sizes = formula.get("sizes", {})

    sizes_html = ""
    for size_label in ("10ml", "30ml", "50ml"):
        size_data = sizes.get(size_label)
        if not size_data:
            continue

        def notes_rows(notes: list[dict]) -> str:
            return "".join(
                f"<tr>"
                f'<td style="padding:3px 8px 3px 0;font-size:0.9rem;">{n["name"]}</td>'
                f'<td style="padding:3px 0;font-size:0.9rem;color:#555;text-align:right;">{n["ml"]} ml</td>'
                f"</tr>"
                for n in notes
            )

        sections = [
            ("Notes de tête", size_data.get("top_notes", [])),
            ("Notes de cœur", size_data.get("heart_notes", [])),
            ("Notes de fond", size_data.get("base_notes", [])),
            ("Booster", size_data.get("boosters", [])),
        ]

        sections_html = ""
        for title, notes in sections:
            if not notes:
                continue
            sections_html += (
                f'<p style="margin:12px 0 4px;font-size:0.75rem;text-transform:uppercase;'
                f'letter-spacing:0.08em;color:#aaa;font-weight:bold;">{title}</p>'
                f'<table style="border-collapse:collapse;width:100%;"><tbody>'
                f'{notes_rows(notes)}'
                f"</tbody></table>"
            )

        sizes_html += (
            f'<div style="margin-bottom:24px;padding:16px;background:#f9f9f9;border-radius:6px;">'
            f'<p style="margin:0 0 10px;font-weight:bold;font-size:1rem;">{size_label}</p>'
            f"{sections_html}"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <title>[Interne] {profile}</title>
</head>
<body style="font-family:Arial,sans-serif;background:#faf9f7;margin:0;padding:0;color:#333;">
  <div style="max-width:600px;margin:0 auto;background:#ffffff;padding:40px 36px;">
    <p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:#bbb;margin:0 0 16px;">
      Récapitulatif interne
    </p>
    <p style="font-size:1.4rem;font-weight:bold;letter-spacing:0.04em;margin:0 0 4px;">{profile}</p>
    <p style="font-style:italic;color:#888;font-size:0.92rem;margin:0 0 6px;">{description}</p>
    <p style="font-size:0.82rem;color:#aaa;margin:0 0 28px;">Type : {formula_type}</p>
    {sizes_html}
  </div>
</body>
</html>"""


def send_internal_formula_mail(to_email: str, session_id: str, formula: dict) -> None:
    """Send a complete internal recap email with all notes and all sizes."""
    settings = get_settings()
    if not settings.smtp_host or not settings.smtp_user:
        raise RuntimeError("SMTP is not configured")

    html = _build_internal_html(formula)

    session_meta = session_store.get_session_meta(session_id)
    language = session_meta.get("language", "fr") if session_meta else "fr"
    profile = formula.get("profile", "formule")

    subject = f"[Lylo Interne] {profile} — Fiche complète"

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = to_email

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(msg["From"], to_email, msg.as_string())


def send_mail(to_email: str, session_id: str, formula: dict) -> None:
    """Send the formula email to the user."""
    settings = get_settings()
    if not settings.smtp_host or not settings.smtp_user:
        raise RuntimeError("SMTP is not configured")

    session_meta = session_store.get_session_meta(session_id)
    language = session_meta.get("language", "fr") if session_meta else "fr"
    labels = _LABELS.get(language, _LABELS["fr"])

    notes_30ml = formula.get("sizes", {}).get("30ml", {})
    html = _build_formula_html(
        profile=formula.get("profile", ""),
        description=formula.get("description", ""),
        notes_30ml=notes_30ml,
        language=language,
        image_base_url=settings.backend_url,
    )

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = labels["subject"]
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = to_email

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(msg["From"], to_email, msg.as_string())


def send_formula_mail_stateless(to_email: str, formula: dict, language: str = "fr") -> None:
    """Envoie le mail de formule sans session — pour le mode batch."""
    settings = get_settings()
    if not settings.smtp_host or not settings.smtp_user:
        raise RuntimeError("SMTP is not configured")

    labels = _LABELS.get(language, _LABELS["fr"])
    notes_30ml = formula.get("sizes", {}).get("30ml", {})
    html = _build_formula_html(
        profile=formula.get("profile", ""),
        description=formula.get("description", ""),
        notes_30ml=notes_30ml,
        language=language,
        image_base_url=settings.backend_url,
    )

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = labels["subject"]
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = to_email

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(msg["From"], to_email, msg.as_string())

    internal_email = settings.internal_email
    if not internal_email:
        return
    for email in [e.strip() for e in internal_email.split(",") if e.strip()]:
        html_internal = _build_internal_html(formula)
        profile = formula.get("profile", "formule")
        msg_int = MIMEText(html_internal, "html", "utf-8")
        msg_int["Subject"] = f"[Lylo Interne] {profile} — Fiche complète"
        msg_int["From"] = settings.smtp_from or settings.smtp_user
        msg_int["To"] = email
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as srv:
            srv.starttls()
            srv.login(settings.smtp_user, settings.smtp_password)
            srv.sendmail(msg_int["From"], email, msg_int.as_string())
