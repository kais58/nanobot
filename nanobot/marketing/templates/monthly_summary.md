# Monatsbericht: {{ month_name }} {{ year }}

## Gesamtstatistik

- Erkannte Signale: **{{ total_signals }}**
- Erstellte Empfehlungen: **{{ total_recommendations }}**
- Genehmigte Kontaktierungen: **{{ approved_count }}**
- Versendete E-Mails: **{{ sent_count }}**
- Lead-Konversionen: **{{ conversions }}**

## Signal-Trends

{% for type_name, count in signal_trends.items() %}
- {{ type_name }}: {{ count }}
{% endfor %}

## Lead-Pipeline

- Hot (>70%): **{{ hot_count }}** Leads
- Warm (40-70%): **{{ warm_count }}** Leads
- Cold (<40%): **{{ cold_count }}** Leads

## Top-Servicegebiete

{% for area, count in top_services.items() %}
{{ loop.index }}. {{ area }}: {{ count }} Empfehlungen
{% endfor %}

---
*Generiert: {{ generated_at }}*
