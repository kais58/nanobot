# Tagesbrief: {{ date }}

## Neue Signale ({{ signals | length }})

{% for signal in signals %}
### {{ signal.company_name }} - {{ signal.signal_type }}
- **Titel:** {{ signal.title }}
- **Relevanz:** {{ "%.0f" | format(signal.relevance_score * 100) }}%
- **K&P Match:** {{ signal.kp_service_match or "nicht zugeordnet" }}
- **Quelle:** [{{ signal.source_name }}]({{ signal.source_url }})
- **Erkannt:** {{ signal.detected_at[:10] }}
{% endfor %}

## Offene Empfehlungen ({{ recommendations | length }})

{% for rec in recommendations %}
- **{{ rec.company_name }}** - {{ rec.service_area }} ({{ rec.consultant_name or "nicht zugewiesen" }})
{% endfor %}

---
*Generiert: {{ generated_at }}*
