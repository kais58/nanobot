# Wochenbericht: {{ week_start }} bis {{ week_end }}

## Zusammenfassung

- Neue Signale: **{{ signal_count }}**
- Neue Empfehlungen: **{{ recommendation_count }}**
- Versendete Kontaktierungen: **{{ sent_count }}**
- Konvertierte Leads: **{{ converted_count }}**

## Signale nach Typ

{% for type_name, count in signal_types.items() %}
- {{ type_name }}: {{ count }}
{% endfor %}

## Top-Leads

{% for lead in top_leads[:10] %}
### {{ loop.index }}. {{ lead.company_name }}
- **Score:** {{ "%.0f" | format(lead.total_score * 100) }}%
- **Tier:** {{ lead.tier }}
- **Signale:** {{ lead.signal_count }}
- **Empfohlener Berater:** {{ lead.recommended_services|join(', ') or "nicht zugewiesen" }}
{% endfor %}

## Empfehlungsstatus

| Status | Anzahl |
|--------|--------|
{% for status, count in recommendation_stats.items() %}
| {{ status }} | {{ count }} |
{% endfor %}

---
*Generiert: {{ generated_at }}*
